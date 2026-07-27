"""Micro-batch throughput sweep: find the batch that saturates the card.

There are TWO batch sizes in this training loop and they do different jobs.

  optimizer batch   --batch x --grad-accum x --seq-len tokens. This is the
                    one that changes the MATH: it sets how many tokens each
                    gradient step averages over, and therefore what learning
                    rate is right and what optimisation problem you are
                    solving. It is a scientific choice and is held fixed.
  micro-batch       --batch alone: sequences per forward pass. Pure hardware
                    throughput. Bigger means larger GEMMs and fewer kernel
                    launches per token, until the activations stop fitting.
                    Changing it while holding --batch x --grad-accum constant
                    is mathematically a no-op (tests/test_vocab.py::ACC) and
                    can be tuned freely.

So "saturate the GPU" means: sweep the micro-batch, hold the optimizer batch
fixed, take the fastest one that fits with headroom.

Synthetic tokens, so this needs no data cache and can run the minute an
instance boots — the answer transfers because throughput does not depend on
which token ids are in the batch.

  python scripts/bench_batch.py --preset ref_dense_130m --seq-len 4096 \
      --micro 4,8,12,16,24,32 --step-tokens 262144 --compile \
      --total-tokens 1.08e9
"""
import argparse, contextlib, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from core import SWTransformer
from core.losses import ce_chunk_default
from m5_arch import (compile_dynamic, flops_per_token,
                     flops_per_token_executed, presets, train_loss)


def ddp_setup():
    """(world_size, rank, local_rank). Single-process unless under torchrun."""
    ws = int(os.environ.get("WORLD_SIZE", "1"))
    if ws == 1:
        return 1, 0, 0
    import torch.distributed as dist
    local = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)
    dist.init_process_group("nccl")
    return ws, dist.get_rank(), local


def unwrap(m):
    """Strip torch.compile and DDP wrappers, in either order."""
    for attr in ("_orig_mod", "module", "_orig_mod"):
        m = getattr(m, attr, m)
    return m


def measure(cfg, preset, T, micro, accum, device, compile_on, steps, warmup,
            ce_chunk, lr=6e-4, compile_mode=None, world=1, dynamic=None):
    """Steady-state seconds per OPTIMIZER step at this micro-batch."""
    torch.manual_seed(0)
    torch._dynamo.reset()          # each config gets its own static compile
    model = SWTransformer(cfg).to(device)
    if world > 1:
        # DDP first, then compile: dynamo's DDPOptimizer splits the graph at
        # the gradient bucket boundaries so the all-reduce of one bucket
        # overlaps the backward of the next. Compiling the inner module and
        # wrapping afterwards loses that overlap.
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[torch.cuda.current_device()],
                    gradient_as_bucket_view=True)
    compiled_ok = False
    if compile_on:
        # Force the lazy compile here so a host whose triton cannot build
        # (measured: a 5090 whose gcc could not link libcuda.so.1) degrades to
        # an eager row rather than taking the whole sweep down. The row is
        # still a measurement -- it is just labelled as eager, because an
        # eager number and a compiled number are not comparable.
        try:
            kw = {"dynamic": dynamic}
            if compile_mode:
                kw["mode"] = compile_mode
            c = torch.compile(model, **kw)
            with torch.no_grad(), torch.autocast(
                    device, dtype=torch.bfloat16, enabled=(device == "cuda")):
                c(torch.zeros(micro, T, dtype=torch.long, device=device))
            model, compiled_ok = c, True
        except Exception as e:
            print(f"    [compile] unavailable ({type(e).__name__}); "
                  f"this row is EAGER", flush=True)
    raw = unwrap(model)
    head_w = raw.head.weight
    try:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1,
                                betas=(0.9, 0.95), fused=(device == "cuda"))
    except (RuntimeError, ValueError):
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1,
                                betas=(0.9, 0.95))
    model.train()
    g = torch.Generator(device="cpu").manual_seed(0)
    idx = torch.randint(0, cfg.vocab_size, (micro, T + 1), generator=g).to(device)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_warm = time.time()
    for it in range(warmup + steps):
        if it == warmup:                       # start the clock after compile
            if device == "cuda":
                torch.cuda.synchronize()
            compile_s = time.time() - t_warm
            t0 = time.time()
        opt.zero_grad(set_to_none=True)
        for a in range(accum):
            # no_sync on every micro-step but the last: without it DDP
            # all-reduces `accum` times per optimizer step instead of once,
            # which at accum 16 is 16x the communication for no benefit and
            # would make any scaling number here meaningless.
            last = (a == accum - 1)
            ctx = (model.no_sync() if (world > 1 and not last)
                   else contextlib.nullcontext())
            with ctx:
                with torch.autocast(device, dtype=torch.bfloat16,
                                    enabled=(device == "cuda")):
                    ce, aux, _ = train_loss(model, head_w, idx, ce_chunk)
                ((ce + aux) / accum).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    if device == "cuda":
        torch.cuda.synchronize()
    dt = (time.time() - t0) / steps
    peak = (torch.cuda.max_memory_allocated() / 2 ** 30
            if device == "cuda" else 0.0)
    del model, opt, idx
    if device == "cuda":
        torch.cuda.empty_cache()
    return dt, peak, compile_s, compiled_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="ref_dense_130m")
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--micro", default="4,8,12,16,24,32",
                    help="micro-batch sizes to sweep")
    ap.add_argument("--step-tokens", type=int, default=262144,
                    help="tokens per OPTIMIZER step, held fixed across the "
                         "sweep so every row is the same experiment")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--total-tokens", type=float, default=1.08e9,
                    help="budget to project a wall-clock estimate for")
    ap.add_argument("--mem-headroom", type=float, default=0.85,
                    help="fraction of VRAM a config may use and still be "
                         "recommended; real runs fragment more than a "
                         "4-step benchmark does")
    # --- spending spare VRAM. The 130M model peaks at 11.7 GB of 32 (5090) or
    # 96 (RTX PRO 6000), so there is headroom, and headroom is a currency:
    #   ce-chunk 0  stop chunking the cross-entropy. Chunking exists to bound
    #               the 824 MB/sequence fp32 logits, and it costs one extra
    #               head matmul in backward (2*d*V = 51.5M FLOPs/token, ~5% of
    #               a step). With room to hold the logits outright, that
    #               recompute is pure waste. Needs ~8 GB more at micro 4.
    #   max-autotune    inductor benchmarks candidate kernel configs instead
    #               of picking heuristically. Costs minutes of compile and
    #               autotuning workspace; usually pays on GEMM-heavy graphs.
    #   reduce-overhead CUDA graphs: capture the step, remove per-kernel
    #               launch cost. Costs static buffers. Matters most at small
    #               micro-batches, which is exactly where we landed.
    ap.add_argument("--ce-chunk", type=int, default=None,
                    help="rows per CE chunk; 0 disables chunking entirely")
    ap.add_argument("--compile-mode", default=None,
                    choices=("default", "max-autotune", "reduce-overhead"))
    ap.add_argument("--dynamic", choices=("auto", "true", "false"),
                    default="auto")
    # Capacity is a DIRECT MULTIPLIER on expert FLOPs, not just a memory
    # bound: the packed buffer is C * N/M slots per expert and the batched
    # matmul runs over all of them, padding included (`valid` is only applied
    # after the expert block, at scatter time). At C=2.5 with balanced routing
    # 60% of every expert GEMM is arithmetic on padding. Sweepable here so the
    # cost is measured rather than argued about.
    ap.add_argument("--capacity", type=float, default=None,
                    help="override every core's capacity_factor")
    ap.add_argument("--profile", action="store_true",
                    help="torch.profiler one step and print the top CUDA ops. "
                         "Replaces guessing about where the routed overhead "
                         "goes -- three theories so far (micro-batch, padded "
                         "GEMMs, banded attention) and only the second was "
                         "even partly right.")
    ap.add_argument("--label", default="", help="tag for this row in the JSON")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    world, rank, local = ddp_setup()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision("high")
    T = args.seq_len

    def say(*a, **k):
        if rank == 0:
            print(*a, **k)

    cfg = presets(T)[args.preset]
    if args.capacity is not None and cfg.cores:
        from dataclasses import replace as _replace
        cfg.cores = [_replace(c, capacity_factor=args.capacity)
                     for c in cfg.cores]
    ce_chunk = (ce_chunk_default(cfg.vocab_size) if args.ce_chunk is None
                else args.ce_chunk)
    probe = SWTransformer(cfg)
    fpt = flops_per_token(probe, cfg, T)
    n_params = probe.num_params()
    del probe
    total_vram = (torch.cuda.get_device_properties(0).total_memory / 2 ** 30
                  if device == "cuda" else 0.0)
    name = (torch.cuda.get_device_name(0) if device == "cuda" else "cpu")
    say(f"{args.preset} on {world}x {name} ({total_vram:.0f} GB each): "
        f"{n_params/1e6:.1f}M params, {fpt/1e6:.1f}M FLOPs/token fwd, "
        f"T={T}, optimizer batch {args.step_tokens/1e3:.0f}k tokens, "
        f"compile={args.compile} mode={args.compile_mode or 'default'} "
        f"ce_chunk={ce_chunk} capacity="
        f"{cfg.cores[0].capacity_factor if cfg.cores else '-'} {args.label}",
        flush=True)

    dyn = (compile_dynamic(cfg) if args.dynamic == "auto"
           else {"true": True, "false": False}[args.dynamic])
    ex_f, sem_f = flops_per_token_executed(SWTransformer(cfg), cfg, T)
    say(f"  dynamic={dyn}  semantic {sem_f/1e6:.1f}M FLOPs/tok, "
        f"EXECUTED {ex_f/1e6:.1f}M ({ex_f/sem_f:.2f}x -- capacity "
        f"padding + band overscan)", flush=True)

    rows = []
    for micro in [int(x) for x in args.micro.split(",")]:
        # The optimizer batch is held fixed as GPUs are added, so accum falls
        # as world grows. That is STRONG scaling and it is the demanding test:
        # the per-rank step shrinks while the all-reduce stays the same size,
        # so comms overhead is maximally exposed. Weak scaling (fix accum, let
        # the batch grow with world) hides it -- and is what we would actually
        # run, since the reference's batch is 16x ours.
        if args.step_tokens % (micro * T * world):
            say(f"  micro {micro:>3}: skipped — {micro}x{T}x{world} does not "
                f"divide the {args.step_tokens}-token optimizer batch",
                flush=True)
            continue
        accum = args.step_tokens // (micro * T * world)
        try:
            dt, peak, cs, cok = measure(cfg, args.preset, T, micro, accum,
                                        device, args.compile, args.steps,
                                        args.warmup, ce_chunk,
                                        compile_mode=args.compile_mode,
                                        world=world, dynamic=dyn)
        except Exception as e:
            if "out of memory" not in str(e).lower() and not isinstance(
                    e, torch.OutOfMemoryError):
                print(f"  micro {micro:>3} (accum {accum:>2}): FAILED "
                      f"({type(e).__name__}: {str(e)[:120]})", flush=True)
                torch.cuda.empty_cache()
                continue
            say(f"  micro {micro:>3} (accum {accum:>2}): OOM", flush=True)
            torch.cuda.empty_cache()
            continue
        tps = args.step_tokens / dt
        # 3x forward is the standard fwd+bwd rule; the chunked CE recomputes
        # the head in backward, which this ignores, so MFU is a slight
        # under-estimate rather than a flattering one
        rows.append({"micro": micro, "accum": accum, "s_per_step": dt,
                     "tok_per_s": tps, "peak_gb": peak, "compile_s": cs,
                     "compiled": cok,
                     "hours": args.total_tokens / tps / 3600})
        say(f"  micro {micro:>3} (accum {accum:>2}): {dt:7.3f} s/step  "
              f"{tps:>9,.0f} tok/s  {peak:5.1f} GB peak  "
              f"{rows[-1]['hours']:5.2f} h for {args.total_tokens/1e9:.2f}B  "
              f"({'compiled' if cok else 'EAGER'}, warmup {cs:.0f}s)",
              flush=True)

    if args.profile and rows:
        m = rows[0]["micro"]
        say(f"\n--- profiling micro {m} (accum {rows[0]['accum']}) ---",
            flush=True)
        from torch.profiler import profile, ProfilerActivity
        model = SWTransformer(cfg).to(device)
        if args.compile:
            model = torch.compile(model, dynamic=True if cfg.cores else None)
        raw = unwrap(model)
        opt = torch.optim.AdamW(model.parameters(), lr=6e-4)
        idx = torch.randint(0, cfg.vocab_size, (m, T + 1)).to(device)
        for _ in range(3):                      # compile + warm the allocator
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device, dtype=torch.bfloat16,
                                enabled=(device == "cuda")):
                ce, aux, _ = train_loss(model, raw.head.weight, idx, ce_chunk)
            (ce + aux).backward()
            opt.step()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device, dtype=torch.bfloat16,
                                enabled=(device == "cuda")):
                ce, aux, _ = train_loss(model, raw.head.weight, idx, ce_chunk)
            (ce + aux).backward()
            opt.step()
            torch.cuda.synchronize()
        say(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=25),
            flush=True)

    if not rows:
        sys.exit("no configuration ran")
    fits = [r for r in rows
            if not total_vram or r["peak_gb"] <= args.mem_headroom * total_vram]
    best = max(fits or rows, key=lambda r: r["tok_per_s"])
    slowest = min(rows, key=lambda r: r["tok_per_s"])
    say(f"\nBEST micro-batch {best['micro']} (grad-accum {best['accum']}): "
          f"{best['tok_per_s']:,.0f} tok/s, {best['peak_gb']:.1f} GB of "
          f"{total_vram:.0f}, {best['hours']:.2f} h for "
          f"{args.total_tokens/1e9:.2f}B tokens. "
          f"{best['tok_per_s']/slowest['tok_per_s']:.2f}x the worst row.",
          flush=True)
    say("BENCH_JSON " + json.dumps({"preset": args.preset, "seq_len": T,
                                      "gpu": name, "vram_gb": total_vram, "world_size": world,
                                      "step_tokens": args.step_tokens,
                                      "flops_per_token": fpt,
                                      "params": n_params,
                                      "ce_chunk": ce_chunk,
                                      "compile_mode": args.compile_mode,
                                      "label": args.label,
                                      "best": best, "rows": rows}), flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"best": best, "rows": rows}, f, indent=2)


if __name__ == "__main__":
    main()
