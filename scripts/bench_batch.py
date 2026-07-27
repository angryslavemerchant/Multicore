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
import argparse, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from core import SWTransformer
from m5_arch import (ce_chunk_default, flops_per_token, presets, train_loss)


def measure(cfg, preset, T, micro, accum, device, compile_on, steps, warmup,
            ce_chunk, lr=6e-4, compile_mode=None):
    """Steady-state seconds per OPTIMIZER step at this micro-batch."""
    torch.manual_seed(0)
    torch._dynamo.reset()          # each config gets its own static compile
    model = SWTransformer(cfg).to(device)
    compiled_ok = False
    if compile_on:
        # Force the lazy compile here so a host whose triton cannot build
        # (measured: a 5090 whose gcc could not link libcuda.so.1) degrades to
        # an eager row rather than taking the whole sweep down. The row is
        # still a measurement -- it is just labelled as eager, because an
        # eager number and a compiled number are not comparable.
        try:
            kw = {"dynamic": True if cfg.cores else None}
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
    raw = getattr(model, "_orig_mod", model)
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
        for _ in range(accum):
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
    ap.add_argument("--label", default="", help="tag for this row in the JSON")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision("high")
    T = args.seq_len
    cfg = presets(T)[args.preset]
    ce_chunk = (ce_chunk_default(cfg.vocab_size) if args.ce_chunk is None
                else args.ce_chunk)
    probe = SWTransformer(cfg)
    fpt = flops_per_token(probe, cfg, T)
    n_params = probe.num_params()
    del probe
    total_vram = (torch.cuda.get_device_properties(0).total_memory / 2 ** 30
                  if device == "cuda" else 0.0)
    name = (torch.cuda.get_device_name(0) if device == "cuda" else "cpu")
    print(f"{args.preset} on {name} ({total_vram:.0f} GB): "
          f"{n_params/1e6:.1f}M params, {fpt/1e6:.1f}M FLOPs/token fwd, "
          f"T={T}, optimizer batch {args.step_tokens/1e3:.0f}k tokens, "
          f"compile={args.compile} mode={args.compile_mode or 'default'} "
          f"ce_chunk={ce_chunk} {args.label}", flush=True)

    rows = []
    for micro in [int(x) for x in args.micro.split(",")]:
        if args.step_tokens % (micro * T):
            print(f"  micro {micro:>3}: skipped — {micro}x{T} does not divide "
                  f"the {args.step_tokens}-token optimizer batch", flush=True)
            continue
        accum = args.step_tokens // (micro * T)
        try:
            dt, peak, cs, cok = measure(cfg, args.preset, T, micro, accum,
                                        device, args.compile, args.steps,
                                        args.warmup, ce_chunk,
                                        compile_mode=args.compile_mode)
        except Exception as e:
            if "out of memory" not in str(e).lower() and not isinstance(
                    e, torch.OutOfMemoryError):
                print(f"  micro {micro:>3} (accum {accum:>2}): FAILED "
                      f"({type(e).__name__}: {str(e)[:120]})", flush=True)
                torch.cuda.empty_cache()
                continue
            print(f"  micro {micro:>3} (accum {accum:>2}): OOM", flush=True)
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
        print(f"  micro {micro:>3} (accum {accum:>2}): {dt:7.3f} s/step  "
              f"{tps:>9,.0f} tok/s  {peak:5.1f} GB peak  "
              f"{rows[-1]['hours']:5.2f} h for {args.total_tokens/1e9:.2f}B  "
              f"({'compiled' if cok else 'EAGER'}, warmup {cs:.0f}s)",
              flush=True)

    if not rows:
        sys.exit("no configuration ran")
    fits = [r for r in rows
            if not total_vram or r["peak_gb"] <= args.mem_headroom * total_vram]
    best = max(fits or rows, key=lambda r: r["tok_per_s"])
    slowest = min(rows, key=lambda r: r["tok_per_s"])
    print(f"\nBEST micro-batch {best['micro']} (grad-accum {best['accum']}): "
          f"{best['tok_per_s']:,.0f} tok/s, {best['peak_gb']:.1f} GB of "
          f"{total_vram:.0f}, {best['hours']:.2f} h for "
          f"{args.total_tokens/1e9:.2f}B tokens. "
          f"{best['tok_per_s']/slowest['tok_per_s']:.2f}x the worst row.",
          flush=True)
    print("BENCH_JSON " + json.dumps({"preset": args.preset, "seq_len": T,
                                      "gpu": name, "vram_gb": total_vram,
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
