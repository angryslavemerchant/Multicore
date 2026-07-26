# Offer selection judgment guide

For the agent operating `vast/launch.py`: do NOT blindly auto-pick. Run
`launch.py search`, look at the full table, choose an offer with the criteria
below, then `launch.py launch --offer <ID>`.

## User briefing (2026-07-17) — authoritative, supersedes GPU-class rules below

1. **GPU profiles (`launch.py --profile {5090,6000,b200}`)** — each sets
   gpu_name, price cap, and the matched training recipe (explicit
   `--train-args` replaces the recipe wholesale); pick per run by what
   the user wants that day:
   - **`5090` (default, the workhorse — overnight runs). Hard cap
     $0.38/hr** (user-set; floor ~$0.30, so shop the $0.30–0.38 band).
     Recipe: batch 256, lr 7.5e-4 (linearly rescaled from 1024/3e-3),
     `--data ram` with the blob in **system RAM** (25 GB does not fit
     32 GB VRAM — never `--data_device cuda` here), `--compile_mode
     default`, `--checkpoint_rounds 3` (~15 GB for loop-R7). Expect ~2x
     PRO-6000 epoch time (~160 s for an R7-class run).
   - **`6000` (RTX PRO 6000 WS, 96 GB) — fast same-day results when the
     user is optimistic about an idea.** Cap $1.2/hr. The proven recipe:
     batch 1024, lr 3e-3, `--data ram --data_device cuda`, ck3.
   - **`b200` — occasional very fast runs, user-approved only.** Profile
     cap $8/hr but the market floor hovers near it (~$6.9/hr at
     briefing) — confirm price with the user first. Batch 1024, no
     checkpointing (180 GB).
2. **CPU rules relax on the 5090 class**: that market is nearly all EPYC
   hosts, and with the RAM-blob loader (GPU-side aug, torch-only) training
   is GPU-bound — EPYC is fine now. Still avoid the genuine toasters
   (6-core Ryzen 3600 / 7500F class). **cpu_ram ≥ 48 GB is a hard
   requirement on every profile** (25 GB blob builds/lives in system RAM).

## User briefing (2026-07-14) — GPU-class rules superseded above

1. **GPU classes: RTX 6000 Blackwell (`RTX_PRO_6000_WS`) only**, plus the
   occasional **B200** rapid run when the user asks for one.
   (`RTX_6000Ada` is the older Ada card — NOT what is meant.)
2. **Hard price cap: $1.20/hr.** Market note at briefing time: RTX PRO 6000
   WS floors around $0.81/hr (22 offers); B200 floors at ~$6.25/hr, i.e.
   above the cap — a B200 run requires the user explicitly approving a
   temporary cap override.
3. **Reliability score: ignore it.** (Empirically 0.996 hosts still dropped
   contracts; the score doesn't predict the failures that matter.)
4. **CPU preference dominates everything, including location**: consumer
   CPUs (Ryzen, Intel Core) over server CPUs (EPYC, Xeon, Threadripper) —
   unless the consumer chip is an absolute toaster (ancient/low-end, e.g. an
   old 6-core Ryzen 3600-class part). Rule of thumb from the user: an 8-core
   modern consumer CPU beats a 32-core Threadripper more often than not.
5. **Location**: prefer North America over elsewhere; prefer Canada over the
   US. Subordinate to the CPU preference.
6. **Container/disk size: 80 GB.**
7. **Template (user-specified): the official "PyTorch (Vast)" template**,
   template_id `2ad6d615db5927a06fef0c9cd51d77c4` — replicated in launch.py
   from the CLI command the Vast console generates: image
   `vastai/pytorch:@vastai-automatic-tag` plus the template's port/env
   config (portal, jupyter, tensorboard), launched `--jupyter --ssh
   --direct`, with the template's `entrypoint.sh` kept in the foreground
   and our provisioning chained in the background. Its DockerHub
   description text is outdated boilerplate (mentions CUDA 10) — ignore;
   the image is auto-updated and Blackwell-capable. Deviation from the
   template: disk 80 GB (not 16) per the user's briefing. Do not
   substitute other images without asking the user.

## Standing criteria (from operational experience)

- **Never the cheapest offers in the list** — the bottom of the price range
  over-samples broken/flaky hosts (3 consecutive bottom-price hosts failed
  to boot on day one). Shop the middle of the band.
- **On retry after a failure, always switch physical machine** (`m<id>` in
  search output) — same machine = same problem.
- **PCIe**: gen4 x8+ preferred; gen3 x8 acceptable (DALI overlaps H2D with
  compute); below that, skip.
- **Internet down**: ≥500 Mbps required (dataset ~13 GB); ≥1 Gbps preferred.

## Known-good machines

- machine 32649 — Alberta, Ryzen 9 9950X3D + RTX PRO 6000 WS, ~$1.10/hr.
  Ran the full smoke pipeline flawlessly 2026-07-15: boot→gate in ~1 min,
  best fleet benchmark yet (416 bf16 TFLOPS, 36.9 GB/s H2D, 4.2 GB/s disk,
  15.8k jpeg/s), full smoke boot→train→eval→upload in ~10 min. First choice
  when its offers are listed.
- machine 36615 — Alberta, Core Ultra 7 265K + RTX PRO 6000 WS. Benchmarked
  superbly (384 TFLOPS bf16, 12.8k jpeg/s). Was briefly suspected of
  dropping a contract on 2026-07-15 — that instance was actually STOPPED
  MANUALLY BY THE USER; the machine is fine. Good second choice.

## Machines with a caution flag (usable, watch them)

- machine 9105 — EPYC 7551P RTX 5090, Romania. Two flawless runs
  (13-h pixel-ICL sweep 2026-07-20, dino-codebook 6-way same night,
  both verified artifacts), then DROPPED a contract 2026-07-21:
  instance 45439245 accepted and vanished within ~60 s (no container,
  no logs, nothing in wandb). One drop after two clean runs = caution,
  not blacklist. Bonus if revisited: /workspace/data on it caches
  dino_percepts_vits14.pt (skips the encode step).

- machine 69554 — RTX 5090. EXONERATED 2026-07-19: the "dropped contract"
  was a monitoring false alarm — the run had COMPLETED and self-destroyed
  inside the watcher's 5-min poll gap (wandb shows finished + verified
  artifact). Machine is fine. Lesson for watchers: an instance vanishing
  is only a failure if wandb does NOT show the run finished — always
  cross-check before re-renting.

- machine 137683 — Ryzen 9 9950X3D, Germany. Excellent performer (402
  TFLOPS, ran two clean speed tests 2026-07-17), but ONCE spontaneously
  stopped a fresh container ~2 min after boot (intended_status=stopped,
  no error; not user-initiated). `vastai start instance <id>` revived it
  and the run completed normally. If it stops twice, blacklist it.

## Known-bad machines (add as found; also `.vast/blacklist.json`)

- machine 140318 — EPYC 7K62, "(, US)" (unnamed location), RTX 5090.
  Advertised 4192 Mbps; gate MEASURED 40.3 Mbps to HF / 25.1 to Drive
  and self-destroyed, 2026-07-21. Fits the 2026-07-19 California-EPYC
  slow-network pool profile: EPYC 7K62 + blank US location. Treat the
  whole "(, US)" 7K62 family (m144419, m139253, m139921, m140318...)
  as suspect until one passes the gate; the gate is cheap, but two
  gate-fails in a row from the pool = shop elsewhere first.

- machine 43567 — EPYC 9654, Washington US, rel 0.997. Booted into a
  DNS-dead container (status_msg `curl: (6) Could not resolve host:
  cloud.vast.ai`, 16+ min "loading", zero logs — onstart cannot even
  clone). 2026-07-19. High reliability score, broken networking: the
  status_msg field is the tell — check it when logs stay empty.

- machine 12092 — Core Ultra 9 285K, Nevada. Instance never booted (no
  ONSTART after ~7 min, no logs) and vanished from the API; user confirmed
  it never loaded, 2026-07-15.

- machine 48680 — i7-12700KF, Washington. Boots and trains fine, then GPU
  hangs mid-run: DALI decode hit CUDA_ERROR_LAUNCH_TIMEOUT (702) in epoch 4
  of a probe run, process aborted, 2026-07-15
- machine 140634 — Ryzen 7 9700X, South Korea. Zombie boot: API reports
  running/success but zero onstart logs for 3 h and ssh refused on the
  advertised port. (Same machine booted fine earlier that day —
  intermittent.) 2026-07-16
- machine 91308 — EPYC 7742 RTX 5090, California. Zombie boot: stuck in
  "loading" 35 min, zero onstart logs, gpu_util null, no status_msg;
  user saw an error in the console. 2026-07-17
- machine 14825 — docker daemon broken (OCI runtime create failed), 2026-07-14
- machine 9020  — instances wedge in created/stopped, never boot; ignores
  explicit start, 2026-07-14
- machine 34887 — Ryzen 9 9950X 4090, looks great on paper, boots fast, but
  silently DROPS contracts: accepted twice, instance vanished within ~90s
  both times (no status, no logs, no billing), 2026-07-14

## Post-boot judgment

The boot health gate (benchmark vs thresholds.json) handles measurable
sickness automatically. If an instance sits in `created`/null status >10 min
with no logs: check `vastai show instance <id> --raw` → `status_msg`;
destroy, record the machine above, pick a different machine. If the instance
disappears from `show instances` entirely, the host dropped the contract —
same response. Note: current `thresholds.json` GPU floor (90 bf16 TFLOPS) is
a broken-hardware floor, not a Blackwell performance bar — rerun
`launch.py scan` on the Blackwell fleet to calibrate real thresholds.
- machine 58908 — EPYC 7742, "(, US)", RTX 5090. NEW FAILURE MODE
  (2026-07-22): container status_msg "success, running" but ZERO
  onstart/log output for 30+ minutes — a silent zombie billing
  $0.44/hr with nothing verifiably executing. `launch.py logs`
  completely empty (not truncated, not unicode-crashed). Destroyed
  manually. Lesson: "running" status is NOT evidence of progress —
  if logs are empty past ~8 min (boot+deps normally show output by
  then), treat as dead and destroy. Watchers should alarm on
  log-silence, not just on GONE/error markers.
- machine 122781 — EPYC 9454, "(, US)", RTX 5090. Gate self-destroyed
  at the bench stage 2026-07-22 (logs died with the container; cause
  unrecorded, likely network floor). One strike, not blacklisted.

## Known-good machines (recent clean runs)

- m108899 (Ryzen 9950X, NL) — 3 clean provisions 2026-07-21.
- m61489 (i9-14900K, MA, PRO 6000) — fast for python-bound toys.
- m55313 (EPYC 7R32, CA, 5090) — clean stream-world run 2026-07-22.
- m144477 (EPYC 7C13, Estonia, 5090) — clean stream-world run
  2026-07-22, ~0.35 s/step on the d256/6L toy; rel 0.945 was fine.
- machine 67286 (i7-14700K, Sweden, PRO 6000): silent zombie #2 —
  "success, running" 72 min, zero onstart output (2026-07-22).
  rel 0.945-class hosts now 1-for-2 on this failure mode.
- RULE (2026-07-22): OFFER IDS GO STALE once a machine is rented —
  relaunching on a previously-used offer id yields create-failure
  WITH a phantom billing contract (success:False + new_contract).
  Always re-run `search` for fresh ids; always `status` + destroy
  after any failed create. Four phantoms caught today, all this.

## Ops ledger addendum (2026-07-22, rcore campaign)

- **`--thresholds` takes a REPO-ROOT-relative path**: pass
  `vast/thresholds_hf_light.json`, NOT the bare filename. A bare
  filename makes benchmark.py crash (FileNotFoundError), the crash
  exits nonzero, onstart reads it as GATE_FAILED and SELF-DESTROYS a
  perfectly healthy machine. This executed three healthy boxes in 30
  minutes (m61489, m100187, m68306 x1 each) before a KEEP_ALIVE
  diagnostic caught the traceback. When several instances on
  DIFFERENT hosts die at the gate in a row, suspect the gate, not
  the hosts. m61489 remains known-good.
