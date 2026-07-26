# Multicore

Sparse, causal, KV-cache-safe attention side-modules ("cores") on a
sliding-window transformer. Cores admit a threshold-selected subset of tokens
into a K-slot FIFO and let them attend privately — the model's only
long-range channel, and (at scale) a home for conditional parameters.

Full design, invariants, and milestones: [CORE_ROUTING_PLAN.md](CORE_ROUTING_PLAN.md).

## Layout

- `core/` — model library: gating, resident-set logic, core module, base
  transformer, MQAR task.
- `tests/test_invariants.py` — the M0/M1/M2 gates (bit-identity, spec vs
  implementation, prefill vs incremental decode). Run on every change:
  `python tests/test_invariants.py`.
- `scripts/m3_mechanism.py` — the load-bearing experiment: frozen
  sliding-window base + one core on associative recall, vs controls that fail
  by construction.
- `scripts/m4_selection.py` — what did the gate learn to admit?
- `vast/` — GPU rental automation (see `vast/README.md`).

## M3 quickstart

```bash
python scripts/m3_mechanism.py --stage all --variant core     # the mechanism
python scripts/m3_mechanism.py --stage core --variant adapter # control
python scripts/m3_mechanism.py --stage core --variant none    # frozen base
```

Success = `acc_gap_64+` buckets high for `core`, at chance for both controls.
