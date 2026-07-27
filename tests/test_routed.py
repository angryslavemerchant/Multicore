"""Invariants for the top-1 recurrent eight-core middle stack."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from core import CoreConfig, ModelConfig, SWTransformer


def routed_cfg(M=4):
    rc = CoreConfig(K=3, d_core=48, n_heads=4, n_core_layers=1,
                    routing="top1_recurrent", n_loops=2, ffn_hidden=64,
                    inter_core_window=8, residual_scale_init=0.1,
                    router_aux_weight=0.01)
    return ModelConfig(vocab_size=37, d_model=48, n_layers=3, n_heads=4,
                       window=8, max_seq_len=64, core_layer=1,
                       cores=[rc] * M)


def test_top1_exact_and_gradients():
    torch.manual_seed(10)
    model = SWTransformer(routed_cfg())
    idx = torch.randint(0, 37, (4, 24))
    logits, aux = model(idx, collect_aux=True)
    masks = torch.stack([a["m"] for a in aux], dim=1)  # (L,M,B,T)
    assert torch.equal(masks.sum(1), torch.ones_like(masks[:, 0], dtype=torch.long))
    assert abs(sum(float(a["rate"]) for a in aux) - 1.0) < 1e-6
    ce = F.cross_entropy(logits.reshape(-1, 37), idx.reshape(-1))
    loss = ce + aux[0]["router_aux_loss"]
    loss.backward()
    routed = model.cores[0]
    assert routed.router_w.grad is not None and routed.router_w.grad.norm() > 0
    assert routed.expert.q_w.grad is not None and routed.expert.q_w.grad.norm() > 0
    with torch.no_grad():
        routed.router_w[1].copy_(routed.router_w[0])
    model.reproject_gates()
    rw = routed.router_w.detach()
    rn = rw / rw.norm(dim=1, keepdim=True)
    off = ~torch.eye(rw.shape[0], dtype=torch.bool)
    assert float((rn @ rn.t()).abs()[off].max()) < 1e-5
    assert routed.expert.o_w.grad is not None and routed.expert.o_w.grad.norm() > 0
    assert routed.expert.f1_w.grad is not None and routed.expert.f1_w.grad.norm() > 0
    print("RT PASSED: exactly one expert/token/loop; router and expert gradients live")


def test_top1_prefill_decode():
    torch.manual_seed(11)
    model = SWTransformer(routed_cfg()).eval()
    idx = torch.randint(0, 37, (3, 31))
    with torch.no_grad():
        full = model(idx)
        cache = model.init_caches(3, "cpu")
        step = torch.stack([model.forward_step(idx[:, t], cache)
                            for t in range(idx.shape[1])], 1)
    err = float((full - step).abs().max())
    assert err < 3e-6, err
    ring = cache["rings"][0]
    assert ring["count"].shape == (2, 4, 3)
    assert torch.equal(ring["count"].sum((1, 2)),
                       torch.full((2,), 3 * 31, dtype=torch.long))
    assert len({id(c) for c in ring["mixer"]}) == 2
    assert all(c["k"].shape[2] == 8 for c in ring["mixer"])
    print(f"RT-CACHE PASSED: prefill == decode ({err:.2e}); loop caches independent")


def test_top1_compute_match():
    from scripts.m5_arch import presets, flops_per_token
    T = 2048
    routed_cfg_ = presets(T)["smoke_cores_top1_loopmix"]
    dense_cfg = presets(T)["smoke_dense_local"]
    routed = SWTransformer(routed_cfg_)
    dense = SWTransformer(dense_cfg)
    rf = flops_per_token(routed, routed_cfg_, T)
    df = flops_per_token(dense, dense_cfg, T)
    rel = abs(rf - df) / df
    assert rel < 0.001, (rf, df, rel)
    assert routed.num_params() > dense.num_params()
    print(f"RT-FLOPS PASSED: routed {rf:,} vs dense {df:,} ({rel:.3%})")


if __name__ == "__main__":
    test_top1_exact_and_gradients()
    test_top1_prefill_decode()
    test_top1_compute_match()
    print("ALL ROUTED GATES PASSED")
