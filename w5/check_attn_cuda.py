"""Exit 0 only if attn-cuda imports AND its kernel runs on this GPU (catches arch mismatch)."""
import sys

try:
    import torch
    import attn_cuda

    if not attn_cuda.is_available():
        raise RuntimeError("attn_cuda.is_available() is False")
    q, k, v = (torch.randn(1, 16, 64, 64, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    out, _, colsum = attn_cuda.flash_attn_bias_colsum_bhnd(q, k, v, softmax_scale=0.125, return_colsum=True, subsample_ratio=1.0)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=0.125)
    torch.cuda.synchronize()
    err = (out.float() - ref.float()).abs().max().item()
    if not err < 5e-2:
        raise RuntimeError(f"kernel output mismatch vs SDPA (max abs err {err:.3g})")
    print(f"attn-cuda kernel OK (max abs err vs SDPA {err:.2e})")
except Exception as exc:  # noqa: BLE001
    print(f"attn-cuda check failed: {str(exc).splitlines()[0]}")
    sys.exit(1)
