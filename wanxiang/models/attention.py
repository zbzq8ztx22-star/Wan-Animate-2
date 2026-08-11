import torch
import math
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention as flex_attention_func


try:
    from flash_attn_interface import flash_attn_varlen_func

    FLASH_VER = 3

except ModuleNotFoundError:
    try:
        from flash_attn import flash_attn_varlen_func

        FLASH_VER = 2
    except ModuleNotFoundError:
        flash_attn_varlen_func = None  # in compatible with CPU machines
        FLASH_VER = None

print(f"[PreInfo] Use flash attention={FLASH_VER}")

flex_attention_func = torch.compile(
    flex_attention_func,
    dynamic=False,             
    # max-autotune tries Hopper-sized Triton configurations on Blackwell
    # workstation GPUs.  Some of those exceed the SM 12.0 shared-memory
    # limit and can spend minutes benchmarking unusable kernels.
    mode="default",
    fullgraph=True,     
    backend="inductor"
)

__all__ = [
    'flash_attention',
    'flex_attention'
]

def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.0,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == "cuda" and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor([lq] * b, dtype=torch.int32).to(
            device=q.device, non_blocking=True
        )
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor([lk] * b, dtype=torch.int32).to(
            device=k.device, non_blocking=True
        )
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale
    # apply attention
    if FLASH_VER is None:
        # Native PyTorch fallback for platforms where flash-attn has no
        # compatible wheel (notably Windows and Blackwell GPUs).  SDPA still
        # dispatches to an optimized CUDA kernel when the inputs permit it.
        outputs = []
        for batch_idx in range(b):
            q_len = int(q_lens[batch_idx])
            k_len = int(k_lens[batch_idx])
            q_item = q[sum(int(v) for v in q_lens[:batch_idx]):sum(int(v) for v in q_lens[:batch_idx + 1])]
            k_item = k[sum(int(v) for v in k_lens[:batch_idx]):sum(int(v) for v in k_lens[:batch_idx + 1])]
            v_item = v[sum(int(v) for v in k_lens[:batch_idx]):sum(int(v) for v in k_lens[:batch_idx + 1])]

            q_item = q_item.transpose(0, 1).unsqueeze(0)
            k_item = k_item.transpose(0, 1).unsqueeze(0)
            v_item = v_item.transpose(0, 1).unsqueeze(0)
            item = F.scaled_dot_product_attention(
                q_item,
                k_item,
                v_item,
                dropout_p=dropout_p,
                is_causal=causal,
                scale=softmax_scale,
                enable_gqa=q_item.size(1) != k_item.size(1),
            ).squeeze(0).transpose(0, 1)
            if q_len < lq:
                item = F.pad(item, (0, 0, 0, 0, 0, lq - q_len))
            outputs.append(item)
        x = torch.stack(outputs)
    elif FLASH_VER == 3:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens])
            .cumsum(0, dtype=torch.int32)
            .to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens])
            .cumsum(0, dtype=torch.int32)
            .to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic,
        )[0].unflatten(0, (b, lq))
    else:
        assert FLASH_VER == 2
        x = flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens])
            .cumsum(0, dtype=torch.int32)
            .to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens])
            .cumsum(0, dtype=torch.int32)
            .to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
        ).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


# Author: Guangyuan Wang
def flex_attention(
    q, 
    k, 
    v,
    q_lens=None,
    k_lens=None, 
    block_mask=None, 
    kernel_options=None,
    dtype=torch.bfloat16,
    score_mod=None
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda'
    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype
    def half(x): return x if x.dtype in half_dtypes else x.to(dtype)

    assert lq % 128 == 0, "q_len must be divisible by 128."
    assert lk % 128 == 0, "k_len must be divisible by 128."

    # preprocess query
    if q_lens is None:
        q = half(q)
    else:
        q = half(q)
        assert q_lens.max() == q_lens.min(), 'varlen of query is not supported'

    # preprocess key, value
    if k_lens is None:
        k, v = half(k), half(v)
    else:
        k, v = half(k), half(v)
        assert k_lens.max() == k_lens.min(), 'varlen of key is not supported'

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if kernel_options is None:
        # Conservative inference configuration for Windows / SM 12.0.  The
        # 64x64, one-stage kernel stays below the per-block shared-memory
        # ceiling while preserving the custom sparse mask and score modifier.
        kernel_options = {
            "BLOCK_M": 64,
            "BLOCK_N": 64,
            "num_warps": 4,
            "num_stages": 1,
        }


    x = flex_attention_func(
        query=q.transpose(2,1),
        key=k.transpose(2,1),
        value=v.transpose(2,1),
        block_mask=block_mask,
        kernel_options=kernel_options,
        score_mod=score_mod
    ).transpose(2, 1)
    
    return x.type(out_dtype)
