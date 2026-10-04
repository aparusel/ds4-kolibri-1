#!/usr/bin/env python3
"""Torch port of Aleph Alpha's inference-only Kolibri-1 forward.

This is the phase-2 real-weight referee: the same block semantics as
tests/kolibri1_reference.py, taken from aleph-alpha-inference's kolibri1.py
(Kolibri1DecoderLayer.forward, Kolibri1Attention, sigmoid_logit_add_routing,
Qwen3MoeSparseMoeBlock/Qwen3MoeMLP), expressed with plain torch ops instead of
vLLM primitives. Weights come from a ds4 GGUF — the mini fixture (F32) here,
the real Kolibri-1-Q8 artifact on the CUDA box — so this port and ds4 always
see bit-identical weights.

One decoder layer, exactly as kolibri1.py:

    if residual is None:            # layer 0, per forward call
        residual = hidden_states
        hidden_states = input_layernorm(hidden_states)
    else:                           # fused add-RMSNorm: sublayer outputs join
        hidden_states, residual = input_layernorm(hidden_states, residual)
    hidden_states = self_attn(...)             # GQA + per-head QK norm
    hidden_states = post_attn_norm(...)        # sandwich norm on attention out
    hidden_states, residual = post_attention_layernorm(hidden_states, residual)
    hidden_states = mlp(inp.name)              # ungated shared + weighted routed
    hidden_states = post_ffn_norm(...)         # sandwich norm on MoE out
    [the MoE output joins the residual at the NEXT pre-norm; the model's final
     norm performs the same join for the last layer]

Differences from the NumPy anchor, by design:

- Whole-sequence batched prefill instead of a per-token KV loop. The math is
  identical; only floating-point summation order differs.
- Configurable compute dtype (float32 default, float64 for semantic checks)
  and device (cpu/cuda/mps), because the real-weight runs happen on CUDA.
- Q8_0 tensors dequantize like ds4's dot_q8_0_row: fp16 block scale times the
  int8 quants (fp16->fp32 is exact, so the rendered weights are identical).
- The router path stays fp32 per kolibri1.py (out_dtype=torch.float32,
  gating_output.float()), in both compute dtypes.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kolibri1_reference import load_gguf  # noqa: E402  (same reader, same bytes)


def dequant_q8_0(data, dtype):
    """GGUF Q8_0 flat bytes -> dequanted float ndarray in GGUF element order.

    A Q8_0 tensor is a stream of 34-byte blocks: 2-byte fp16 scale, then 32
    int8 quants. Value = scale * quant, exactly like ds4's dot_q8_0_row (its
    fp16->f32 conversion is an exact widening). Callers reshape with reversed
    dims afterwards, so values land in the same arithmetic positions as the
    anchor's F32 reads.
    """
    grid = np.frombuffer(data, dtype=np.uint8)
    if grid.size % 34:
        raise ValueError(f"q8_0 payload of {grid.size} bytes is not a block multiple")
    grid = grid.reshape(-1, 34)
    scales = grid[:, 0:2].copy().view("<f2").astype(dtype)
    quants = grid[:, 2:34].copy().view(np.int8).astype(dtype)
    return (scales.reshape(-1, 1) * quants).reshape(-1)


class Tensors:
    """Lazy GGUF weights in linear-algebra layout, optionally materialized.

    Dense matrices come out as [out, in] rows and expert stacks as
    [n, out, in] planes (see the anchor's tensor_array docstring). Q8_0 is
    dequanted on demand; F32/F16 are plain reads. Tensors of at most 1 MiB
    (norms, routers, q_norm) are cached; big ones only with --materialize.
    """

    def __init__(self, meta, tensors, blob, data_start, dtype):
        self.meta = meta
        self.tensors = tensors
        self.blob = blob
        self.data_start = data_start
        self.dtype = dtype
        self.cache = {}
        self.materialize = False

    def raw(self, name):
        info = self.tensors[name]
        count = int(np.prod(info["dims"]))
        start = self.data_start + info["offset"]
        if info["type"] == 0:  # F32
            values = np.frombuffer(self.blob, dtype="<f4", count=count,
                                   offset=start).astype(self.dtype)
        elif info["type"] == 1:  # F16
            values = np.frombuffer(self.blob, dtype="<f2", count=count,
                                   offset=start).astype(self.dtype)
        elif info["type"] == 8:  # Q8_0
            nbytes = ((count + 31) // 32) * 34
            raw_bytes = self.blob[start:start + nbytes]
            if len(raw_bytes) != nbytes:
                raise ValueError(f"{name}: truncated q8_0 payload")
            values = dequant_q8_0(raw_bytes, self.dtype)
        else:
            raise ValueError(f"unsupported tensor type {info['type']} for {name}")
        math_view = values.reshape(tuple(reversed(info["dims"])))
        if math_view.size * array_itemsize(info["type"]) <= (1 << 20):
            self.cache[name] = math_view
        return math_view

    def get(self, name):
        stored = self.cache.get(name)
        return stored if stored is not None else self.raw(name)


def array_itemsize(gguf_type):
    return {0: 4, 1: 2}[gguf_type]


def torchify(array, device):
    return torch.from_numpy(np.ascontiguousarray(array)).to(device)


class Layer:
    """One decoder layer's tensor bundle, materialized on demand.

    The routed-expert stacks are the big tensors of a layer (~1.5 GB fp32 on
    the released shape); keeping exactly one layer resident bounds peak
    memory to one layer regardless of device RAM.
    """

    KEYS = (
        "attn_norm", "post_attn_norm", "ffn_norm", "post_ffn_norm",
        "q_norm", "k_norm",
        "q", "k", "v", "o", "gate_inp", "bias",
        "s_gate", "s_up", "s_down", "r_gate", "r_up", "r_down",
    )

    def __init__(self, t: Tensors, index: int, is_full: bool):
        self.t, self.index, self.is_full = t, index, is_full
        self.bundle = None

    def names(self):
        p = f"blk.{self.index}."
        big = [f"{p}attn_{x}.weight" for x in ("q", "k", "v", "output")]
        big.append(f"{p}ffn_gate_inp.weight")
        big.append(f"{p}exp_probs_b.bias")
        big += [f"{p}ffn_{x}_shexp.weight" for x in ("gate", "up", "down")]
        big += [f"{p}ffn_{x}_exps.weight" for x in ("gate", "up", "down")]
        small = [f"{p}attn_norm.weight", f"{p}post_attn_norm.weight",
                 f"{p}ffn_norm.weight", f"{p}post_ffn_norm.weight",
                 f"{p}attn_q_norm.weight", f"{p}attn_k_norm.weight"]
        return small + big

    def load(self, device):
        if self.bundle is None:
            arrays = [self.t.get(n) for n in self.names()]
            self.bundle = dict(zip(self.KEYS,
                                   [torchify(a, device) for a in arrays]))
        return self.bundle

    def free(self):
        self.bundle = None


def rms_norm(x, weight, eps):
    return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps) * weight


def fused_rms_norm(x, residual, weight, eps):
    """vLLM's add-RMSNorm: return (normed(x + residual), x + residual).

    In fp32/f64 both copies of the sum are exact, so this matches the kernel's
    float32 path (it rounds the stored residual back to the input dtype only
    when that dtype is bf16/fp16, which the port never uses).
    """
    total = residual + x
    return rms_norm(total, weight, eps), total


def rotate_half_rope(values, positions, base, head_dim):
    """RoPE over the last head_dim of [..., n, head_dim] rows, NeoX/HF
    rotate-half style, matching the anchor's rope(): pairs (i, i + dim/2),
    angle pos * base ** (-2i/dim)."""
    half = head_dim // 2
    exponent = base ** (-2.0 * torch.arange(half, dtype=values.dtype,
                                            device=values.device) / head_dim)
    angles = (positions.to(values.dtype).unsqueeze(-1) * exponent).unsqueeze(1)
    cos, sin = torch.cos(angles), torch.sin(angles)
    left, right = values[..., :half], values[..., half:]
    return torch.cat((left * cos - right * sin,
                      right * cos + left * sin), dim=-1)


def per_head_rms_norm(x, weight, n_head, head_dim, eps):
    """Per-head QK RMSNorm over the last head_dim, weight broadcast per head."""
    grouped = x.view(*x.shape[:-1], n_head, head_dim)
    return rms_norm(grouped, weight, eps).reshape(x.shape)


def attention(q, k, v, n_head, n_kv, head_dim, window, is_full):
    """Batched GQA attention over the whole prompt.

    Sliding layers attend positions max(0, t+1-window) .. t — the last
    min(t+1, window) positions; full layers attend 0 .. t. Scores carry
    head_dim ** -0.5 and softmax subtracts the row max, like the anchor.
    """
    seq = q.shape[0]
    group = n_head // n_kv
    scale = head_dim ** -0.5
    qh = q.view(seq, n_head, head_dim)                 # [T, H, D]
    kh = k.view(seq, n_kv, head_dim)                   # [T, KV, D]
    vh = v.view(seq, n_kv, head_dim)                   # [T, KV, D]

    kv_head_of = torch.arange(n_head, device=q.device) // group
    kh = kh.index_select(1, kv_head_of)                # [T, H, D]
    vh = vh.index_select(1, kv_head_of)                # [T, H, D]
    scores = torch.einsum("thd,shd->ths", qh, kh) * scale   # [T, H, S]

    idx = torch.arange(seq, device=q.device)
    # allowed[t, j] = j <= t; for sliding layers j >= t + 1 - window, so the
    # softmax row covers exactly the anchor's start = max(0, pos+1-window).
    allowed = idx.unsqueeze(0) <= idx.unsqueeze(1)
    if not is_full:
        allowed &= idx.unsqueeze(0) > idx.unsqueeze(1) - float(window)
    probs = torch.softmax(
        scores.masked_fill(~allowed.unsqueeze(1), float("-inf")), dim=-1)
    out = torch.einsum("ths,shd->thd", probs, vh)      # [T, H, D]
    return out.reshape(seq, n_head * head_dim)


def sigmoid_logit_add_routing(logits, bias, topk):
    """Select top-k on logits + bias, weight by the unbiased sigmoid."""
    f32 = logits.float()
    ids = torch.topk(f32 + bias, k=topk, dim=-1).indices      # [T, topk]
    weights = torch.sigmoid(f32.gather(1, ids))
    return weights, ids


def routed_swiglu(x, ids, r_gate, r_up, r_down, weights):
    """Per slot: slot weight * expert_plane(x), summed over the topk slots."""
    routed = torch.zeros_like(x)
    for slot in range(ids.shape[1]):
        expert_ids = ids[:, slot]                              # [T]
        planes_g = r_gate[expert_ids]                          # [T, inter, in]
        planes_u = r_up[expert_ids]                            # [T, inter, in]
        hidden = torch.nn.functional.silu(
            torch.einsum("td,tfd->tf", x, planes_g)) * \
            torch.einsum("td,tfd->tf", x, planes_u)            # [T, inter]
        contribution = torch.einsum("tf,tof->to", hidden,
                                    r_down[expert_ids])     # [T, out]
        routed = routed + contribution * weights[:, slot:slot + 1]
    return routed


def moe_block(w, x, topk):
    router = x @ w["gate_inp"].T                        # [T, n_expert] fp32
    weights, ids = sigmoid_logit_add_routing(router, w["bias"], topk)
    routed = routed_swiglu(x, ids, w["r_gate"], w["r_up"], w["r_down"], weights)
    shared = torch.nn.functional.silu(x @ w["s_gate"].T) * (x @ w["s_up"].T)
    shared = shared @ w["s_down"].T
    return shared + routed


def forward(ts: Tensors, tokens, device=None):
    meta = ts.meta
    n_embd = meta["kolibri1.embedding_length"]
    n_head = meta["kolibri1.attention.head_count"]
    n_kv = meta["kolibri1.attention.head_count_kv"]
    head_dim = meta["kolibri1.attention.key_length"]
    window = meta["kolibri1.attention.sliding_window"]
    eps = meta["kolibri1.attention.layer_norm_rms_epsilon"]
    rope_base = meta["kolibri1.rope.freq_base"]
    layer_types = meta["kolibri1.attention.layer_types"]
    topk = meta["kolibri1.expert_used_count"]

    device = torch.device(device or "cpu")
    seq = len(tokens)
    embed_rows = ts.get("token_embd.weight")             # [vocab, n_embd]
    x = torchify(embed_rows[tokens], device)             # [T, n_embd]
    positions = torch.arange(seq, device=device)
    residual = None

    for il, layer_type in enumerate(layer_types):
        w = Layer(ts, il, layer_type == 1).load(device)
        is_full = layer_type == 1

        if residual is None:
            residual = x
            x = rms_norm(x, w["attn_norm"], eps)
        else:
            x, residual = fused_rms_norm(x, residual, w["attn_norm"], eps)

        q = x @ w["q"].T
        k = x @ w["k"].T
        v = x @ w["v"].T
        q = per_head_rms_norm(q, w["q_norm"], n_head, head_dim, eps)
        k = per_head_rms_norm(k, w["k_norm"], n_kv, head_dim, eps)
        if not is_full:
            q = rotate_half_rope(q.view(seq, n_head, head_dim),
                                 positions, rope_base, head_dim).reshape(q.shape)
            k = rotate_half_rope(k.view(seq, n_kv, head_dim),
                                 positions, rope_base, head_dim).reshape(k.shape)
        attn = attention(q, k, v, n_head, n_kv, head_dim, window, is_full)
        attention_out = attn @ w["o"].T
        attention_out = rms_norm(attention_out, w["post_attn_norm"], eps)
        x, residual = fused_rms_norm(attention_out, residual, w["ffn_norm"], eps)
        moe_out = moe_block(w, x, topk)
        x = rms_norm(moe_out, w["post_ffn_norm"], eps)
        del w  # free this layer's big tensors before the next one materializes

    out_norm = torchify(ts.get("output_norm.weight"), device)
    logits = rms_norm(residual + x, out_norm, eps)       # final fused norm join
    values = logits @ torchify(ts.get("output.weight"), device).T
    return values[seq - 1]                               # next-token logits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", required=True)
    parser.add_argument("--tokens", required=True,
                        help="comma-separated prompt ids, or @FILE with one sequence per line")
    parser.add_argument("--out", help="write logits JSON here instead of stdout")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--device", default=None, help="cpu (default), cuda, or mps")
    parser.add_argument("--materialize", action="store_true",
                        help="keep every dequanted tensor resident (device permitting)")
    args = parser.parse_args()

    dtype = np.float64 if args.dtype == "float64" else np.float32
    meta, tensors, blob, data_start = load_gguf(args.gguf)
    ts = Tensors(meta, tensors, blob, data_start, dtype)
    ts.materialize = args.materialize
    if args.tokens.startswith("@"):
        with open(args.tokens[1:], "r", encoding="utf-8") as fp:
            sequences = [[int(t) for t in line.split(",") if t.strip()]
                         for line in fp if line.strip()]
    else:
        sequences = [[int(t) for t in args.tokens.split(",") if t.strip()]]

    results = []
    for tokens in sequences:
        if not tokens:
            raise ValueError("empty token sequence")
        values = forward(ts, tokens, args.device).detach().cpu().numpy()
        results.append({
            "source": "torch-reference",
            "prompt_tokens": tokens,
            "vocab": int(values.shape[0]),
            "argmax_token": int(np.argmax(values)),
            "argmax_logit": float(values.max()),
            "logits": [float(v) for v in values],
        })
        print(f"prompt_tokens={len(tokens)} argmax={results[-1]['argmax_token']} "
              f"logit={results[-1]['argmax_logit']:.6f}", file=sys.stderr)

    document = json.dumps(results[0] if len(results) == 1 else results, indent=2) + "\n"
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fp:
            fp.write(document)
    else:
        sys.stdout.write(document)


if __name__ == "__main__":
    main()
