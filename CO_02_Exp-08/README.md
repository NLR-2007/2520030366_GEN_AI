# CO_02_Exp-08 — Scaled Dot-Product and Multi-Head Attention from Scratch

Module II, second experiment. Exp-07 read the distribution a transformer
produces at its *output*. This one builds the mechanism that produced it:
**attention**, written from first principles in PyTorch and then proved correct
against `torch`'s own implementations.

Handout, skilling session 8 (CO2):

> Implement scaled dot-product and multi-head attention from scratch in PyTorch.

## Aim

Attention is usually met as a formula on a slide. Here it is built line by line,
and every claim about it is tested rather than asserted:

- that the output matches `F.scaled_dot_product_attention` to 1e-7,
- that the multi-head block matches `nn.MultiheadAttention` to 1e-7,
- that a causal mask leaks **exactly zero** weight into the future,
- that gradients reach every parameter and the `-inf` mask produces no NaNs.

## The formula

```
Attention(Q, K, V) = softmax( Q Kᵀ / √d_k ) V
```

Read as a lookup table that returns a blend instead of one row:

| Term | Shape | Meaning |
|---|---|---|
| **Q** query | `(L_q, d_k)` | what each position is looking for |
| **K** key | `(L_k, d_k)` | what each position offers as a label |
| **V** value | `(L_k, d_v)` | what each position actually contributes |
| `Q Kᵀ` | `(L_q, L_k)` | similarity of every query to every key |
| `/ √d_k` | — | keeps score variance near 1 so softmax does not saturate |
| `softmax` | `(L_q, L_k)` | each row becomes a distribution over positions |
| `… V` | `(L_q, d_v)` | weighted average of the values |

Five lines of code in [attention_from_scratch.py](attention_from_scratch.py):

```python
scores  = query @ key.transpose(-2, -1)      # 1. similarity
scores  = scores / math.sqrt(d_k)            # 2. scale
scores  = scores.masked_fill(~mask, -inf)    # 3. mask BEFORE softmax
weights = torch.softmax(scores, dim=-1)      # 4. normalise
output  = weights @ value                    # 5. blend
```

**Order matters.** The mask is applied to the scores, not to the weights. `-inf`
exponentiates to exactly 0, so forbidden positions get no probability mass and
the remaining weights still sum to 1. Masking *after* softmax would zero some
entries and leave the rows no longer normalised.

## Files

| File | Purpose |
|---|---|
| `attention_from_scratch.py` | The implementation and all 7 demonstrations |
| `requirements.txt` | `torch` |
| `output.txt` | Full console transcript of the recorded run |
| `attention_results.csv` | Every check and its numeric result |

No `transformers` dependency — this is pure PyTorch.

## Run

```powershell
pip install torch
python attention_from_scratch.py
```

| Flag | Meaning |
|---|---|
| `--seed N` | Global torch seed (default 42) |
| `--no-csv` | Skip writing `attention_results.csv` |

Runs on CPU in a couple of seconds. To capture a transcript, set
`$env:PYTHONIOENCODING = "utf-8"` first.

## What the code contains

| Component | Lines | Notes |
|---|---|---|
| `scaled_dot_product_attention` | ~30 | Works on any leading batch dims — `(L, d)`, `(B, L, d)` or `(B, h, L, d)` |
| `MultiHeadAttention` | ~60 | `nn.Module` with the four projections, head split/merge, dropout |
| `causal_mask(L)` | 3 | Lower-triangular keep-mask |
| `padding_mask(lengths, L)` | 4 | Per-sequence keep-mask over key positions |

Masks use the **keep convention** — `True` means "may attend". `torch`'s
`attn_mask` uses the opposite, which is why the verification passes `~mask`.
Mixing the two up is the single most common attention bug, and it fails
silently: the model trains, it just attends to the wrong half of the sentence.

## Results

`torch 2.12.1+cpu`, seed 42, CPU. **10/10 checks passed.**

### Correctness against PyTorch

| Check | max\|diff\| |
|---|---|
| single-head output vs `F.scaled_dot_product_attention` | 1.192e-07 |
| causal output vs `F.sdpa(is_causal=True)` | 1.192e-07 |
| multi-head output vs `nn.MultiheadAttention` | 8.941e-08 |
| attention weights (averaged over heads) | 2.980e-08 |
| causal multi-head vs `nn.MultiheadAttention` | 1.192e-07 |
| future leakage (causal mask) | **0.000e+00** |
| padding leakage | **0.000e+00** |

Every difference is at fp32 rounding error (~1e-7), which is the strongest
statement available: the implementations are numerically identical, differing
only in floating-point summation order.

The multi-head check copies our four weight matrices into `nn.MultiheadAttention`
first. `torch` packs Q, K and V into one `(3·d_model, d_model)` matrix, so the
copy is `torch.cat([w_q, w_k, w_v], dim=0)` — matching that layout is itself a
useful thing to have had to work out.

The two leakage figures are **exactly zero**, not merely small. That is the
`-inf`-before-softmax design: `exp(-inf) = 0` with no residue.

### Why √d_k is not optional

Q and K are unit-variance, so each dot product sums `d_k` independent products
and its variance grows linearly with `d_k`:

| d_k | std(raw scores) | std(scaled) | max softmax weight, unscaled | scaled |
|---|---|---|---|---|
| 4 | 2.054 | 1.027 | 0.9711 | 0.5854 |
| 16 | 3.963 | 0.991 | 0.9708 | 0.2991 |
| 64 | 7.906 | 0.988 | **1.0000** | 0.2072 |
| 256 | 16.416 | 1.026 | **1.0000** | 0.2887 |
| 1024 | 32.003 | 1.000 | **1.0000** | 0.3595 |

`std(raw)` tracks √d_k almost exactly — 2.05 ≈ √4, 7.91 ≈ √64, 32.00 = √1024.
`std(scaled)` stays pinned near 1.0 at every size. That is the whole job of the
√d_k: it makes the score distribution independent of head dimension.

From d_k = 64 upward, the unscaled maximum weight is **1.0000** — softmax has
collapsed to a hard argmax, its gradient is zero there, and the layer cannot
learn. Scaled, it stays in the 0.2–0.6 range.

Read the last two columns with care: they are the maximum over a 64 × 64 matrix,
so they are extreme values rather than typical ones, and the scaled column is
not monotone in d_k. The reliable signal is the **std** columns — raw variance
growing as √d_k is the actual mechanism, and saturation is its consequence.

### The causal mask is the whole BERT/GPT difference

On `["The", "cat", "sat", "on", "it"]`, unmasked attention lets `"The"` attend
to `"it"`. With `causal_mask(5)`:

```
              The    cat    sat     on     it
The         1.000  0.000  0.000  0.000  0.000
cat         0.165  0.835  0.000  0.000  0.000
sat         0.222  0.411  0.367  0.000  0.000
on          0.111  0.057  0.625  0.207  0.000
it          0.242  0.091  0.246  0.259  0.162
```

Strictly upper-triangular zeros, row sums still exactly 1, and token 0 gets
weight **1.0** on itself because it has nothing else to look at. One
`torch.tril` is the entire architectural difference between an encoder that
reads both directions and a decoder that can be trained to predict the next
token.

### Padding masks

Batching sequences of lengths `[5, 3]` to a common length 5 — without a mask,
the short sequence averages in two meaningless padding vectors, and **the same
sentence gets a different encoding depending on what else is in the batch**.
With the mask, weight on padded keys is 0.000e+00.

### Multi-head

`d_model = 64`, `h = 8`, `d_head = 8`, **16,640 parameters** = 4 projections ×
(64×64 + 64).

**Head count does not change the parameter count.** It only changes how the
same `d_model` dimensions are partitioned. 8 heads of 8 dims cost exactly what
1 head of 64 dims costs — the gain is free in parameters, paid for in nothing
but reshape operations.

Row 0 of the attention matrix, per head (untrained, random init):

| head | k0 | k1 | k2 | k3 | k4 | k5 |
|---|---|---|---|---|---|---|
| 0 | 0.041 | 0.183 | 0.383 | 0.030 | 0.131 | 0.232 |
| 3 | 0.120 | 0.085 | 0.170 | 0.259 | 0.218 | 0.147 |
| 7 | 0.137 | 0.145 | 0.148 | 0.159 | 0.212 | 0.199 |

Mean std across heads: **0.0635**. The heads already disagree about where to
look, and that is the entire point — one head can only average one way,
whereas h heads can track h different relationships (syntax, coreference,
position) and `w_o` recombines them.

### Gradients

Backward pass through a causally-masked block, `loss = out²·mean()`:

| parameter | shape | grad norm |
|---|---|---|
| `w_q.weight` | (32, 32) | 0.023491 |
| `w_k.weight` | (32, 32) | 0.024069 |
| `w_v.weight` | (32, 32) | 0.230499 |
| `w_o.weight` | (32, 32) | 0.194624 |
| input `x` | (1, 4, 32) | 0.029101 |

Every parameter receives a non-zero gradient and **no NaNs** appear despite the
`-inf` entries — `softmax` handles an all-finite row with `-inf` neighbours
cleanly. (A fully `-inf` row *would* produce NaN; that is why a padding mask
must never mask out an entire sequence.)

Note `w_v` and `w_o` carry roughly **10×** the gradient of `w_q`/`w_k`. Query and key
only influence the loss through softmax, which flattens gradients; value and
output sit on the direct path.

### Why long context is expensive

The score matrix is L × L, so memory and compute are **O(L²)**:

| seq len | score matrix | fp32 memory | vs L=512 |
|---|---|---|---|
| 128 | 128 × 128 | 0.1 MB | — |
| 512 | 512 × 512 | 1.0 MB | 1× |
| 2,048 | 2,048 × 2,048 | 16.0 MB | 16× |
| 8,192 | 8,192 × 8,192 | 256.0 MB | 256× |
| 32,768 | 32,768 × 32,768 | **4,096.0 MB** | **4,096×** |

Doubling the context quadruples the matrix. At 32k tokens one head's scores
need 4 GB in fp32 — *per layer, per batch item*. This single table explains
Flash Attention, sliding-window attention, and why context length is priced
the way it is.

## Known limitations

| Limitation | Why |
|---|---|
| No positional encoding | Attention is permutation-invariant; position is added outside this block |
| Materialises the full L×L matrix | Flash Attention tiles it to avoid exactly this — shown in Part 7 |
| No cross-attention demo | The code supports it (`query` from one source, `key`/`value` from another); only self-attention is exercised |
| No KV cache | Inference-time optimisation, not part of the definition |
| Heads are untrained | Head specialisation is shown as disagreement at init, not as learned linguistic roles |
| CPU only | Mechanism is identical on GPU; this runs in seconds either way |

## Viva questions

1. Write the attention formula. What shape is each of Q, K and V?
2. What is the difference between a key and a value? Why are they separate?
3. Why divide by √d_k and not by d_k, or by nothing?
4. What happens to the gradient if you remove the scaling at d_k = 1024?
5. Why must the mask be applied before softmax rather than after?
6. Why `-inf` and not a large negative number like −1e9?
7. What does `torch.tril` have to do with the difference between BERT and GPT?
8. In a causal mask, why does token 0 always have attention weight exactly 1.0?
9. What goes wrong if you batch variable-length sequences without a padding mask?
10. When would a padding mask produce NaN, and how do you avoid it?
11. Does increasing the number of heads increase the parameter count? Justify.
12. Why use h heads of d/h dims instead of one head of d dims?
13. What is `w_o` for? What breaks if you drop it?
14. Why does `torch` pack Q, K, V into a single `in_proj_weight`?
15. Why is attention O(L²)? What is the memory for L = 32,768 in fp32?
16. Attention is permutation-invariant. Why, and what fixes it?
17. How would you turn this self-attention into cross-attention?
18. Why do `w_v` and `w_o` receive much larger gradients than `w_q` and `w_k`?
