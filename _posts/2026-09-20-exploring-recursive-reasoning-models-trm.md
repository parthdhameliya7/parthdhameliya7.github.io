---
layout: post
title: "Exploring Recursive Reasoning Models: Tiny Recursive Reasoning Model"
subtitle: "Reproducing Less is More — recursive reasoning with a 7M-parameter network."
date: 2026-09-20 19:30:00 -0400
tags: [trm, recursive-reasoning, reproduction, arc-agi, sudoku, maze]
reading_time: 10
math: true
---

This post is about exploring and reproducing **[Less is More: Recursive Reasoning with Tiny Networks](https://arxiv.org/pdf/2510.04871)**.

**Paper:** [arXiv:2510.04871](https://arxiv.org/pdf/2510.04871)

## Introduction

Tiny Recursion Model (TRM) is a recursive reasoning model that scores **45% on ARC-AGI-1** and **8% on ARC-AGI-2** with a tiny **7M-parameter** network. The idea that you must rely on massive foundation models — trained for millions of dollars by a big lab — to succeed on hard tasks is a trap. There is too much focus on exploiting LLMs and not enough on new lines of direction.

With recursive reasoning, it turns out that **less is more**: you do not always need to crank up model size for a model to reason and solve hard problems. A tiny model trained from scratch, recursing on itself and updating its answers over time, can achieve a lot without breaking the bank.

![Official ARC-AGI benchmark table from the paper]({{ '/assets/images/trm/fig-arc-agi-table.png' | relative_url }})

## Architecture

TRM recursively improves its predicted answer $Z_h$ with a tiny network. It starts from the embedded input question $x$, an initial embedded answer $Z_h$, and a latent $Z_l$. For up to $N_{\mathrm{supervision}} = 16$ improvement steps, it tries to improve $Z_h$ by:

1. Recursively updating $Z_l$ a total of $n_{\mathrm{recursions}}$ times given $x$, the current answer $Z_h$, and the current latent $Z_l$
2. Updating the answer $Z_h$ given the current $Z_h$ and current $Z_l$

This lets the model progressively correct earlier mistakes in a parameter-efficient way while limiting overfitting.

The paper uses different names for the same objects: $y$ for $Z_h$, $z$ for $Z_l$, $n$ for $n_{\mathrm{recursions}}$, $T$ for $T_{\mathrm{cycles}}$, and $N_{\mathrm{sup}}$ for $N_{\mathrm{supervision}}$. I keep the code / figure names below.

![TRM architecture]({{ '/assets/images/trm/fig-architecture.png' | relative_url }})

The network is tiny: RMSNorm, token mixing (self-attention or MLP-mixer), and a shared MLP — stacked as a 2-layer module $f$ reused at every recursion. Inclusion of $x$ tells the net which job it is doing:

$$
Z_l \leftarrow f(x + Z_h + Z_l), \qquad Z_h \leftarrow f(Z_h + Z_l).
$$

A **latent recursion** is $n_{\mathrm{recursions}}$ updates of $Z_l$, then one update of $Z_h$:

$$
\begin{aligned}
Z_l &\leftarrow f(x + Z_h + Z_l) && \text{for } i = 1,\ldots,n_{\mathrm{recursions}}, \\
Z_h &\leftarrow f(Z_h + Z_l).
\end{aligned}
$$

A **deep recursion** runs that process $T_{\mathrm{cycles}}$ times. The first $T_{\mathrm{cycles}}-1$ cycles are without gradients; the last cycle is the one that trains $f$. Then an output head and a halt head read from $Z_h$:

$$
\hat{y} = \arg\max\, f_O(Z_h), \qquad q_{\mathrm{halt}} = f_Q(Z_h[0]).
$$

The next supervision step reuses detached $(Z_h, Z_l)$, so memory does not grow with $N_{\mathrm{supervision}}$.

![TRM supervision loop]({{ '/assets/images/trm/fig-supervision.png' | relative_url }})

![Deep recursion / latent recursion]({{ '/assets/images/trm/fig-deep-recursion.png' | relative_url }})

Effective depth per supervision step is about $T_{\mathrm{cycles}}(n_{\mathrm{recursions}}+1)\,n_{\mathrm{layers}}$. On Sudoku the useful setting is $T_{\mathrm{cycles}} = 3$, $n_{\mathrm{recursions}} = 6$. Unlike HRM, TRM does **not** assume a fixed point or use the implicit-function / 1-step gradient trick — it backprops through one full recursion.

## Reproduced samples

I reproduced the TRM paper results. The paper leaves out many implementation details, so I had to dig through the author’s official code — which is fairly complex because it reuses legacy pieces from the preceding **Hierarchical Reasoning Model (HRM)**. I built a simpler version that runs with distributed training and even on Mac (with smaller batch sizes).

Below: given a Sudoku input, the model refines its output at every supervision step, with a halt signal answering “am I done?”

![Sudoku sample — refinement over supervision steps]({{ '/assets/images/trm/sudoku_trace_idx392623.gif' | relative_url }})

Settings for Sudoku: $T_{\mathrm{cycles}} = 3$, $n_{\mathrm{recursions}} = 6$, $N_{\mathrm{supervision}} = 16$.

The official study also covers **Maze-Hard 30×30**. I trained the same setup.

![Maze-Hard sample — path refinement over supervision steps]({{ '/assets/images/trm/maze_trace_idx426.gif' | relative_url }})

![Maze-Hard sample (second example)]({{ '/assets/images/trm/maze_trace_idx540.gif' | relative_url }})

## Training

### 1. Adaptive Computation Time (ACT)

TRM does not emit one answer in a single forward pass. Each puzzle can take up to $N_{\mathrm{supervision}} = 16$ supervision steps. Every step:

- Refine $(Z_h, Z_l)$ with a deep recursion
- Decode $Z_h$ into answer logits $\hat{y}$
- Decode a halt logit $q_{\mathrm{halt}}$ from the first position of $Z_h$ (“am I done?”)
- Either keep refining, or halt and load a new puzzle into that batch slot

The outer loop is **ACT**. The inner recursion on $(Z_h, Z_l)$ is the thinking. Halt is trained with binary cross-entropy against whether the current prediction is exactly correct:

$$
\mathcal{L} = \mathcal{L}_{\mathrm{CE}}(\hat{y}, y_{\mathrm{true}}) + \tfrac{1}{2}\,\mathrm{BCE}\!\left(\sigma(q_{\mathrm{halt}}),\; \mathbf{1}[\hat{y} = y_{\mathrm{true}}]\right).
$$

Unlike HRM, TRM drops the Q-learning “continue” head, so there is no extra forward pass. **During training**, a sample halts when

$$
\text{step} \ge N_{\mathrm{supervision}} \quad\text{or}\quad q_{\mathrm{halt}} > 0,
$$

subject to a small exploration rule (below). At test the official code always runs all $N_{\mathrm{supervision}} = 16$ steps so every sample in the batch has the same depth; $q_{\mathrm{halt}}$ is still computed (ARC uses $\sigma(q_{\mathrm{halt}})$ as a vote weight) but it does **not** stop a sample early.

The trainer carries batch state: current inputs, labels, step count, halted flags, and $(Z_h, Z_l)$. Example: at supervision step 7, 4 out of 32 samples halt. Those 4 slots are replaced with new samples and their latents reset — they skip the remaining supervision steps.

There is also a **halt exploration probability** $p_{\mathrm{explore}} = 0.1$. With that probability, a sample draws a random floor $k$ uniformly from $\{2,\ldots,N_{\mathrm{supervision}}\}$ and is not allowed to halt until $\mathrm{step} \ge k$. The other $90\%$ can halt as soon as $q_{\mathrm{halt}} > 0$. So if halt would fire at step $3$ and the floor is $12$, the sample keeps refining; if halt fires at step $10$ and the floor is $3$, it still halts. This is not “$10\%$ of already-halted samples get extra steps” — the floor is applied to a random $10\%$ of the batch so the model can still learn when halt confidence was premature.

![ACT outer loop]({{ '/assets/images/trm/fig-act-outer-loop.png' | relative_url }})

![ACT inner loop — deep recursion]({{ '/assets/images/trm/fig-act-detail.png' | relative_url }})

### 2. Setup and hyperparameters (Sudoku-Extreme & Maze-Hard 30×30)

Training ran on **3× NVIDIA RTX 6000 Ada**.

| Setting | Value |
| --- | --- |
| Optimizer | AdamW (author code uses AdamATan2; paper says AdamW) |
| Beta1 / Beta2 | 0.9 / 0.95 |
| Weight decay | 1.0 |
| Global batch | 768 (256 local × 3 GPUs) |
| Learning rate | 1e-4, cosine with warmup restart, warmup 2000 steps |
| Epochs / steps | ~60–65k |
| EMA | rate 0.999 |

- **Sudoku-Extreme:** $N_{\mathrm{supervision}} = 16$, $T_{\mathrm{cycles}} = 2$, $n_{\mathrm{recursions}} = 6$
- **Maze-Hard 30×30:** $N_{\mathrm{supervision}} = 16$, $T_{\mathrm{cycles}} = 3$, $n_{\mathrm{recursions}} = 4$

Architecture highlights: 2 layers, embed size 512, 8 attention heads, RoPE, SwiGLU MLP, puzzle embedding length 16. The paper uses StableMax cross-entropy; in my reproduction vanilla cross-entropy made no meaningful difference. Puzzle embeddings are optimized with SignSGD.

### 3. ARC-AGI and sparse embeddings

ARC-AGI-1 / ARC-AGI-2 are geometric puzzles that are easy for humans and hard for current AI. Each task has 2–3 input–output demos and 1–2 test inputs. Score is accuracy over test inputs with two attempts. Max grid is 30×30. TRM numbers are on the **public evaluation** sets. Datasets are small, so heavy augmentation is used.

ARC-AGI-1 expands to on the order of **~876k** augmented puzzles / identifiers. A dense embedding table at that scale blows up GPU memory — hence **sparse embeddings**: only the batch’s rows are checked out into a small local buffer that carries gradients, then written back with SignSGD (duplicate IDs in a batch sum their gradients).

![CastedSparseEmbedding]({{ '/assets/images/trm/fig-sparse-embedding.png' | relative_url }})

## Results (paper vs reproduction)

![Official puzzle benchmark table]({{ '/assets/images/trm/fig-paper-benchmark.png' | relative_url }})

Reproduction numbers land close to the paper. I have not chased further accuracy tricks yet — next I care more about applying the idea to practical problems (for example time-series output patch refinement, or complex optimization) than squeezing another point on the leaderboard.

| Models | Sudoku-Extreme | Maze-Hard 30×30 | ARC-AGI-1 |
| --- | --- | --- | --- |
| TRM-MLP-5M (reprod) | 86.90% | — | — |
| TRM-ATTN-7M (reprod) | 75.63% | 82.67% | 36.3% |

ARC-AGI-1 here is under-trained vs the paper: the author used about **500k** optimizer steps; I ran **100k**. That run took about **30 hours** on 3× RTX 6000 Ada at near-max memory and peaked around **36.6%** — not a fair comparison to the full official config.

### Logs: Sudoku-Extreme

<div class="figure-row">
  <img src="{{ '/assets/images/trm/fig-sudoku-loss.png' | relative_url }}" alt="Sudoku training loss">
  <img src="{{ '/assets/images/trm/fig-sudoku-acc.png' | relative_url }}" alt="Sudoku halted exact accuracy">
</div>

### Logs: Maze-Hard 30×30

<div class="figure-row">
  <img src="{{ '/assets/images/trm/fig-maze-loss.png' | relative_url }}" alt="Maze training loss">
  <img src="{{ '/assets/images/trm/fig-maze-acc.png' | relative_url }}" alt="Maze halted exact accuracy">
</div>

### Logs: ARC-AGI-1

<div class="figure-row">
  <img src="{{ '/assets/images/trm/fig-arc-loss.png' | relative_url }}" alt="ARC-AGI-1 training loss">
  <img src="{{ '/assets/images/trm/fig-arc-acc.png' | relative_url }}" alt="ARC-AGI-1 halted exact accuracy">
</div>

## Conclusion

TRM is a simple recursive reasoning approach: one tiny network recurses on a latent reasoning state and progressively improves its answer. Unlike HRM, it needs no fixed-point theorem, no heavy biological story, and no hierarchy. Fewer layers, one shared network, simpler halting (no extra forward pass) — and better generalization.

Today TRM (and HRM) are **supervised** rather than generative: one input maps to one deterministic answer. Many problems have multiple valid answers, so extending TRM toward generative settings is an interesting next step.

## What next?

After TRM, the successor line is the **Generative Recursive Reasoning Model (GRAM)** — turning recursive latent reasoning into probabilistic multi-trajectory computation instead of a single refinement path. I have not explored it fully yet.

[GRAM](https://ahn-ml.github.io/gram-website/)
