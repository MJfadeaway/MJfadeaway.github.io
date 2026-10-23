# From Data to Parameters: an Optimizer-as-Compressor Framework

## 1. Question
Given data {(x_i,y_i)}, an architecture f(x;θ) and an initialization θ0, how does the optimizer turn data into θ*? Following Yi Ma's view (architecture = unrolled optimization of a compression objective), we treat the *optimizer trajectory itself* as an information-carrying object: the data is "written" into the parameters as a displacement Δ_t = θ_t − θ0, and we ask how large/structured that written code is.

## 2. Framework (four components)
Notation: J_t ∈ R^{n×P} per-sample gradient matrix, r_t = y − f(X;θ_t) + f(X;θ0) (centered residual), K_t = J_tJ_tᵀ/n, gradient flow/descent θ_{t+1}=θ_t+η J_tᵀ r_t/n.

**C1 – Data-span principle.** Δ_t = (η/n) Σ_s J_sᵀ r_s ∈ span of per-sample gradients along the path. Parameters never move in directions the data does not "see"; the change is rank ≤ n·(path) in P dims. Measured by the out-of-span fraction ‖(I−Π)Δ‖/‖Δ‖ (Π = projector on rows of J_0 and J_t).

**C2 – Spectral mode dynamics.** In the linearized regime (K_t≈K_0=UΛUᵀ), r_t = U(1−ηΛ)^t Uᵀ r_0. Each kernel eigenmode is fitted at rate ηλ_k: top modes (smooth/signal) first, bottom modes (noise) last. This gives a computable, per-mode account of "when" each part of the data is absorbed. Deviation from the prediction measures feature learning (K_t drift); it scales as 1/α with the output scale α (α large = lazy, small = rich).

**C3 – Compression measure of the written code.** Rate R_ε(ΔW)=½ logdet(I+ (m/(ε²n))ΔWΔWᵀ) (MCR² coding rate, ε=0.1) and effective rank exp(entropy of normalized singular values²). They quantify how many directions/bits the optimizer spends; Δ should be low-effective-rank relative to P and growing sub-linearly in time.

**C4 – Generalization split.** Fraction of function change Δf on the training set lying in the top-k kernel modes (k=5, teacher has 3 hidden units) – a "signal share" of the written code; test MSE is tracked jointly. Early stopping = stopping after signal modes and before noise modes are absorbed.

Pipeline: compute K_0 spectrum → predict mode schedule (C2) → run optimizer → measure span (C1), rate/eff. rank (C3), signal share/test error (C4).

## 3. Experiments (`framework.py`, numpy only; `python framework.py`)
Setup: teacher y = vᵀtanh(W*ᵀx)+noise(0.1), d=20, n=40, 2000 test points; student two-layer tanh net, m=200, output scale α∈{0.3,1,3}, step η=0.5/α², 3000 GD steps, f centered at init. Outputs: `results.json`, `fig_main.png`.

| α | out-of-span (C1, final) | C2 err ‖r̂−r‖/‖y‖ at t=10 | eff.rank ΔW | coding rate | ‖Δ‖ | test MSE |
|---|---|---|---|---|---|---|
| 0.3 | 0.161 | 0.049 | 3.72 | 59.6 | 6.41 | 0.127 |
| 1.0 | 0.052 | 0.014 | 4.54 | 37.6 | 1.95 | 0.142 |
| 3.0 | 0.018 | 0.0053 | 4.74 | 19.1 | 0.65 | 0.146 |

(train MSE → ~0 in all cases.)

## 4. Findings
1. **C1 verified**: Δ lies almost entirely in the gradient span (≤16% outside even for the richest case); the out-of-span part shrinks ≈ 1/α, i.e. exactly vanishes in the lazy limit.
2. **C2 verified**: the linear-mode predictor tracks the actual residual to <5% of ‖y‖ at all times, error decreasing ≈1/α. At early times the error is largest for small α (kernel drift = feature learning).
3. **C3**: Δ has effective rank only ~4–5 out of min(m,d)=20 dims – the optimizer writes a very low-dimensional code, consistent with the 3-unit teacher (rank ≈ 3 + noise directions). Smaller init scale (rich regime) yields lower effective rank (3.7 vs 4.7: more compact code) but much larger coding rate/distance, and the best test error (0.127 vs 0.146).
4. **C4**: signal share in top-5 modes goes 0.70 → 0.59 as training proceeds (noise modes being absorbed); test MSE is minimal around t≈100–300 and flat afterward in this small-noise setting. Rich-regime training generalizes better.

## 5. Debugging log
- Mode predictor initially used rate lr·λ; the correct rate is (η=lr/α²)·λ since K_0 already contains α². Fixed.
- Initial residual must be r_0=y (centered predictor), not y−f_0; otherwise a t=0 error appeared. Fixed.
- Prediction error was normalized by ‖r_t‖ (blows up as r→0); now normalized by ‖y‖.
- eff-rank divide-by-zero at Δ=0 guarded.

## 6. Limitations / next steps
Single seed and one architecture; the rate-reduction link is empirical, not a proven bound. Next: multiple seeds, Adam/SGD (preconditioned span), deeper nets/transformers, a bound of test error by ε-coding-rate of Δ, and using R(Δ) as a regularizer/diagnostic.
