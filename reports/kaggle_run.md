# Recorded Kaggle training run

This excerpt was extracted from the locally saved executed notebook
`hanngman2.ipynb`, dated September 2, 2026. It contains aggregate metrics only;
organizer word lists, label rows, checkpoints, and submissions are excluded.
It has not been rerun as part of preparing the public repository.

Configuration: character Transformer, dim 512, six layers, eight heads,
feed-forward dim 2048, dropout 0.1; 40,000 steps, 1,000 warmup steps,
batch size 256, training seed 1. Loss weights: per-position 1, policy 0,
presence 0. Data split: 5,000 held-out words, split seed 7. Additional states:
25 per training word, with existing rollout labels carried through.
Evaluation: per-position aggregation, six lives, n=2,000 per sample,
sample seeds 1, 2, 3. Samples can overlap within the same validation pool.

```text
[split-check] OK — 0 of 5000 canonical held-out words appear in the state vocabulary
vocabulary 225300 | train 220300 | val 5000 | up to 25 states/word
wrote 6945093 rows -> states.tsv in 93s
  5507500 generated (25.00 per word) + 1437593 carried
params: 19.01M
golden-word check passed: mean rank 1.8 (random ~13)
=== model_wide-512-states.pt seed=1 (n=2000, 4.4s) ===
  win rate     63.60%  [95% CI 61.47% - 65.68%]
  total wrong  7683  (mean 3.841/word)  <- tie-break metric
=== model_wide-512-states.pt seed=2 (n=2000, 3.9s) ===
  win rate     63.05%  [95% CI 60.91% - 65.14%]
  total wrong  7761  (mean 3.881/word)  <- tie-break metric
=== model_wide-512-states.pt seed=3 (n=2000, 4.0s) ===
  win rate     63.15%  [95% CI 61.01% - 65.24%]
  total wrong  7728  (mean 3.864/word)  <- tie-break metric
across 3 seeds: mean 63.27% | spread 0.55% | ['63.60%', '63.05%', '63.15%']
```
