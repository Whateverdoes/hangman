# Meltwater Hangman: Neural Sequence Modeling

A PyTorch character-level Transformer that plays Hangman with a six-wrong-guess limit. The project covers reachable-state generation, model training, batched gameplay, statistical evaluation, and Kaggle submission packaging.

The inference path uses only the trained network: no dictionary, offline oracle, n-gram table, or external service. An earlier leave-one-out oracle remains available for training experiments, but the current recipe uses per-position supervision with the policy and presence losses disabled.

## Model and pipeline

- **Character Transformer:** six encoder layers, 512-dimensional embeddings, eight attention heads, and a 2,048-dimensional feed-forward layer; approximately 19.01M parameters.
- **Game-state conditioning:** the revealed board, a multi-hot vector of guessed letters, and the wrong-guess count.
- **Reachable training states:** guessing a letter reveals every occurrence. State generation samples letter classes, avoids terminal states, and deduplicates the newly generated states for each word.
- **Per-position learning:** targets come from the hidden characters in the training word. At inference, positional probabilities are aggregated to rank unguessed letters.
- **Batched inference:** games of equal word length advance in lockstep, avoiding padding and processing many active games in one model call.
- **Evaluation:** win rate, Wilson 95% confidence intervals, total wrong guesses, and performance by word length.

## Recorded results and evidence

The saved Kaggle training notebook contains a 40,000-step run with the following results. A compact, data-free copy of the relevant output is included in [reports/kaggle_run.md](reports/kaggle_run.md).

| Measurement | Recorded value |
|---|---:|
| Parameters | 19.01M |
| Newly generated training states | 5,507,500 in 93 seconds |
| Total training rows, including carried labels | 6,945,093 |
| Validation split | 5,000 words, split seed 7 |
| Validation sample seed 1, n=2,000 | 63.60% [95% CI 61.47-65.68%] |
| Validation sample seed 2, n=2,000 | 63.05% [95% CI 60.91-65.14%] |
| Validation sample seed 3, n=2,000 | 63.15% [95% CI 61.01-65.24%] |
| Mean across the three samples | 63.27% |

These are recorded notebook results, not a fresh retraining or a private leaderboard score. The samples are drawn independently from the same 5,000-word validation pool and can overlap; they are not 6,000 unique evaluation words or three independent training runs.

Earlier experiments used inconsistent split seeds, contaminating some validation measurements. Their headline scores and numerical ablation gains are not used here. Larger public-test scores appear in local project notes, but their final-run artifacts are not included in this publication, so they are not presented as verified results.

The current canonical split seed is **7**; model initialization and batch sampling use training seed **1**. The state builder checks that its training vocabulary excludes the canonical validation words. Training and evaluation of externally supplied label files/checkpoints still require checking their split provenance.

## Setup

Use Python 3.13 and the pinned dependencies. The local environment used for the invariant checks has PyTorch 2.13.0 and NumPy 2.5.2.

```bash
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python tests/test_pipeline.py
```

Organizer-provided `train.txt` and `test.txt` are not distributed. To run the competition pipeline, obtain them through the organizer and place them at `data/train.txt` and `data/test.txt`. Checkpoints, training rows, experiment logs, and submissions are generated locally and excluded from Git.

## Smoke check

The smoke path validates the pipeline with a small model; it does not reproduce the recorded score. Use separate output paths to preserve existing checkpoints.

```bash
.venv/bin/python scripts/build_labels.py --fast --seed 7 \
  --out experiments/labels_smoke.tsv
.venv/bin/python scripts/train.py --fast --skip-golden \
  --labels experiments/labels_smoke.tsv --out model_smoke.pt \
  --seed 1 --w-policy 0 --w-presence 0 --w-perpos 1
.venv/bin/python scripts/evaluate.py --model model_smoke.pt \
  --val-size 5000 --seed 7 --n 600 --seeds 1
```

## Recorded full training recipe

This recipe follows the saved 40,000-step notebook run. Label generation supplies rollout states; additional teacher-free states expand coverage. Carried rows and generated rows may overlap, so total row count is not a guarantee of global uniqueness.

```bash
.venv/bin/python scripts/build_labels.py --full --seed 7 --val-size 5000 \
  --out experiments/labels.tsv
.venv/bin/python scripts/build_states.py --seed 7 --val-size 5000 \
  --per-word 25 --include experiments/labels.tsv \
  --out experiments/states.tsv
.venv/bin/python scripts/train.py --labels experiments/states.tsv \
  --out model_wide_states.pt --dim 512 --layers 6 --heads 8 --ff 2048 \
  --steps 40000 --warmup 1000 --batch-size 256 --seed 1 \
  --w-policy 0 --w-presence 0 --w-perpos 1
.venv/bin/python scripts/evaluate.py --model model_wide_states.pt \
  --val-size 5000 --seed 7 --n 2000 --seeds 1 2 3
.venv/bin/python scripts/make_submission.py --model model_wide_states.pt \
  --test data/test.txt --out submission.csv
```

The eight-probe golden-word check runs after full training and raises if the model fails its learning sanity check. It is not a substitute for held-out evaluation. Use identical validation sizes, split seeds, and sample seeds when comparing configurations, and do not select configurations using public-test performance.

## Kaggle workflow

```bash
bash scripts/package_for_kaggle.sh
```

Upload `dist/hangman-src.zip` as a private Kaggle source dataset and attach it with the competition inputs. Import [notebooks/kaggle_train_submit.ipynb](notebooks/kaggle_train_submit.ipynb), run with `FAST = True` first, then set it to `False` for the full recipe. The notebook calls the same CLI scripts used locally. CPU, CUDA, and Apple MPS training paths are supported; training time depends on the recipe and hardware.

The training-only notebook is retained as an experiment driver. Published measurements are limited to the run evidence linked above.

## Repository layout

```text
src/hangman/
  dataset.py        Vocabulary normalization, deterministic splits, leakage guard
  states.py         Reachable Hangman states and rollout generation
  encoding.py       Shared training/inference tensor encoding
  model.py          Conditioned character Transformer and LSTM baseline
  train.py          Training losses, sampling, and learning sanity check
  distill.py        Optional offline teacher-label generation
  teacher/          Offline vocabulary oracle; excluded from inference imports
  simulate.py       Per-position scoring and batched lockstep gameplay
  policy.py         Single-state policy using shared per-position scoring
  evaluate.py       Wilson intervals, length bands, and result logging
  submit.py         Submission writing and schema validation
scripts/            CLI stages and Kaggle packaging
notebooks/          Kaggle training and submission drivers
reports/            Data-free excerpts of recorded run evidence
tests/             Invariant checks
```

The single-state policy and batched simulation both default to per-position aggregation. Pass the complete prior-guess set to `NeuralPolicy.guess`; it derives the wrong-guess count from letters absent from the revealed board and stops on solved boards or six wrong guesses.

## Correctness checks

The GitHub Actions workflow runs the invariant suite and verifies that the Kaggle source package excludes data and generated artifacts on pushes and pull requests.

The 20 tests cover deterministic disjoint splits, leakage rejection, reachable states, leave-one-out teacher behavior, guess uniqueness, the six-life limit, per-position aggregation, submission validation, reference/batched policy agreement, and direct teacher-import exclusion in inference modules. They check stable behavior rather than a target benchmark score.

## Next improvements

- Store and validate split provenance with label files and checkpoints.
- Repeat comparative experiments on verified held-out splits.
- Isolate state-count and training-length effects with otherwise identical runs.
- Compare sampling with replacement against shuffled epochs.

## License

The source code is available under the [MIT license](LICENSE). Organizer-provided data and generated models are excluded.
