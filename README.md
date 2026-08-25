# Beyond Binary Churn: Early Prediction of Subscription Renewal Trajectories

Reproducibility repository for the MSc dissertation *"Beyond Binary Churn: Early
Prediction of Subscription Renewal Trajectories"*.

Instead of the usual binary *churn / no-churn* label, this project reframes
subscription outcomes as a **multi-class renewal trajectory** — whether a user
churns or, if they renew, whether their engagement **contracts**, stays **stable**,
or **expands** — and predicts that trajectory *before* the renewal decision using
only information available at a configurable number of days ahead of expiry.

The repository is organised so that an examiner can either **fully reproduce** the
pipeline from the raw Kaggle data, or **quickly verify** the reported statistics
from the committed run-level results without any expensive computation.

**Final dissertation:** [Dissertation_JesperPhillips_Final.pdf](Dissertation_JesperPhillips_Final.pdf)

---

## Repository layout

```
Beyond-Binary-Churn/
├── README.md
├── LICENSE                         # CC-BY-4.0 (code, notebooks, docs)
├── requirements.txt                # pinned Python dependencies
├── .gitignore
├── prepare_raw_data.py             # Kaggle CSV -> Parquet (pure format conversion)
├── truecut_precompute.py           # rebuilds features at the true decision date E-c
├── clf_utils.py                    # shared paths, splits, metrics, sweep cache
│
├── ClassEDA.ipynb                  # exploratory data analysis
├── ClassEngineering.ipynb          # label construction + feature engineering
├── Classify_01_DirectMulticlass.ipynb   # direct multi-class models (XGBoost / LightGBM)
├── Classify_02_TwoStage.ipynb           # two-stage (renew? -> movement) classifier
├── Classify_02b_TwoStageRegression.ipynb# two-stage regression variant
├── Classify_03_DeepLearning.ipynb       # sequence deep-learning models
├── Classify_04_StatisticalAnalysis.ipynb# paired significance tests on the headline results
│
├── KKBoxData/
│   ├── feature_manifest.json       # feature contract (read by clf_utils at import)
│   ├── clf_*.csv, clf_*.json       # committed run-level results & configs
│   ├── sweeps/                     # per-run hyper-parameter tuning (best params + trials)
│   └── RawData/                    # (empty) place Kaggle CSVs here — see its README
│
├── outputs/statistics/             # paired-test outputs (csv + LaTeX)
└── FinalPlots/                     # figures reported in the dissertation
```

Large regenerated artefacts — the raw data, engineered panels (`*.parquet`),
sequence caches, and trained models (`*.pt`, `*.joblib`, `*.npy`) — are **not**
committed (see `.gitignore`). They are all reproduced by the steps below.

---

## Environment

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

All jobs were run and validated in a Python 3.13.8 environment; compatibility with other Python versions is not guaranteed. The pinned versions define the environment for reproducing the repository. A CPU-only install reproduces everything; a CUDA/MPS build of PyTorch only accelerates `Classify_03`.

---

## Option A — Quick verification (no heavy compute)

The committed run-level result tables are enough to re-derive the headline
statistics. After installing the requirements:

1. Open **`Classify_04_StatisticalAnalysis.ipynb`**.
2. Run all cells.

It loads the per-run macro-F1 tables that are already committed under
`KKBoxData/` (`clf_direct_holdout_runs.csv`, `clf_twostage_runs.csv`,
`clf_02b_runs.csv`, `clf_dl_holdout_runs.csv`) and recomputes the pre-specified
paired model comparisons, writing them to `outputs/statistics/`. The committed
copies of those outputs let you diff your results against the reported ones.

The figures reported in the dissertation are already present in `FinalPlots/`.

---

## Option B — Full reproduction (from raw data)

1. **Get the raw data.** Download the *WSDM – KKBox Churn Prediction Challenge*
   data from Kaggle and place the CSVs in `KKBoxData/RawData/`
   (see `KKBoxData/RawData/README.md`).

2. **Convert CSV → Parquet:**
   ```bash
   python prepare_raw_data.py
   ```

3. **Engineer labels + features:** run **`ClassEngineering.ipynb`** end to end.
   This builds the engineered event panel (`KKBoxData/events_engineered.parquet`)
   and the feature contract (`KKBoxData/feature_manifest.json`, already committed).

4. **Pre-compute the "true-cutoff" features** used by the
   time-before-renewal analysis:
   ```bash
   python truecut_precompute.py
   ```
   This rebuilds every base feature at the decision date `E − c` for the
   cutoffs `c ∈ {0, 7, 14, 30, 60}` days.

5. **Run the modelling notebooks** (each writes its run-level results into
   `KKBoxData/` and its figures into `FinalPlots/`):
   - `Classify_01_DirectMulticlass.ipynb`
   - `Classify_02_TwoStage.ipynb`
   - `Classify_02b_TwoStageRegression.ipynb`
   - `Classify_03_DeepLearning.ipynb`

6. **Run the statistical analysis:** `Classify_04_StatisticalAnalysis.ipynb`
   (as in Option A, now over your freshly generated results).

`ClassEDA.ipynb` is exploratory and can be run any time after step 3.

### Reproducibility notes

- The repeated-holdout protocol uses the fixed seeds defined in `clf_utils.py`
  (`RUN_SEEDS = [42, 43, 44, 45, 46]`); repeated-run metrics are reported as
  mean ± standard error over those five seeds.
- Hyper-parameter tuning is cached under `KKBoxData/sweeps/`. The committed
  per-run best-parameter files let the modelling notebooks refit the final
  models without re-searching; delete a sweep file to force a fresh search.
- The deep-learning true-cutoff cell loads per-class decision weights
  (`clf_main_class_weights.npy`) produced earlier in the same notebook; run
  `Classify_03` top to bottom so that artefact exists before the cutoff section.

---

## Data availability

This project uses the publicly available **WSDM – KKBox Churn Prediction
Challenge** dataset (KKBOX Inc., via Kaggle):
<https://www.kaggle.com/competitions/kkbox-churn-prediction-challenge>.
The raw data is not redistributed here and is subject to Kaggle's competition
rules and KKBox's data terms; download it directly from Kaggle. See
`KKBoxData/RawData/README.md` for the required files and conversion step.

## Generative AI Use

I used generative AI tools such as ChatGPT and GitHub Copilot throughout the project as coding and writing support. This included things like debugging, improving or refactoring code, making code more efficient, adding comments, helping with Python syntax, LaTeX and formatting, Markdown explanations in notebooks, code comments, and polishing short sections of writing.

AI was used more heavily when preparing this GitHub repository for submission, mainly to clean up the repo, improve comments and Markdown, organise the files, and make the reproduction steps easier to follow.

I wrote and developed the code used for the project, with AI tools (VSCode GitHub Copilot) used to assist with improving, debugging, refactoring, and documenting that implementation. The actual research and technical work is my own. I decided the research questions, how the dataset and labels were constructed, which models and experiments to run, how the evaluation and leakage controls should work, and how the results were interpreted. Any AI-assisted code was reviewed, tested, and understood by me before being included.



## License

The code, notebooks, and documentation in this repository are released under the
**Creative Commons Attribution 4.0 International (CC-BY-4.0)** license; see
`LICENSE`. This license covers this repository's own material only and does **not**
extend to the KKBox raw dataset.

## Citation

If you refer to this work, please cite the dissertation *"Beyond Binary Churn:
Early Prediction of Subscription Renewal Trajectories"* and the KKBox WSDM Churn
Prediction Challenge dataset.
