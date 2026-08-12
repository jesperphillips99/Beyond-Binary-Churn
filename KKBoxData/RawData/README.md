# Raw KKBox data — not included in this repository

The modelling pipeline is built on the **WSDM – KKBox's Churn Prediction Challenge**
dataset. That raw data is **not redistributed here** because of its size and its
Kaggle competition licence. You must download it yourself to run a full
reproduction.

## 1. Download from Kaggle

Dataset / competition page:
<https://www.kaggle.com/competitions/kkbox-churn-prediction-challenge/data>

You will need a (free) Kaggle account and must accept the competition rules to
access the files. Either download through the website or with the Kaggle CLI:

```bash
kaggle competitions download -c kkbox-churn-prediction-challenge
```

## 2. Required files

Place the following CSV files directly in this folder (`KKBoxData/RawData/`):

| File                 | Approx. rows | Description                              |
|----------------------|--------------|------------------------------------------|
| `members_v3.csv`     | 6.8 M        | User demographics / registration         |
| `transactions.csv`   | 21.5 M       | Subscription transactions (v1)           |
| `transactions_v2.csv`| 1.4 M        | Subscription transactions (v2, March)    |
| `user_logs.csv`      | 392 M        | Daily listening logs (~30 GB)            |
| `user_logs_v2.csv`   | 18.4 M       | Daily listening logs (v2, March)         |

The competition also ships `train.csv`, `train_v2.csv`, `sample_submission_*.csv`
— those are **not** used by this project and can be ignored.

## 3. Convert CSV → Parquet

The notebooks read Parquet, not CSV. From the repository root, run:

```bash
python prepare_raw_data.py
```

This performs a pure format conversion (no feature engineering) and writes
`members_v3.parquet`, `transactions.parquet`, `transactions_v2.parquet`,
`user_logs.parquet`, and `user_logs_v2.parquet` into this folder. The huge
`user_logs` file is streamed in chunks, so conversion stays within a few GB of RAM.
Use `python prepare_raw_data.py --help` for options (`--only`, `--force`,
`--chunksize`).

## 4. Data terms

The KKBox data is provided by KKBOX Inc. through Kaggle and is subject to the
competition rules and KKBox's data terms. Use it only in accordance with those
terms. This repository's own code, notebooks, and documentation are licensed
separately under CC-BY-4.0 (see the top-level `LICENSE`); that licence does **not**
cover the KKBox raw data.
