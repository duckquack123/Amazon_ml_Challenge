# Multimodal Product Pricing

> A dual-model pricing system that reads product language, product imagery, and structured packaging signals, then blends two complementary regressors into one submission.

## The Solution

Product price is rarely explained by one field. This project combines two views of the same catalog record:

```text
catalog_content ──┬── regex + text statistics ───────────────┐
                  ├── CLIP text embedding (512d) ────────────┤
image_link ───────┴── CLIP image embedding (512d) ───────────┤
                                                             │
                                      SVD: 1,030 -> 256d ────┴── LightGBM

catalog_content ── DistilBERT [CLS] (768d) ──┐
                                              ├── LayerNorm -> MLP -> price
image embedding (512d) ──────────────────────┘

              inverse-validation-SMAPE weighted ensemble
                         -> positive price submission
```

### Branch 1: CLIP + LightGBM

- Extracts quantity, pack/count, weight/volume, text length, word count,
  capital ratio, and digit ratio from `catalog_content`.
- Generates normalized 512-dimensional CLIP text and image embeddings with
  `openai/clip-vit-base-patch32`.
- Uses truncated SVD to compress the combined feature space from 1,030 to 256
  dimensions while retaining the dominant signal.
- Trains LightGBM with shuffled 5-fold validation, early stopping, and modest
  L1/L2 regularization.

This branch is fast, tolerant of missing values, and straightforward to inspect
through feature importance.

### Branch 2: DistilBERT + MLP

- Encodes the catalog text with `distilbert-base-uncased`.
- Concatenates the DistilBERT `[CLS]` representation with precomputed image
  embeddings.
- Applies LayerNorm, dense layers, ReLU, and dropout before the regression head.
- Trains on `log1p(price)` with AdamW and evaluates after converting predictions
  back to the original price scale.

The log transform reduces the influence of expensive outliers and makes the
neural branch better aligned with the relative-error nature of SMAPE.

### Ensemble

Each branch receives a validation SMAPE score. The final prediction gives more
weight to the stronger branch:

```text
w_i = 1 / (SMAPE_i + 1e-8)
price = (w_lgb * price_lgb + w_nn * price_nn) / (w_lgb + w_nn)
```

Predictions are clipped to a minimum of `0.01` and written with the required
`sample_id,price` columns. The submitted report describes this as the intended
final architecture; the repository does not claim a verified final leaderboard
score because one is not recorded in the report.

## Project Map

```text
.
├── README.md                    # Architecture and usage
├── sample_code.py               # Deterministic format-check baseline
├── requirements.txt             # Python dependencies
├── Documentation_template.md   # Challenge documentation template
├── src/
│   ├── Transformer_Code.py      # DistilBERT + image-embedding branch
│   ├── data_preprocessing.py    # Quantity extraction and OCR features
│   ├── generate_plots.py        # Exploratory analysis plots
│   ├── csv_comparison.py        # Submission comparison utility
│   └── utils.py                 # Image download helper
├── main.ipynb                   # Combined experiment notebook
└── src/*.ipynb                  # Supporting experiments and analysis
```

Large challenge inputs, downloaded images, embeddings, trained checkpoints,
plots, archives, and generated submissions are intentionally excluded from Git.
Keep those in local artifact storage or Git LFS rather than committing them to
the source repository.

## Run The Baseline

The baseline is only a submission-format check. It predicts the training-set
median and does not represent the multimodal model:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python sample_code.py
```

Output: `submission/baseline_submission.csv`.

## Run Preprocessing

```bash
python src/data_preprocessing.py \
  --input_file dataset/train.csv \
  --output_file dataset2/processed_for_model.csv
```

The OCR stage uses Tesseract when configured. Set `TESSERACT_CMD` if the
executable is not on `PATH`.

The transformer script reads generated CSV and embedding files from
`dataset2/` and `embeddings/` by default. Override them with
`TRAIN_FILE`, `TEST_FILE`, `TRAIN_EMBEDDINGS_FILE`, and
`TEST_EMBEDDINGS_FILE`.

## Reproducibility Notes

- GPU acceleration is recommended for CLIP and DistilBERT.
- CPU execution is sufficient for preprocessing and LightGBM.
- Failed image downloads use a white placeholder so one unavailable image does
  not stop feature generation.
- External price lookup is prohibited by the challenge and is not part of this
  solution.
- The final report is preserved locally as a generated artifact and is not
  committed because PDFs are excluded by `.gitignore`.

## Data Contract

Training and test records contain `sample_id`, `catalog_content`, and
`image_link`; training records additionally contain `price`. A valid submission
must contain exactly these columns:

```text
sample_id,price
```

with one positive floating-point prediction for every test `sample_id`.
