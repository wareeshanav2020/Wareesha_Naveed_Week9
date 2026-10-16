# AVX-DS-CAP — Phase A README

## Project

**E-Commerce Customer Intelligence and Sales Forecasting System**

Phase A builds the analytical foundation using the Brazilian E-Commerce Public Dataset by Olist. It covers data integration, data quality, exploratory data analysis, feature engineering, RFM customer segmentation, low-review prediction, model tuning, and monthly sales forecasting.

> This README covers \*\*Phase A only\*\*. Explainable AI, dashboard/application development, and application testing belong to Phase B and are not claimed as completed here.

## Dataset

Source: Olist Brazilian E-Commerce Public Dataset on Kaggle  
https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

The notebook works with related source tables including customers, geolocation, order items, payments, reviews, orders, products, sellers, and product-category translation. Make sure the data files are available in the paths expected by the notebook before running it.

## Main notebook

* `AVX-Phase-A.ipynb` — executed Phase A analytical notebook.

## Phase A scope

1. Inspect and integrate the Olist source tables.
2. Aggregate item, payment, and review information before order-level joins to reduce row multiplication.
3. Check missing values, duplicates, key monetary values, and order status.
4. Explore sales, categories, reviews, delivery, geography, and payment patterns.
5. Engineer order and customer features.
6. Create RFM-style customer features and evaluate K-Means values from K=2 to K=6.
7. Train and compare a dummy baseline, Random Forest, HistGradientBoosting, and Logistic Regression for low-review classification.
8. Tune Random Forest with GridSearchCV.
9. Evaluate a monthly item-sales forecast using a holdout period.

## Environment setup

Use Python 3.10 or a compatible version. Create a virtual environment if desired.

### Windows

```bash
python -m venv .venv
.venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install pandas numpy matplotlib seaborn scikit-learn scipy statsmodels jupyter ipykernel
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install pandas numpy matplotlib seaborn scikit-learn scipy statsmodels jupyter ipykernel
```

If the notebook imports additional libraries, install the exact missing package shown by the notebook error. The data files must be downloaded separately from Kaggle; they are not embedded in this README.

## Run the notebook

1. Put `AVX-Phase-A.ipynb` in your project folder.
2. Place the downloaded Olist CSV files in the folder structure expected by the notebook, or update the notebook's data paths.
3. Activate the Python environment.
4. Start Jupyter:

```bash
   jupyter notebook
   ```

5. Open `AVX-Phase-A.ipynb`.
6. Select the correct Python kernel.
7. Run all cells from top to bottom and verify that the outputs are produced without errors.
8. Save the executed notebook.

## Key Phase A results

* Final analytical dataset: **99,441 rows**, 30 columns before feature engineering and 47 after feature engineering.
* Duplicate rows and duplicate order IDs: **0**.
* Missing cells: **124,518 (2.664%)**.
* RFM segmentation: **K=2** selected; silhouette score approximately **0.7137**.
* Cluster 1: **2,997 customers (3.12%)**, mean frequency approximately **2.12**, mean monetary value approximately **259.87**.
* Tuned Random Forest test results: accuracy **0.8984**, low-review precision **0.7416**, recall **0.3141**, F1 **0.4413**, ROC AUC **0.7636**.
* Important limitation: the tuned model has a substantial train–test performance gap and low recall for low-review cases.
* Forecasting limitation: September/October 2018 actual values are 145 and 0, and the future forecast becomes negative. Investigate data completeness and improve the model before using it for planning.

All values above should be checked against the executed notebook outputs if the notebook or data are changed.

## Phase A deliverables

Use the actual files produced in your project folder and include only those that exist:

* `AVX-Phase-A.ipynb` — executed notebook.
* Clean analytical dataset CSV.
* Clustered customer dataset CSV with cluster labels.
* Model comparison and tuning outputs.
* Forecast output dataset/results.
* `AVX\_Phase\_A\_Technical\_Report.pdf` — Phase A technical report.
* `README\_Phase\_A.md` — this file.

Do not rename an output or claim it exists unless it has actually been saved.

## Interpretation and limitations

* Category totals reported in Phase A are **item sales**, not profit or margin.
* The low-review target is defined as a mean review score less than or equal to 2.
* The model's prediction point is after delivery but before the review is submitted; it is not an at-checkout or pre-delivery prediction.
* Accuracy alone is misleading because low-review cases are a minority class.
* The model is not ready for unattended operational decisions without further validation and error analysis.
* The forecast should not be used for inventory or staffing decisions until the late-period data anomaly and negative future predictions are resolved.

