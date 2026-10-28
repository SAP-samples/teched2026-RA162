# Exercise 2 — TabPFN-3.5 Plus

Predict Miami house prices with TabPFN-3.5 Plus through SAP AI Core, compare with a naïve median and XGBoost, then repeat with your own CSV. The [notebook](TabPFN_notebook_TechEd2026.ipynb) keeps the workflow concise; request handling and baseline implementations live in [helpers.py](helpers.py).

## Setup

The same `.env` file within the `credentials` folder contains all the required credentials for this exercise. 

The helper combines the base URL with the deployment ID. The notebook uses SAP authentication; it does not require a Prior Labs API token or a local GPU. You can run data preparation and the baselines without SAP credentials.

Open the [Python notebook](../ex2/TabPFN_notebook_TechEd2026.ipynb) in the `ex2` folder to get started.

## Exercise 2.1 - Predict Miami house prices

Open the notebook and run its cells in order:

1. Download the Miami housing dataset automatically on first use; subsequent runs use the local `miami_housing.csv` cache.
2. Choose `SALE_PRC`, the sale price in US dollars, as the target. The dataset contains 13,776 rows and 15 input features; its parcel identifier has already been removed.
3. Hold out 20% of rows with seed 42 and use up to `MAX_CONTEXT_ROWS = 1_000_000` training rows as context (all of them for this dataset).
4. Fit the median and XGBoost baselines on that same context.
5. Send context and test features to the SAP TabPFN deployment.
6. Compare MAE, MAPE (%), and RMSE, then inspect the largest errors.
7. Request 10th/50th/90th quantiles, plot central 80% predictive intervals, and visualize the full predictive distribution for one test home.
8. Explain two individual house-price predictions with readable SHAP contribution charts; each cell sends all its perturbations in one batched request.

All methods use the same labelled context and held-out rows. XGBoost's preprocessing is fitted only on context data; its 300-tree configuration is fixed rather than tuned on the test set. TabPFN receives the original feature values. The 1,000,000-row cap matches the SAP TabPFN-3.5 Plus context limit and applies to both the regression and classification sections. Requests follow the [SAP TabPFN-3.5 Plus documentation](https://help.sap.com/docs/sap-ai-core/generative-ai/tabpfn-3-5-plus).

MAE and RMSE are in US dollars; MAPE is displayed as a percentage. Zero targets are excluded from MAPE, which is undefined if every target is zero. Predictive intervals describe model uncertainty and do not guarantee empirical coverage. The full-output plot displays probability mass per bin and zooms to the central 99% of mass to keep extreme tail boundaries from hiding the distribution.

The random split illustrates the workflow. For a real valuation application, consider a chronological or geographic split and verify that each feature was available at the valuation date.

## Exercise 2.2 - Try your own dataset

Change these three values in the notebook's configuration cell:

```python
CSV_PATH = Path("my_data.csv")
TARGET_COLUMN = "sales"
DROP_COLUMNS = ["record_id", "revenue_after_sale"]
```

Use a numeric target and list columns that identify rows, reveal the answer, or become available only after prediction. Set `DROP_COLUMNS = []` if none apply. Re-run from that cell: splitting, baselines, TabPFN predictions, the metric table and error inspection all adapt to your dataset.

For time-dependent data, set `SORT_COLUMN` to your observation date; the notebook parses it as a date, holds out the latest rows, and selects the latest context rows before the cutoff. For repeated entities, use a group-aware split instead of the default random split. The final optional notebook section covers **classification** with its own baselines and metrics.

Missing targets are reported and removed; invalid or infinite targets raise an error. Missing feature values are handled by the helpers. Increase `MAX_CONTEXT_ROWS` to explore how more labelled examples affect accuracy, keeping within your deployment's documented limits. Use a validation set for tuning and keep the test set for the final comparison.

## Data source and attribution

The notebook downloads [Miami housing, OpenML dataset 46942, version 3](https://www.openml.org/search?type=data&id=46942). The data source lists the license as **CC BY-NC-SA**. The downloaded CSV is cached locally and excluded from Git; it is not distributed under this repository's code license.

Reference: Bourassa, Steven C., et al. “Big data, accessibility and urban house prices.” *Urban Studies* 58(15), 2021, pp. 3176–3195. [DOI: 10.1177/0042098020982508](https://doi.org/10.1177/0042098020982508).

## Optional: website phishing classification

The notebook ends with a three-class website phishing example: **Legitimate**, **Phishy**, and **Suspicious**. It compares the majority-class baseline, XGBoost and SAP TabPFN using accuracy, balanced accuracy and macro F1, then displays a confusion matrix. Identical feature patterns stay in the same split to avoid train/test overlap; the seed is 42.

To try your own classification CSV, change `CLASSIFICATION_CSV`, `CLASSIFICATION_TARGET`, and `CLASSIFICATION_DROP_COLUMNS` in that section. All three models use the same context rows. The helpers support string or numeric class labels and categorical or numeric features.

Data: [OpenML dataset 46963, version 1](https://www.openml.org/search?type=data&id=46963), 1,353 rows and nine features, **CC BY 4.0**, derived from [UCI Website Phishing](https://doi.org/10.24432/C5B301). The local CSV cache is excluded from Git.

Reference: Abdelhamid, Neda, Aladdin Ayesh, and Fadi Thabtah. “Phishing detection based associative classification data mining.” *Expert Systems with Applications* 41(13), 2014, pp. 5948–5959. [DOI: 10.1016/j.eswa.2014.03.019](https://doi.org/10.1016/j.eswa.2014.03.019).

## Additional material

[Prediction_Notebook](additional_material/Prediction_Notebook.ipynb) is a self-contained, prediction-only notebook for Miami housing and website phishing. It builds and prints the request bodies and prints five predictions per dataset, without importing `helpers.py`.

## Instructor validation

Run all notebook cells on a CPU worker with the SAP credentials configured. The notebook retains sample outputs, including tables and contribution charts from live validation of the two SHAP cells.

## Single-prediction SHAP

`explain_sap_tabpfn` in `helpers.py` returns a SHAP explanation and request details. It supports one or several finite numeric regression rows, deduplicates all permutation queries, and sends exactly one mean-prediction request for the complete set. OAuth authentication is separate. `plot_shap_explanation` draws one row's waterfall, with contributions in thousands by default. The separate `plot_miami_shap_explanation` helper draws the Miami charts with explicit raises/lowers dollar labels and geographic comparisons against the same SHAP background homes. It only plots existing explanations and makes no requests. `shap==0.49.1` is pinned because the helper verifies and replays its fixed-budget query plan.

The [SHAP benchmark](shap_benchmark/README.md) measures model-agnostic permutation SHAP for one Miami house through the SAP endpoint, with fixed context, batched perturbations, saved attributions, waterfall plots, and request-level timing. It distinguishes context size, SHAP background size, and permutation budget.

## Summary

Once you are completed the exercises in parts 1 and 2, you can head to [Exercise 3 — Tabular orchestration](../ex3/README.md).