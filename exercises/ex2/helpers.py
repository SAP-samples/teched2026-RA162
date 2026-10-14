"""Small tabular utilities for Exercise 2 (SAP TabPFN deployment).

Request schema: https://help.sap.com/docs/sap-ai-core/generative-ai/tabpfn-3-5-plus
SAP returns positional predictions, not RPT-style row IDs. Input order is retained.
"""
import json
import os
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split


def download_miami_housing(path="miami_housing.csv"):
    """Cache Miami housing from OpenML dataset 46942 as a local CSV.

    Source license: CC BY-NC-SA; attribution and source links are in README.md.
    """
    path = Path(path)
    if path.exists():
        return path
    response = requests.get(
        "https://data.openml.org/datasets/0004/46942/dataset_46942.pq", timeout=120,
    )
    response.raise_for_status()
    data = pd.read_parquet(BytesIO(response.content))
    if len(data) != 13_776 or "SALE_PRC" not in data.columns:
        raise ValueError("The downloaded Miami housing data has an unexpected schema.")
    temporary = path.with_suffix(path.suffix + ".tmp")
    data.to_csv(temporary, index=False)
    temporary.replace(path)
    return path


def split_regression_data(df, target, drop_columns=(), test_size=0.2,
                          random_state=42, sort_column=None):
    """Split labeled rows; chronological mode holds out the latest rows.

    Missing targets are removed, never imputed. Invalid/infinite targets raise.
    Dates used for chronological sorting must already be datetime or ISO dates.
    The sort column is excluded from features; engineer date features explicitly.
    """
    if not df.columns.is_unique or not all(isinstance(c, str) for c in df.columns):
        raise ValueError("Use unique string column names.")
    if target not in df or target in drop_columns or target == sort_column:
        raise ValueError("Choose a target present in the data and separate from drop/sort columns.")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    missing_targets = int(df[target].isna().sum())
    if missing_targets:
        print(f"Removed {missing_targets} rows with missing targets.")
    data = df.loc[df[target].notna()].copy().reset_index(drop=True)
    data[target] = pd.to_numeric(data[target], errors="raise")
    if len(data) < 3 or not np.isfinite(data[target]).all():
        raise ValueError("Need at least three labeled rows and finite numeric targets.")
    if sort_column:
        if sort_column not in data or data[sort_column].isna().any():
            raise ValueError("The sort column must exist and contain no missing values.")
        data = data.sort_values(sort_column, kind="stable")
        n_test = max(1, int(len(data) * test_size))
        train, test = data.iloc[:-n_test], data.iloc[-n_test:]
    else:
        train, test = train_test_split(data, test_size=test_size, random_state=random_state)
    features = [c for c in data if c not in {target, sort_column, *drop_columns}]
    if not features or train.empty:
        raise ValueError("Need at least one feature and one training row.")
    return train[features].copy(), test[features].copy(), train[target].copy(), test[target].copy()


def select_context(X_train, y_train, max_rows=2048, random_state=42, chronological=False):
    """Use a reproducible sample, or the most recent training rows for time splits."""
    if max_rows < 1 or len(X_train) != len(y_train) or not X_train.index.equals(y_train.index):
        raise ValueError("Context size must be positive; training features and targets must align.")
    if not X_train.index.is_unique:
        raise ValueError("Training row indices must be unique.")
    context = (X_train.tail(max_rows) if chronological else
               X_train.sample(n=min(max_rows, len(X_train)), random_state=random_state))
    return context.copy(), y_train.loc[context.index].copy()


def _baseline_frame(X):
    frame = X.copy()
    for col in frame:
        if pd.api.types.is_numeric_dtype(frame[col]):
            frame[col] = pd.to_numeric(frame[col]).astype(float).replace([np.inf, -np.inf], np.nan)
        else:
            frame[col] = frame[col].map(lambda x: str(x) if pd.notna(x) else np.nan).astype(object)
    return frame


def baseline_predictions(X_context, y_context, X_test, random_state=42):
    """Median + fixed XGBoost baseline; all learned preprocessing fits context only."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder
    from xgboost import XGBRegressor

    train, test = _baseline_frame(X_context), _baseline_frame(X_test)
    if list(train.columns) != list(test.columns):
        raise ValueError("Train and test feature columns must match in order.")
    numeric = train.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in train if c not in numeric]
    preprocess = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median", keep_empty_features=True), numeric),
        ("categorical", make_pipeline(
            SimpleImputer(strategy="constant", fill_value="__MISSING__", keep_empty_features=True),
            OneHotEncoder(handle_unknown="ignore")), categorical),
    ])
    model = make_pipeline(preprocess, XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, objective="reg:squarederror",
        tree_method="hist", n_jobs=2, random_state=random_state,
    ))
    model.fit(train, y_context)
    return {"Median": np.full(len(test), float(np.median(y_context))),
            "XGBoost": model.predict(test)}


def regression_metrics(y_true, predictions):
    """MAE/RMSE in target units; MAPE (%) excludes exactly-zero targets."""
    truth = np.asarray(y_true, dtype=float)
    if truth.ndim != 1 or not len(truth) or not np.isfinite(truth).all():
        raise ValueError("Metric targets must be a finite, nonempty vector.")
    nonzero = truth != 0
    result = {}
    for name, values in predictions.items():
        pred = np.asarray(values, dtype=float)
        if pred.shape != truth.shape or not np.isfinite(pred).all():
            raise ValueError(f"{name}: predictions must match targets and be finite.")
        error = pred - truth
        result[name] = {"MAE": np.abs(error).mean(),
                        "MAPE (%)": (100 * np.abs(error[nonzero] / truth[nonzero]).mean()
                                     if nonzero.any() else np.nan),
                        "RMSE": np.sqrt(np.square(error).mean())}
    return pd.DataFrame.from_dict(result, orient="index").rename_axis("Model")


def dataframe_to_json(frame):
    """Column-oriented JSON data; pandas missing values become JSON null."""
    if not frame.columns.is_unique or not all(isinstance(c, str) for c in frame.columns):
        raise ValueError("Use unique string column names.")
    numeric = frame.select_dtypes(include="number")
    if np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).any():
        raise ValueError("Replace infinite features with missing values or finite values.")
    records = json.loads(frame.to_json(orient="records", date_format="iso"))
    return {column: [row[column] for row in records] for column in frame.columns}


def build_payload(X_context, y_context, X_query, target_name, random_state=42, task_type="regression"):
    """Build the SAP TabPFN schema, including known targets only in y_train."""
    if not len(X_context) or not len(X_query) or not len(X_context.columns):
        raise ValueError("Context and query must contain rows and features.")
    if list(X_context.columns) != list(X_query.columns):
        raise ValueError("Context and query columns must match in order.")
    if target_name in X_context:
        raise ValueError("Remove the target from feature columns.")
    if task_type not in {"regression", "classification"}:
        raise ValueError("Unknown task type.")
    target = np.asarray(y_context, dtype=float if task_type == "regression" else object)
    if target.shape != (len(X_context),) or pd.isna(target).any():
        raise ValueError("Context requires one nonmissing target per row.")
    if task_type == "regression" and not np.isfinite(target).all():
        raise ValueError("Regression targets must be finite.")
    if task_type == "classification":
        target = target.astype(str)
        if len(np.unique(target)) < 2:
            raise ValueError("Classification context needs at least two classes.")
    if isinstance(y_context, pd.Series) and not X_context.index.equals(y_context.index):
        raise ValueError("Context features and target indices must align.")
    return {"task_config": {"task": task_type, "predict_params": {"output_type": "mean" if task_type == "regression" else "preds"},
                            "tabpfn_config": {"random_state": random_state}},
            "x_train": dataframe_to_json(X_context),
            "y_train": {target_name: target.tolist()},
            "x_test": dataframe_to_json(X_query)}


def _response_json(response):
    response.raise_for_status()
    body = response.json()
    if not isinstance(body, dict):
        raise RuntimeError("SAP returned an unexpected response; expected a JSON object.")
    status = body.get("status")
    if body.get("error") or (isinstance(status, dict) and status.get("code", 0) != 0):
        raise RuntimeError("SAP returned an API error. Check deployment status and request configuration.")
    return body


def extract_predictions(body, expected_rows):
    """Validate SAP's positional mean-output vector; never silently flatten matrices."""
    if "prediction" not in body:
        raise RuntimeError("SAP response is missing 'prediction'. Check the deployment and output_type.")
    pred = np.asarray(body["prediction"], dtype=float)
    if pred.shape != (expected_rows,) or not np.isfinite(pred).all():
        raise RuntimeError("SAP predictions must be a finite vector with one value per query row.")
    metadata = body.get("metadata", {})
    if metadata.get("test_set_num_rows", expected_rows) != expected_rows:
        raise RuntimeError("SAP response metadata does not match the query row count.")
    return pred


def _predict_batches(X_context, y_context, X_test, target_name, batch_size=100, random_state=42,
                     output_type="mean", quantiles=None, task_type="regression"):
    """Authenticate once and predict batches using the environment's SAP deployment.

    Required: AICORE_AUTH_URL, AICORE_CLIENT_ID, AICORE_CLIENT_SECRET,
    BASE_DEPLOYMENT_URL (ending /v2/inference/deployments), TABPFN_DEPLOYMENT_ID.
    AICORE_RESOURCE_GROUP defaults to 'default'. No live service is used at import.
    """
    if not isinstance(batch_size, int) or batch_size < 1 or not len(X_test):
        raise ValueError("batch_size must be positive and test data nonempty.")
    required = ("AICORE_AUTH_URL", "AICORE_CLIENT_ID", "AICORE_CLIENT_SECRET",
                "BASE_DEPLOYMENT_URL", "TABPFN_DEPLOYMENT_ID")
    missing = [key for key in required if not os.getenv(key)]
    if missing:
        raise ValueError("Set these environment variables: " + ", ".join(missing))
    auth_url = os.environ["AICORE_AUTH_URL"].rstrip("/")
    if not auth_url.endswith("/oauth/token"):
        auth_url += "/oauth/token"
    url = (os.environ["BASE_DEPLOYMENT_URL"].rstrip("/") + "/" +
           os.environ["TABPFN_DEPLOYMENT_ID"] + "/predict")
    with requests.Session() as session:
        token_response = _response_json(session.post(auth_url, data={
            "grant_type": "client_credentials", "client_id": os.environ["AICORE_CLIENT_ID"],
            "client_secret": os.environ["AICORE_CLIENT_SECRET"]}, timeout=60))
        token = token_response.get("access_token")
        if not token:
            raise RuntimeError("SAP authentication response is missing an access token.")
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json",
                   "AI-Resource-Group": os.getenv("AICORE_RESOURCE_GROUP", "default")}
        batches = []
        for start in range(0, len(X_test), batch_size):
            query = X_test.iloc[start:start + batch_size]
            payload = build_payload(X_context, y_context, query, target_name, random_state, task_type)
            payload["task_config"]["predict_params"]["output_type"] = output_type
            if quantiles is not None:
                payload["task_config"]["predict_params"]["quantiles"] = list(quantiles)
            body = _response_json(session.post(url, headers=headers, json=payload, timeout=300))
            if "prediction" not in body:
                raise RuntimeError("SAP response is missing 'prediction'.")
            if body.get("metadata", {}).get("test_set_num_rows", len(query)) != len(query):
                raise RuntimeError("SAP response metadata does not match the query row count.")
            batches.append((body, query.index))
    return batches


def predict_sap_tabpfn(X_context, y_context, X_test, target_name, batch_size=100, random_state=42):
    """Return positional mean predictions from the SAP deployment configured in .env."""
    batches = _predict_batches(X_context, y_context, X_test, target_name, batch_size, random_state)
    return np.concatenate([extract_predictions(body, len(index)) for body, index in batches])


def predict_sap_tabpfn_quantiles(X_context, y_context, X_test, target_name,
                                quantiles=(0.1, 0.5, 0.9), batch_size=100, random_state=42):
    """Return a frame with one column per quantile, in X_test row order.

    The TabPFN quantile convention is [quantile, query row]. SAP's public
    documentation specifies the request but does not illustrate this response.
    The expected shape is checked explicitly rather than guessing orientation.
    """
    q = np.asarray(quantiles, dtype=float)
    if q.ndim != 1 or not len(q) or not np.isfinite(q).all() or not ((q > 0) & (q < 1)).all():
        raise ValueError("Quantiles must be probabilities strictly between zero and one.")
    if not (np.diff(q) > 0).all():
        raise ValueError("Quantiles must be unique and increasing.")
    batches = _predict_batches(X_context, y_context, X_test, target_name, batch_size,
                               random_state, "quantiles", q)
    frames = []
    for body, index in batches:
        values = np.asarray(body["prediction"], dtype=float)
        if values.shape != (len(q), len(index)) or not np.isfinite(values).all():
            raise RuntimeError("Expected SAP quantile output with shape [quantiles, query rows].")
        if (np.diff(values, axis=0) < 0).any():
            raise RuntimeError("SAP returned crossing quantiles.")
        frames.append(pd.DataFrame(values.T, index=index, columns=q))
    return pd.concat(frames)


def predict_sap_tabpfn_full(X_context, y_context, X_test, target_name, random_state=42):
    """Return the full prediction dictionary for at most 400 query rows.

    Full-output fields (borders/logits) follow the TabPFN client convention;
    SAP's public page has no full-response example; verified live on 2026-10-01.
    """
    if not 1 <= len(X_test) <= 400:
        raise ValueError("SAP full distributions support 1–400 query rows per request.")
    batches = _predict_batches(X_context, y_context, X_test, target_name, len(X_test),
                               random_state, "full")
    full = batches[0][0]["prediction"]
    if not isinstance(full, dict) or not {"borders", "logits"} <= full.keys():
        raise RuntimeError("Expected full prediction fields 'borders' and 'logits'; inspect SAP's response schema.")
    borders, logits = np.asarray(full["borders"], dtype=float), np.asarray(full["logits"], dtype=float)
    if (borders.ndim != 1 or len(borders) < 2 or not np.isfinite(borders).all()
            or not (np.diff(borders) > 0).all() or logits.shape != (len(X_test), len(borders) - 1)):
        raise RuntimeError("Invalid borders/logits shape in SAP full response.")
    return full


def plot_predictive_distribution(full, row=0, ax=None):
    """Plot normalized bin probability masses, including outer tail-bin masses.

    Outer bins include unbounded tails, so this is a probability-mass overview,
    not an exact continuous density or a finite-support claim.
    JSON null logits represent masked bins and are restored to negative infinity.
    """
    import matplotlib.pyplot as plt

    borders = np.asarray(full["borders"], dtype=float)
    logits = np.asarray(full["logits"], dtype=float)
    if (borders.ndim != 1 or len(borders) < 2 or not np.isfinite(borders).all()
            or not (np.diff(borders) > 0).all() or logits.ndim != 2
            or logits.shape[1] != len(borders) - 1 or not 0 <= row < len(logits)):
        raise ValueError("Invalid distribution borders, logits, or row selection.")
    values = np.where(np.isnan(logits[row]), -np.inf, logits[row])
    if np.isposinf(values).any() or not np.isfinite(values).any():
        raise ValueError("Distribution must have at least one finite logit and no positive infinity.")
    probabilities = np.exp(values - values.max())
    probabilities /= probabilities.sum()
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 3))
    ax.bar(borders[:-1], probabilities, width=np.diff(borders), align="edge", alpha=0.7)
    # Extreme tail boundaries can dwarf the interesting range of target values.
    cumulative = np.cumsum(probabilities)
    left = int(np.searchsorted(cumulative, 0.005))
    right = min(int(np.searchsorted(cumulative, 0.995)) + 1, len(borders) - 1)
    ax.set_xlim(borders[left], borders[right])
    ax.set(xlabel="Target value (zoomed to central 99% of mass)", ylabel="Probability mass per bin",
           title="Full predictive distribution: bin probabilities")
    return ax


def download_website_phishing(path="website_phishing.csv"):
    """Cache OpenML dataset 46963 (CC BY 4.0); see README for attribution."""
    path = Path(path)
    if not path.exists():
        response = requests.get(
            "https://data.openml.org/datasets/0004/46963/dataset_46963.pq", timeout=120)
        response.raise_for_status()
        data = pd.read_parquet(BytesIO(response.content))
        if len(data) != 1353 or "WebsiteType" not in data:
            raise ValueError("Unexpected website phishing data schema.")
        temporary = path.with_suffix(path.suffix + ".tmp")
        data.to_csv(temporary, index=False)
        temporary.replace(path)
    return path


def split_classification_data(df, target, drop_columns=(), test_size=0.2,
                              random_state=42, group_identical_features=True):
    """Hold out feature-pattern groups, or stratify rows when grouping is disabled.

    Grouping prevents identical feature vectors leaking between train and test.
    It preserves conflicting labels rather than arbitrarily discarding them.
    Group sizes vary, so the held-out row fraction is approximate.
    """
    from sklearn.model_selection import GroupShuffleSplit

    if not df.columns.is_unique or not all(isinstance(c, str) for c in df.columns):
        raise ValueError("Use unique string column names.")
    if target not in df or target in drop_columns or not 0 < test_size < 1:
        raise ValueError("Choose a valid target and test_size between zero and one.")
    missing = int(df[target].isna().sum())
    if missing:
        print(f"Removed {missing} rows with missing targets.")
    data = df.loc[df[target].notna()].reset_index(drop=True)
    X = data.drop(columns=[target, *drop_columns])
    y = data[target].astype(str)
    if not len(X.columns) or y.nunique() < 2:
        raise ValueError("Need features and at least two target classes.")
    if group_identical_features:
        groups = pd.util.hash_pandas_object(X, index=False)
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train, test = next(splitter.split(X, y, groups))
        result = X.iloc[train], X.iloc[test], y.iloc[train], y.iloc[test]
    else:
        result = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    if set(result[2]) != set(y) or set(result[3]) != set(y):
        raise ValueError("Every class must occur in train and test; adjust the split or add examples.")
    return tuple(part.copy() for part in result)


def select_classification_context(X_train, y_train, max_rows=2048, random_state=42):
    """Reproducible stratified context sample; all models receive these same rows."""
    if max_rows < 1 or not X_train.index.equals(y_train.index) or not X_train.index.is_unique:
        raise ValueError("Use a positive context size and aligned, unique row indices.")
    if len(X_train) <= max_rows:
        return X_train.copy(), y_train.copy()
    context, _, labels, _ = train_test_split(
        X_train, y_train, train_size=max_rows, stratify=y_train, random_state=random_state)
    if set(labels) != set(y_train):
        raise ValueError("Increase max_rows so context contains every class.")
    return context.copy(), labels.copy()


def classification_baseline_predictions(X_context, y_context, X_test, random_state=42):
    """Majority class and XGBoost, with preprocessing fitted only on context."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    from xgboost import XGBClassifier

    train, test = _baseline_frame(X_context), _baseline_frame(X_test)
    if list(train.columns) != list(test.columns):
        raise ValueError("Train and test feature columns must match in order.")
    numeric = train.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in train if c not in numeric]
    preprocess = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median", keep_empty_features=True), numeric),
        ("categorical", make_pipeline(
            SimpleImputer(strategy="constant", fill_value="__MISSING__", keep_empty_features=True),
            OneHotEncoder(handle_unknown="ignore")), categorical),
    ])
    encoder = LabelEncoder()
    labels = encoder.fit_transform(np.asarray(y_context, dtype=str))
    model = make_pipeline(preprocess, XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=2,
        random_state=random_state,
    ))
    model.fit(train, labels)
    majority = pd.Series(y_context).astype(str).mode().iloc[0]
    return {"Majority class": np.full(len(test), majority),
            "XGBoost": encoder.inverse_transform(model.predict(test).astype(int))}


def classification_metrics(y_true, predictions):
    """Accuracy, balanced accuracy (mean class recall), and macro F1; higher is better."""
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    truth = np.asarray(y_true, dtype=str)
    rows = {}
    for name, values in predictions.items():
        pred = np.asarray(values, dtype=str)
        if truth.ndim != 1 or not len(truth) or pred.shape != truth.shape:
            raise ValueError("Predictions and targets must be matching nonempty vectors.")
        rows[name] = {"Accuracy": accuracy_score(truth, pred),
                      "Balanced accuracy": balanced_accuracy_score(truth, pred),
                      "Macro F1": f1_score(truth, pred, average="macro", zero_division=0)}
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("Model")


def predict_sap_tabpfn_classification(X_context, y_context, X_test, target_name,
                                      batch_size=100, random_state=42):
    """Return class labels from SAP using task=classification and output_type=preds."""
    batches = _predict_batches(X_context, y_context, X_test, target_name, batch_size,
                               random_state, output_type="preds", task_type="classification")
    predictions = []
    classes = set(np.asarray(y_context, dtype=str))
    for body, index in batches:
        values = np.asarray(body["prediction"], dtype=object)
        if values.shape != (len(index),) or pd.isna(values).any():
            raise RuntimeError("Expected one nonmissing class label per query row.")
        values = values.astype(str)
        if not set(values) <= classes:
            raise RuntimeError("SAP returned labels absent from the context.")
        predictions.append(values)
    return np.concatenate(predictions)


class _BatchedShapPlan:
    """Collect nonadaptive permutation queries, then replay their predictions."""

    def __init__(self, X_query, background, n_permutations, random_state):
        import shap

        self.query = X_query
        self.background = background
        self.n_permutations = n_permutations
        self.random_state = random_state
        self.calls = []
        self.shap = shap

        def collect(values):
            self.calls.append(np.asarray(values).copy())
            return np.zeros(len(values))

        self._explain(collect)
        # Include original query rows explicitly for an independent additivity check.
        self.rows = np.concatenate([*self.calls, X_query.to_numpy(dtype=float)])
        self.unique_rows, self.inverse = np.unique(self.rows, axis=0, return_inverse=True)

    def _explain(self, predict):
        values, bases = [], []
        state = np.random.get_state()
        try:
            for position in range(len(self.query)):
                row = self.query.iloc[[position]]
                masker = self.shap.maskers.Independent(
                    self.background, max_samples=len(self.background))
                varying = int((~masker.invariants(row.to_numpy()[0])).any(axis=0).sum())
                explainer = self.shap.PermutationExplainer(predict, masker, seed=self.random_state)
                result = explainer(row, max_evals=self.n_permutations * (2 * varying + 1),
                                   batch_size=100000, silent=True)
                values.append(result.values[0])
                bases.append(result.base_values[0])
        finally:
            np.random.set_state(state)
        return self.shap.Explanation(values=np.asarray(values), base_values=np.asarray(bases),
                                     data=self.query.to_numpy(), feature_names=list(self.query.columns))

    def reconstruct(self, predictions):
        predictions = np.asarray(predictions, dtype=float)
        if predictions.shape != (len(self.unique_rows),) or not np.isfinite(predictions).all():
            raise ValueError("Need one finite prediction per unique SHAP query row.")
        expanded = predictions[self.inverse]
        call, offset = 0, 0

        def replay(values):
            nonlocal call, offset
            if call >= len(self.calls) or not np.array_equal(values, self.calls[call]):
                raise RuntimeError("SHAP replay changed its query plan; check the installed SHAP version.")
            result = expanded[offset:offset + len(values)]
            call += 1
            offset += len(values)
            return result

        explanation = self._explain(replay)
        if call != len(self.calls) or offset != len(expanded) - len(self.query):
            raise RuntimeError("SHAP replay did not consume the complete prediction plan.")
        return explanation, expanded[offset:]


def explain_sap_tabpfn(X_context, y_context, X_query, target_name,
                       background_size=32, n_permutations=32, random_state=42):
    """Explain one or several numeric regression rows in ONE SAP prediction request.

    Returns (shap.Explanation, details). The context stays fixed. Hidden features
    are replaced by values from a reproducible context sample. All permutation
    queries for ALL requested rows are deduplicated and sent together, using mean
    output. OAuth authentication is separate from the single prediction request.

    Requires finite numeric features and shap==0.49.1. The fixed-budget query plan
    is independent of predictions; a verified replay preserves the estimator and
    background weights. Each row uses the same seed, so explaining it alone uses
    the same permutations. Attributions are approximate and not causal effects.
    Large query plans are held in RAM; this helper raises rather than silently
    splitting a plan exceeding the documented row/cell limits.
    """
    import time

    tick = time.perf_counter()
    if (not isinstance(background_size, int) or background_size < 1
            or not isinstance(n_permutations, int) or n_permutations < 1):
        raise ValueError("background_size and n_permutations must be positive integers.")
    # Reuse the prediction payload's context, target and column validation.
    build_payload(X_context, y_context, X_query.iloc[:1], target_name, random_state)
    for name, frame in (("Context", X_context), ("Query", X_query)):
        if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in frame.dtypes):
            raise ValueError(f"{name} SHAP features must be numeric; encode categories explicitly.")
        if not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise ValueError(f"{name} SHAP features must be finite and nonmissing.")
    background = X_context.sample(n=min(background_size, len(X_context)), random_state=random_state)
    plan = _BatchedShapPlan(X_query.astype(float), background.astype(float),
                            n_permutations, random_state)
    query = pd.DataFrame(plan.unique_rows, columns=X_context.columns)
    # SAP's mean-output limits are distinct from its 400-row full-output limit.
    context_rows, columns, query_rows = len(X_context), len(X_context.columns), len(query)
    row_limit = 250000 if context_rows == 1000000 else 1000000
    cells = context_rows * (columns + 1) + query_rows * columns
    if context_rows > 1000000 or columns > 2000 or query_rows > row_limit or cells > 200000000:
        raise ValueError("SHAP plan exceeds SAP request limits; reduce rows, background or permutations.")
    planning_seconds = time.perf_counter() - tick
    request_tick = time.perf_counter()
    batches = _predict_batches(X_context, y_context, query, target_name,
                               batch_size=query_rows, random_state=random_state)
    request_seconds = time.perf_counter() - request_tick
    if len(batches) != 1:
        raise RuntimeError("Expected exactly one batched SHAP prediction request.")
    prediction = extract_predictions(batches[0][0], query_rows)
    explanation, original_predictions = plan.reconstruct(prediction)
    residual = explanation.base_values + explanation.values.sum(axis=1) - original_predictions
    details = {"prediction_requests": 1, "explained_rows": len(X_query),
               "background_rows": len(background), "n_permutations": n_permutations,
               "planned_rows": len(plan.rows), "unique_query_rows": query_rows,
               "planning_seconds": planning_seconds,
               "request_seconds_including_auth": request_seconds,
               "total_seconds": time.perf_counter() - tick,
               "predictions": original_predictions, "additivity_residual": residual,
               "source_indices": X_query.index.tolist()}
    return explanation, details


def plot_shap_explanation(explanation, row=0, actual=None, target_name="Sale price ($)", scale=1000):
    """Waterfall of signed contributions, scaled for readable monetary labels."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, MaxNLocator
    import shap

    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be positive and finite.")
    selected = explanation[row]
    plotted = shap.Explanation(values=selected.values / scale,
                               base_values=selected.base_values / scale,
                               data=selected.data, feature_names=selected.feature_names)
    shap.plots.waterfall(plotted, max_display=min(10, len(selected.feature_names)), show=False)
    figure = plt.gcf()
    figure.set_size_inches(11, figure.get_figheight())
    ax = figure.axes[0]
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    ax.set_xlabel(target_name if scale == 1 else f"{target_name} / {scale:,.0f}")
    if actual is not None:
        ax.axvline(float(actual) / scale, color="black", linestyle="--", linewidth=1.2,
                   label=f"Actual: {actual:,.0f}")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12))
    plt.tight_layout()
    return ax


_SHAP_BLUE, _SHAP_RED, _SHAP_INK, _SHAP_MUTED = "#1674ad", "#c34d42", "#182d40", "#566879"
_MIAMI_SHAP_NAMES = {
    "LATITUDE": "North–south location (latitude)",
    "LONGITUDE": "East–west location (longitude)",
    "TOT_LVG_AREA": "Living area", "LND_SQFOOT": "Land area",
    "SPEC_FEAT_VAL": "Assessed value of special features",
    "OCEAN_DIST": "Distance to ocean", "WATER_DIST": "Distance to nearest water body",
    "CNTR_DIST": "Distance to central Miami business district",
    "SUBCNTR_DI": "Distance to nearest business subcenter",
    "RAIL_DIST": "Distance to railway", "HWY_DIST": "Distance to highway",
    "age": "Building age", "structure_quality": "Structure quality (dataset code)",
    "avno60plus": "High aircraft-noise indicator", "month_sold": "Month sold",
}


def _shap_impact(value):
    direction = "Raises" if value > 0 else "Lowers" if value < 0 else "No change:"
    return f"{direction} ${abs(value) / 1000:,.1f}k"


def _describe_miami_feature(name, value, reference):
    """Separate physical feature values from model contribution direction."""
    median = float(reference[name].median())
    if name == "LATITUDE":
        where = "north" if value > median else "south" if value < median else "at the latitude"
        return f"{value:.3f}°N · {where} of reference median ({median:.3f}°N)"
    if name == "LONGITUDE":
        where = "east" if value > median else "west" if value < median else "at the longitude"
        return f"{abs(value):.3f}°W · {where} of reference median ({abs(median):.3f}°W)"
    if name.endswith("_DIST") or name == "SUBCNTR_DI":
        where = "closer" if value < median else "farther" if value > median else "equal"
        return f"{value:,.0f} ft · {where} than reference median ({median:,.0f} ft)" if where != "equal" else f"{value:,.0f} ft · at reference median"
    if name in ("TOT_LVG_AREA", "LND_SQFOOT"):
        return f"{value:,.0f} sq ft · reference median {median:,.0f} sq ft"
    if name == "SPEC_FEAT_VAL":
        return f"${value:,.0f} · reference median ${median:,.0f}"
    if name == "age":
        return f"{value:.0f} years · reference median {median:.0f} years"
    if name == "avno60plus":
        return "Yes" if value else "No"
    if name == "month_sold":
        import calendar
        return calendar.month_name[int(value)]
    return f"Code {value:g} · reference median code {median:g}"


def _plot_readable_miami_shap(values, base, data, feature_names, reference, prediction,
                       actual, row_position, source_row, limit=300):
    """Plot one explanation in dollars; aggregate undisplayed features exactly."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, MaxNLocator
    values = np.asarray(values, dtype=float)
    names = list(feature_names)
    if not np.isfinite(values).all() or not np.isfinite([base, prediction, actual, limit]).all() or limit <= 0:
        raise ValueError("Expected finite SHAP values, prices, and a positive axis limit.")
    order = np.argsort(-np.abs(values), kind="stable")
    visible = set(order[:7]) | {names.index(n) for n in
                              ("WATER_DIST", "LATITUDE", "LONGITUDE", "OCEAN_DIST")}
    selected = [int(i) for i in order if i in visible]
    rows = [(_MIAMI_SHAP_NAMES.get(names[i], names[i]), _describe_miami_feature(names[i], data[i], reference),
             float(values[i])) for i in selected]
    hidden = [i for i in range(len(names)) if i not in visible]
    if hidden:
        rows.append((f"Remaining {len(hidden)} features combined", "Net contribution of smaller effects",
                     float(values[hidden].sum())))
    assert np.isclose(sum(r[2] for r in rows), values.sum())
    limit = max(limit, float(np.ceil(max(abs(r[2]) for r in rows) / 50000) * 50))
    net = float(values.sum())
    residual = float(base + net - prediction)

    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 14, "text.parse_math": False,
                         "text.color": _SHAP_INK, "axes.labelcolor": _SHAP_MUTED}):
        fig = plt.figure(figsize=(16, 11.8), facecolor="white")
        fig.text(.04, .957, f"Why is this home's predicted price ${prediction / 1000:,.1f}k?",
                 fontsize=28, weight="bold")
        fig.text(.04, .923, f"Miami-Dade · test position {row_position} · dataset row {source_row}",
                 fontsize=15, color=_SHAP_MUTED)
        for x, title, number in [(.04, "REFERENCE PREDICTION", f"${base / 1000:,.1f}k"),
                                 (.34, "TOTAL FEATURE CONTRIBUTION", f"{'+' if net >= 0 else '−'}${abs(net) / 1000:,.1f}k"),
                                 (.68, "PREDICTED PRICE", f"${prediction / 1000:,.1f}k")]:
            fig.text(x, .877, title, fontsize=13, color=_SHAP_MUTED, weight="bold")
            fig.text(x, .837, number, fontsize=31, weight="bold",
                     color=(_SHAP_RED if net > 0 else _SHAP_BLUE) if x == .34 else _SHAP_INK)
        fig.text(.287, .837, "+", fontsize=28, color=_SHAP_MUTED)
        fig.text(.625, .837, "≈", fontsize=28, color=_SHAP_MUTED)
        fig.text(.87, .844, f"Actual sale\n${actual / 1000:,.1f}k", fontsize=15, color=_SHAP_MUTED)

        ax = fig.add_axes([.455, .14, .335, .60])
        ax.set_ylim(len(rows) - .5, -.5)
        ax.set_xlim(-limit, limit)
        ax.axvspan(-limit, 0, color=_SHAP_BLUE, alpha=.045)
        ax.axvspan(0, limit, color=_SHAP_RED, alpha=.045)
        ax.axvline(0, color="#8293a1", linewidth=1)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "0" if v == 0 else f"{'−' if v < 0 else '+'}${abs(v):,.0f}k"))
        ax.tick_params(axis="x", colors=_SHAP_MUTED, labelsize=13)
        ax.set_yticks([])
        ax.set_xlabel("Contribution to predicted price (US dollars)", labelpad=12, fontsize=14)
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.text(.04, .772, "FEATURE + THIS HOME'S VALUE", weight="bold", fontsize=13, color=_SHAP_MUTED)
        fig.text(.455, .772, "← LOWERS PRICE", weight="bold", fontsize=14, color=_SHAP_BLUE)
        fig.text(.79, .772, "RAISES PRICE →", weight="bold", fontsize=14, color=_SHAP_RED, ha="right")
        fig.text(.82, .772, "MODEL CONTRIBUTION", weight="bold", fontsize=13, color=_SHAP_MUTED)
        for y, (label, detail, value) in enumerate(rows):
            color = _SHAP_RED if value > 0 else _SHAP_BLUE
            ax.barh(y, value / 1000, height=.39, color=color, zorder=3)
            fy = .14 + .60 * (1 - (y + .5) / len(rows))
            fig.text(.04, fy + .010, label, fontsize=15, weight="bold", va="center")
            fig.text(.04, fy - .015, detail, fontsize=13, color=_SHAP_MUTED, va="center")
            fig.text(.82, fy, _shap_impact(value), fontsize=15, weight="bold", color=color, va="center")

        water = names.index("WATER_DIST")
        latitude = names.index("LATITUDE")
        fig.text(.04, .032, f"For this home: water distance {_shap_impact(values[water]).lower()}; latitude {_shap_impact(values[latitude]).lower()}.",
                 fontsize=17, weight="bold")
        return fig, {"row_position": row_position, "source_row": source_row,
                     "reference_prediction": float(base), "prediction": float(prediction),
                     "actual": float(actual), "net_contribution": net,
                     "additivity_residual_dollars": residual,
                     "rows": [{"feature": a, "context": b, "contribution": c} for a, b, c in rows]}



def plot_miami_shap_explanation(explanation, background, *, prediction, actual,
                                row=0, row_position=None, source_row=None, limit=300):
    """Readable Miami house-price SHAP chart; returns a Matplotlib figure.

    Pass the exact background sample used by explain_sap_tabpfn, so geographic
    and distance comparisons describe the same reference homes. Contributions
    and prices are supplied in dollars; limit is in thousands of dollars and
    expands when needed. This function only plots; it makes no model requests.
    Feature definitions: https://github.com/ModelOriented/shapviz/blob/main/R/data.R
    Use plot_shap_explanation for other datasets/targets.
    """
    selected = explanation[row]
    fig, _ = _plot_readable_miami_shap(
        selected.values, float(selected.base_values), selected.data,
        selected.feature_names, background, prediction, actual,
        row_position, source_row, limit,
    )
    return fig
