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
