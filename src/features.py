"""
Feature preparation and the scikit-learn preprocessing pipeline.

Everything that transforms the raw customer columns lives INSIDE the model
pipeline. That way the API can send raw customer data and the exact same
transformations are applied at training time and at prediction time.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src import config


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Clean undocumented category codes and add three simple derived features.

    - EDUCATION codes 0, 5 and 6 are undocumented -> grouped into 4 ("others").
    - MARRIAGE code 0 is undocumented -> grouped into 3 ("others").
    - AVG_BILL_AMT: average bill over the last 6 months.
    - AVG_PAY_AMT: average payment over the last 6 months.
    - NUM_DELAYED_PAYMENTS: in how many of the last 6 months the customer
      was late (repayment status > 0).
    """
    df = df.copy()

    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})

    df["AVG_BILL_AMT"] = df[config.BILL_FEATURES].mean(axis=1)
    df["AVG_PAY_AMT"] = df[config.PAY_AMOUNT_FEATURES].mean(axis=1)
    df["NUM_DELAYED_PAYMENTS"] = (df[config.PAY_STATUS_FEATURES] > 0).sum(axis=1)
    return df


def build_preprocessor() -> Pipeline:
    """Feature creation + scaling of numeric columns + one-hot of categoricals."""
    column_transformer = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), config.NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                config.CATEGORICAL_FEATURES,
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("prepare", FunctionTransformer(prepare_features)),
            ("columns", column_transformer),
        ]
    )


def build_model_pipeline(estimator) -> Pipeline:
    """Full pipeline: preprocessing followed by the classifier."""
    return Pipeline(
        steps=[
            ("preprocessing", build_preprocessor()),
            ("classifier", estimator),
        ]
    )
