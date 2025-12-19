from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd

df = df.replace("?", pd.NA)

categorical = [
    "workclass", "education", "marital_status", "occupation",
    "relationship", "race", "native_country"
]

binary = ["sex"]
numerical = ["age", "fnlwgt", "education_num", "hours_per_week"]

# Capital features
df["capital_gain_flag"] = (df["capital_gain"] > 0).astype(int)
df["capital_gain_log"] = np.log1p(df["capital_gain"])

df["capital_loss_flag"] = (df["capital_loss"] > 0).astype(int)
df["capital_loss_log"] = np.log1p(df["capital_loss"])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("bin", "passthrough", binary),
    ]
)
