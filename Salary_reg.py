import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Paths
DATA_PATH = "Churn_Modelling.csv"
ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# 1. Load and inspect dataset
df = pd.read_csv(DATA_PATH)

print("First five rows:")
print(df.head())

print("\nDataset information:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

# 2. Remove identifiers
identifier_columns = [
    col for col in ["RowNumber", "CustomerId", "Surname"]
    if col in df.columns
]
df = df.drop(columns=identifier_columns)

# 3. Define features and target
target = "EstimatedSalary"

if target not in df.columns:
    raise ValueError(
        "The dataset must contain an EstimatedSalary column."
    )

X = df.drop(columns=[target])
y = df[target].astype(float)

# Remove rows with missing target values
valid_rows = y.notna()
X = X.loc[valid_rows].copy()
y = y.loc[valid_rows].copy()

# 4. Split before fitting preprocessing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

numeric_features = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    exclude=["number"]
).columns.tolist()

# 5. Preprocessing pipelines
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# Fit preprocessing on training data only
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Convert sparse matrices if necessary
if hasattr(X_train_processed, "toarray"):
    X_train_processed = X_train_processed.toarray()
if hasattr(X_test_processed, "toarray"):
    X_test_processed = X_test_processed.toarray()

X_train_processed = X_train_processed.astype("float32")
X_test_processed = X_test_processed.astype("float32")

# 6. Build ANN regression model
model = tf.keras.Sequential([
    tf.keras.layers.Input(
        shape=(X_train_processed.shape[1],)
    ),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.15),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1, activation="linear")
])

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

model.summary()

# 7. Train with validation
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train_processed,
    y_train,
    validation_split=0.20,
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

# 8. Evaluate regression performance
predictions = model.predict(
    X_test_processed, verbose=0
).flatten()

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\nRegression Evaluation")
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2:   {r2:.4f}")

# 9. Save model and preprocessing artifacts
model.save(os.path.join(ARTIFACT_DIR, "salary_ann.keras"))

preprocessing_bundle = {
    "preprocessor": preprocessor,
    "feature_columns": X.columns.tolist(),
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "metrics": {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2)
    }
}

joblib.dump(
    preprocessing_bundle,
    os.path.join(ARTIFACT_DIR, "preprocessing_bundle.joblib")
)

print("\nModel and preprocessing objects saved.")

# 10. Reload and verify
loaded_model = tf.keras.models.load_model(
    os.path.join(ARTIFACT_DIR, "salary_ann.keras")
)

loaded_bundle = joblib.load(
    os.path.join(ARTIFACT_DIR, "preprocessing_bundle.joblib")
)

# Predict one test example using reloaded objects
example = X_test.iloc[[0]].copy()
example_processed = loaded_bundle["preprocessor"].transform(example)

if hasattr(example_processed, "toarray"):
    example_processed = example_processed.toarray()

example_processed = example_processed.astype("float32")

example_prediction = loaded_model.predict(
    example_processed, verbose=0
)[0][0]

print("\nExample prediction after reloading:")
print(f"Predicted EstimatedSalary: {example_prediction:.2f}")
print(f"Actual EstimatedSalary:    {y_test.iloc[0]:.2f}")