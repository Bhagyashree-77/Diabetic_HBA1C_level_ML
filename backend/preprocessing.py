import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, roc_auc_score
from scipy.sparse import issparse
from scipy.sparse import issparse, csr_matrix, csc_matrix
def load_and_preprocess_data(filepath, return_preprocessor=False):
    df = pd.read_csv(filepath)

    # Drop uninformative columns
    df.drop(columns=["encounter_id", "patient_nbr", "weight", "examide", "citoglipton"], errors='ignore', inplace=True)

    # Replace '?' with NaN
    df.replace('?', np.nan, inplace=True)

    # Filter for valid HbA1c levels
    df = df[df['A1Cresult'].isin(['Norm', '>7', '>8'])].copy()

    # Encode A1Cresult
    a1c_encoder = OrdinalEncoder(categories=[['Norm', '>7', '>8']])
    df['A1C_encoded'] = a1c_encoder.fit_transform(df[['A1Cresult']])

    target_col = 'A1C_encoded'
    X = df.drop(columns=['A1Cresult', target_col])
    y = df[target_col].astype(int)

    # Column types
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

    # Pipelines
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # Returns sparse matrix
    ])

    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, numerical_cols),
        ('cat', categorical_pipeline, categorical_cols)
    ])

    # Fit transform
    X_processed = preprocessor.fit_transform(X)

    X_processed = np.array(X_processed, dtype=np.float32)
    print(f"✅ Preprocessed data shape: {X_processed.shape} | Type: {type(X_processed)}")

    # Isolation Forest for outlier removal
    iso = IsolationForest(contamination=0.01, random_state=42)
    mask = iso.fit_predict(X_processed) == 1
    X_filtered = X_processed[mask]
    y_filtered = y[mask]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_filtered, y_filtered, test_size=0.2, random_state=42, stratify=y_filtered)

    if return_preprocessor:
        return (X_train, X_test, y_train, y_test), preprocessor
    else:
        return X_train, X_test, y_train, y_test


# Load and train specified model
def get_trained_model(model_name, X_train, y_train):
    if model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    elif model_name == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_name == "XGBoost":
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    elif model_name == "LightGBM":
        model = LGBMClassifier()
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    model.fit(X_train, y_train)
    return model

# For exporting the fitted preprocessor to use in inference
def get_fitted_preprocessor():
    (_, _, _, _), preprocessor = load_and_preprocess_data("diabetic_data.csv", return_preprocessor=True)
    return preprocessor
