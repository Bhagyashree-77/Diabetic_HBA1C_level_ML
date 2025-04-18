from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import label_binarize
import numpy as np

def run_classification_models(model_name, X_train, X_test, y_train, y_test):
    if model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    elif model_name == "Random Forest":
        model = RandomForestClassifier(n_estimators=100)
    elif model_name == "XGBoost":
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    elif model_name == "LightGBM":
        model = LGBMClassifier()
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # Handle multiclass or binary ROC AUC
    unique_classes = np.unique(y_test)
    if len(unique_classes) > 2:
        y_test_bin = label_binarize(y_test, classes=unique_classes)
        proba = model.predict_proba(X_test)
        auc_score = roc_auc_score(y_test_bin, proba, multi_class='ovr')
    else:
        auc_score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    report = classification_report(y_test, preds)
    return report, auc_score
