import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SklearnPipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, roc_curve, auc
import joblib
import os
from collections import Counter

# Set MLflow experiment name
EXPERIMENT_NAME = "Weather_Prediction_Optimized"

def prepare_data(file_path="C:/Users/LENOVO/Desktop/S2/Python for D 2/TrafficFlow/data/weather.csv"):
    if not os.path.exists(file_path):
        # Fallback to relative path if absolute doesn't work
        file_path = "data/weather.csv"
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path)
    
    # Clean data: Extract numbers from strings (e.g. "20 °C" -> 20.0)
    df_clean = pd.DataFrame()
    for col in df.columns:
        # Regex to find float/int (handles negative numbers and comma/dot decimals)
        extracted = df[col].astype(str).str.extract(r'(-?\d+[.,]?\d*)')[0].str.replace(',', '.')
        try:
            # Convert to float and add if valid
            series = extracted.astype(float)
            if series.notna().sum() > 0:
                df_clean[col] = series
        except:
            continue
            
    # Define Target: Use 'Temp' column if available, else last column
    if not df_clean.empty:
        target_col = next((c for c in df_clean.columns if 'Temp' in c), df_clean.columns[-1])
    else:
        raise ValueError("Cleaned dataframe is empty. Check scraping and cleaning logic.")

    # Drop rows where the target is missing
    df = df_clean.dropna(subset=[target_col])

    if df.shape[0] < 10: # Ensure there's enough data to model
        raise ValueError(f"Not enough data for modeling after cleaning. Only {df.shape[0]} rows available.")
    
    # Create Binary Classification Target (Above Median vs Below)
    y = (df[target_col] > df[target_col].median()).astype(int)
    X = df.drop(columns=[target_col])
    
    numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Create a pipeline for numerical features to handle missing values and scale
    numeric_transformer = SklearnPipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y), preprocessor

def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix: {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    filename = f"data/confusion_matrix_{model_name}.png"
    plt.savefig(filename)
    plt.close()
    return filename

def plot_roc_curve(y_true, y_prob, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve: {model_name}')
    plt.legend(loc="lower right")
    filename = f"data/roc_curve_{model_name}.png"
    plt.savefig(filename)
    plt.close()
    return filename

def run_optimized_experiment(model_name, model, param_grid):
    (X_train, X_test, y_train, y_test), preprocessor = prepare_data()
    
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    with mlflow.start_run(run_name=model_name):
        # Custom handling for Stacking
        min_class_count = min(Counter(y_train).values())

        if model == "STACKING_PLACEHOLDER":
            # StackingCV needs at least 2 splits.
            stacking_cv_splits = min(5, min_class_count)
            if stacking_cv_splits < 2:
                print(f"Skipping Stacking model: not enough samples in smallest class ({min_class_count}) for internal CV.")
                return None, -1

            base_learners = [
                ('rf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)),
                ('xgb', XGBClassifier(n_estimators=100, max_depth=3, use_label_encoder=False, eval_metric='logloss', random_state=42))
            ]
            model = StackingClassifier(
                estimators=base_learners,
                final_estimator=LogisticRegression(),
                cv=stacking_cv_splits
            )

        # Build Imbalanced Pipeline (SMOTE happens only during fit)
        pipeline = ImbPipeline(steps=[
            ('preprocessor', preprocessor),
            # k_neighbors=1 to handle small datasets (like daily weather scrapes)
            ('smote', SMOTE(random_state=42, k_neighbors=1)),
            ('classifier', model)
        ])
        
        # Determine the number of splits for CV dynamically
        cv_splits = min(3, min_class_count)
        if cv_splits < 2:
            print(f"Skipping {model_name}: not enough samples in smallest class ({min_class_count}) to perform cross-validation.")
            return None, -1

        # Grid Search
        # Note: parameters should be prefixed with 'classifier__'
        grid_search = GridSearchCV(pipeline, param_grid, cv=cv_splits, scoring='f1', verbose=1, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        # Log Best Parameters
        mlflow.log_params(best_params)
        
        # Predictions
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else None
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        
        print(f"[{model_name}] Best Params: {best_params}")
        print(f"[{model_name}] Accuracy: {acc:.4f}, F1: {f1:.4f}")
        
        # Plots
        cm_path = plot_confusion_matrix(y_test, y_pred, model_name)
        mlflow.log_artifact(cm_path)
        
        if y_prob is not None:
            roc_path = plot_roc_curve(y_test, y_prob, model_name)
            mlflow.log_artifact(roc_path)
            
        # Log Model
        mlflow.sklearn.log_model(best_model, "model")
        
        return best_model, f1

if __name__ == "__main__":
    # Define models and grids
    models_config = {
        "RandomForest_SMOTE": {
            "model": RandomForestClassifier(random_state=42),
            "params": {
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [10, 20, None],
                'classifier__min_samples_split': [2, 5]
            }
        },
        "XGBoost_SMOTE": {
            "model": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
            "params": {
                'classifier__learning_rate': [0.01, 0.1],
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [3, 6]
            }
        },
        "Stacking_Ensemble": {
            "model": "STACKING_PLACEHOLDER",  # Will handle in run_optimized_experiment
            "params": {
                'classifier__final_estimator__C': [0.1, 1.0, 10.0]
            }
        }
    }
    
    best_overall_model = None
    best_overall_f1 = -1
    
    for name, config in models_config.items():
        print(f"\nRunning {name}...")
        model, f1 = run_optimized_experiment(name, config["model"], config["params"])
        if model is None:
            continue
        if f1 > best_overall_f1:
            best_overall_f1 = f1
            best_overall_model = model
            
    # Save best overall model
    if best_overall_model:
        joblib.dump(best_overall_model, "data/best_model_pipeline.pkl")
        print(f"\nBest model saved: F1={best_overall_f1:.4f}")