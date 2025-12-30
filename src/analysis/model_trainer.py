"""
Model Training Module
Trains simple discriminative models using top features from AutoGluon
Supports: Logistic Regression, SVM, Decision Tree, Random Forest
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import warnings

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

warnings.filterwarnings("ignore")


class ModelTrainer:
    """
    Train and evaluate simple discriminative models on OK/KO classification.
    Uses top-N features from AutoGluon feature importance ranking.
    """

    def __init__(self, random_state: int = 42, test_size: float = 0.2):
        self.random_state = random_state
        self.test_size = test_size
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()

    def train_models_with_feature_selection(
        self,
        df: pd.DataFrame,
        feature_importance_ranking: List[str],
        target_col: str = "OK_KO_Label",
        feature_counts: List[int] = None,
        model_names: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Train multiple models with different numbers of top features.
        
        Args:
            df: Preprocessed DataFrame
            feature_importance_ranking: List of features ranked by importance (best first)
            target_col: Column name for target labels (OK/KO)
            feature_counts: List of top-N feature counts to try (e.g., [5, 10, 15, 20])
            model_names: List of model types to train (e.g., ['logistic', 'svm', 'dt', 'rf'])
            
        Returns:
            {
                "success": bool,
                "message": str,
                "performance_summary": DataFrame with all metrics,
                "best_model": {name, features, metrics},
                "plot_data": {feature_count_vs_accuracy, model_comparison, etc.},
                "detailed_results": full results dict
            }
        """
        try:
            # Default values
            if feature_counts is None:
                max_features = min(20, len(feature_importance_ranking))
                feature_counts = [5, 10, max(15, max_features), max_features]
                feature_counts = sorted(set([f for f in feature_counts if f > 0]))

            if model_names is None:
                model_names = ["logistic", "svm", "dt", "rf"]

            # Validate target column
            if target_col not in df.columns:
                return {
                    "success": False,
                    "message": f"❌ Target column '{target_col}' not found",
                }

            # Encode labels (OK=1, KO=0)
            label_map = {"OK": 1, "KO": 0}
            y = df[target_col].map(label_map)

            if y.isna().any():
                return {
                    "success": False,
                    "message": "❌ Invalid labels in target column",
                }

            # Train models with varying feature counts
            results = []
            best_accuracy = -1
            best_result = None

            for n_features in feature_counts:
                top_features = feature_importance_ranking[:n_features]
                
                # Filter out features that don't exist in DataFrame
                available_features = [f for f in top_features if f in df.columns]
                if len(available_features) < len(top_features):
                    missing = set(top_features) - set(available_features)
                    print(f"⚠️ Warning: {len(missing)} features not found in DataFrame: {missing}")
                
                X = df[available_features].copy()
                
                # Identify and handle categorical columns (object/string type)
                categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
                
                if categorical_cols:
                    print(f"📋 Found {len(categorical_cols)} categorical columns: {categorical_cols}")
                    # One-hot encode categorical columns
                    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
                    print(f"✅ After encoding: {X.shape[1]} features")
                
                # Handle missing values in numerical columns only
                numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
                if len(numerical_cols) > 0:
                    X[numerical_cols] = X[numerical_cols].fillna(X[numerical_cols].median())
                
                # Verify all columns are numeric
                non_numeric = X.select_dtypes(exclude=['int64', 'float64', 'int32', 'float32', 'bool']).columns
                if len(non_numeric) > 0:
                    raise ValueError(f"❌ Non-numeric columns remain after encoding: {non_numeric.tolist()}")

                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=self.test_size, random_state=self.random_state
                )

                # Scale features
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_test_scaled = self.scaler.transform(X_test)

                # Train each model
                for model_name in model_names:
                    model = self._create_model(model_name)
                    model.fit(X_train_scaled, y_train)

                    # Predictions
                    y_pred = model.predict(X_test_scaled)
                    if hasattr(model, "predict_proba"):
                        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                    else:
                        y_pred_proba = y_pred

                    # Metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred, zero_division=0)
                    precision = precision_score(y_test, y_pred, zero_division=0)
                    recall = recall_score(y_test, y_pred, zero_division=0)

                    try:
                        auc = roc_auc_score(y_test, y_pred_proba)
                    except Exception:
                        auc = None

                    result = {
                        "model": model_name,
                        "n_features": n_features,
                        "features": top_features,
                        "accuracy": float(accuracy),
                        "f1": float(f1),
                        "precision": float(precision),
                        "recall": float(recall),
                        "auc": float(auc) if auc else None,
                        "y_test": y_test.values,
                        "y_pred": y_pred,
                        "y_pred_proba": y_pred_proba,
                        "model_obj": model,
                    }
                    results.append(result)

                    # Track best model
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_result = result

            # Convert to DataFrame for easy viewing
            results_df = pd.DataFrame(
                [
                    {
                        "Model": r["model"],
                        "Features": r["n_features"],
                        "Accuracy": f"{r['accuracy']:.4f}",
                        "F1": f"{r['f1']:.4f}",
                        "Precision": f"{r['precision']:.4f}",
                        "Recall": f"{r['recall']:.4f}",
                        "AUC": f"{r['auc']:.4f}" if r["auc"] else "N/A",
                    }
                    for r in results
                ]
            )

            # Prepare plot data
            plot_data = self._prepare_plot_data(results)

            return {
                "success": True,
                "message": f"✅ Trained {len(results)} models. Best: {best_result['model']} ({best_result['n_features']} features, acc={best_result['accuracy']:.4f})",
                "performance_summary": results_df,
                "best_model": {
                    "name": best_result["model"],
                    "n_features": best_result["n_features"],
                    "features": best_result["features"],
                    "accuracy": best_result["accuracy"],
                    "f1": best_result["f1"],
                    "precision": best_result["precision"],
                    "recall": best_result["recall"],
                    "auc": best_result["auc"],
                    "y_test": best_result["y_test"],
                    "y_pred": best_result["y_pred"],
                },
                "plot_data": plot_data,
                "detailed_results": results,
            }

        except Exception as e:
            return {"success": False, "message": f"❌ Error during training: {str(e)}"}

    def _create_model(self, model_name: str):
        """Factory method to create model instances."""
        models = {
            "logistic": LogisticRegression(
                max_iter=1000, random_state=self.random_state, n_jobs=-1
            ),
            "svm": SVC(kernel="rbf", probability=True, random_state=self.random_state),
            "dt": DecisionTreeClassifier(random_state=self.random_state),
            "rf": RandomForestClassifier(
                n_estimators=100, random_state=self.random_state, n_jobs=-1
            ),
        }
        return models.get(model_name, LogisticRegression(max_iter=1000))

    def _prepare_plot_data(self, results: List[Dict]) -> Dict[str, Any]:
        """
        Prepare data for visualization.
        
        Returns:
            {
                "feature_count_vs_accuracy": {feature_counts, models_data},
                "model_comparison": {model_names, avg_accuracies},
                "confusion_matrices": {model_name: [matrix, feature_count]},
            }
        """
        # Feature count vs accuracy
        feature_counts = sorted(set(r["n_features"] for r in results))
        model_names = sorted(set(r["model"] for r in results))

        feature_vs_acc = {}
        for model in model_names:
            accs = [
                r["accuracy"]
                for r in results
                if r["model"] == model
            ]
            feature_vs_acc[model] = accs

        # Model comparison (average accuracy across feature counts)
        model_comparison = {
            model: float(np.mean([r["accuracy"] for r in results if r["model"] == model]))
            for model in model_names
        }

        return {
            "feature_counts": feature_counts,
            "feature_vs_accuracy": feature_vs_acc,
            "model_comparison": model_comparison,
            "results": results,
        }

    def get_best_model_report(self, best_result: Dict) -> str:
        """Generate detailed classification report for best model."""
        y_test = best_result["y_test"]
        y_pred = best_result["y_pred"]

        report = f"""
🏆 **Best Model: {best_result["name"].upper()}**
- Features Used: {best_result["n_features"]}
- Selected Features: {best_result["features"][:5]}{'...' if len(best_result["features"]) > 5 else ''}

📊 **Performance Metrics**
- Accuracy:  {best_result["accuracy"]:.4f}
- F1 Score:  {best_result["f1"]:.4f}
- Precision: {best_result["precision"]:.4f}
- Recall:    {best_result["recall"]:.4f}
- AUC:       {best_result["auc"]:.4f if best_result["auc"] else 'N/A'}

📋 **Classification Report**
{classification_report(y_test, y_pred, target_names=['KO', 'OK'])}
"""
        return report
