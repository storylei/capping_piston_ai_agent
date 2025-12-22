"""
Feature Importance Analyzer Module (AutoGluon Version)
Uses AutoGluon AutoML to identify feature importance for OK/KO classification
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import warnings
import os

warnings.filterwarnings('ignore')


class FeatureImportanceAnalyzer:
    """AutoGluon based feature importance analyzer"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.predictor = None
        self.feature_importances = {}
        self.model_path = None
        
    def analyze_feature_importance(self, df: pd.DataFrame, target_col: str = 'OK_KO_Label', 
                                   time_limit: int = 120, preset: str = 'medium_quality',
                                   save_path: str = None) -> Dict[str, Any]:
        """
        Comprehensive feature importance analysis using AutoGluon
        
        Args:
            df: DataFrame with preprocessed data
            target_col: Column name containing OK/KO labels
            time_limit: Training time limit in seconds (default 120s)
            preset: Quality preset ('best_quality', 'high_quality', 'medium_quality', 'good_quality', 'fast_training')
            save_path: Path to save the trained predictor (optional)
            
        Returns:
            Dictionary containing feature importance results
        """
        try:
            from autogluon.tabular import TabularPredictor
        except ImportError:
            raise ImportError("AutoGluon not installed. Please run: pip install autogluon")
        
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame")
        
        if df.shape[0] == 0:
            raise ValueError("No valid data for analysis")
        
        # Create a copy and exclude columns that shouldn't be ML features
        df_ml = df.copy()
        
        # Get exclusion list from attrs + common time index columns
        exclude_cols = set(df.attrs.get('exclude_from_ml', []))
        time_index_patterns = ['time_cycles', 'time_cycle', 'cycle', 'timestamp', 'index']
        for col in df_ml.columns:
            if str(col).lower() in time_index_patterns:
                exclude_cols.add(col)
        
        # Drop excluded columns
        cols_to_drop = [c for c in exclude_cols if c in df_ml.columns and c != target_col]
        if cols_to_drop:
            df_ml = df_ml.drop(columns=cols_to_drop)
            print(f"ℹ️  Excluded from ML analysis: {cols_to_drop}")
        
        # Initialize results
        results = {
            'feature_names': [col for col in df_ml.columns if col != target_col],
            'data_shape': df_ml.shape,
            'class_distribution': df_ml[target_col].value_counts().to_dict(),
            'feature_importance': None,
            'model_leaderboard': None,
            'best_model': None,
            'training_time': 0
        }
        
        # Set model path
        if save_path is None:
            save_path = 'autogluon_models_temp'
        self.model_path = save_path
        
        # Train AutoGluon predictor
        print(f"Training AutoGluon predictor with time_limit={time_limit}s, preset={preset}...")
        import time
        start_time = time.time()
        
        self.predictor = TabularPredictor(
            label=target_col,
            eval_metric='accuracy',
            path=save_path,
            verbosity=2
        ).fit(
            train_data=df_ml,  # Use filtered DataFrame
            time_limit=time_limit,
            presets=preset
        )
        
        training_time = time.time() - start_time
        results['training_time'] = training_time
        print(f"Training completed in {training_time:.2f}s")
        
        # Get feature importance
        print("Computing feature importance...")
        feature_importance_df = self.predictor.feature_importance(df_ml)
        results['feature_importance'] = {
            'importance_scores': feature_importance_df.to_dict(),
            'feature_ranking': [
                {'feature': idx, 'importance': float(val), 'rank': i+1} 
                for i, (idx, val) in enumerate(feature_importance_df['importance'].items())
            ]
        }
        
        # Get model leaderboard
        print("Generating model leaderboard...")
        leaderboard = self.predictor.leaderboard(df_ml, silent=True)
        results['model_leaderboard'] = leaderboard.to_dict('records')
        
        # Get best model info
        best_model_name = leaderboard.iloc[0]['model']
        results['best_model'] = {
            'name': best_model_name,
            'score_val': float(leaderboard.iloc[0]['score_val']),
            'score_test': float(leaderboard.iloc[0]['score_test']) if 'score_test' in leaderboard.columns else None,
            'pred_time_val': float(leaderboard.iloc[0]['pred_time_val']) if 'pred_time_val' in leaderboard.columns else None,
            'fit_time': float(leaderboard.iloc[0]['fit_time']) if 'fit_time' in leaderboard.columns else None
        }
        
        self.feature_importances = results
        print("Feature importance analysis completed!")
        return results
    

    
    def get_top_features(self, n: int = 10) -> List[Dict[str, Any]]:
        """
        Get top N important features
        
        Args:
            n: Number of top features to return
            
        Returns:
            List of top features with their importance scores
        """
        if not self.feature_importances or 'feature_importance' not in self.feature_importances:
            return []
        
        ranking = self.feature_importances['feature_importance']['feature_ranking']
        return ranking[:n]
    
    def get_best_model(self) -> Tuple[str, float]:
        """
        Get the best performing model
        
        Returns:
            Tuple of (model_name, accuracy_score)
        """
        if not self.feature_importances or 'best_model' not in self.feature_importances:
            return None, 0.0
        
        best = self.feature_importances['best_model']
        return best['name'], best['score_val']
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained predictor
        
        Args:
            df: DataFrame with features (without target column)
            
        Returns:
            Array of predictions
        """
        if self.predictor is None:
            raise ValueError("Model not trained yet. Call analyze_feature_importance first.")
        
        return self.predictor.predict(df)
    
    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get prediction probabilities
        
        Args:
            df: DataFrame with features (without target column)
            
        Returns:
            DataFrame with class probabilities
        """
        if self.predictor is None:
            raise ValueError("Model not trained yet. Call analyze_feature_importance first.")
        
        return self.predictor.predict_proba(df)
    
    def get_leaderboard(self) -> pd.DataFrame:
        """
        Get model leaderboard as DataFrame
        
        Returns:
            DataFrame with model performance metrics
        """
        if not self.feature_importances or 'model_leaderboard' not in self.feature_importances:
            return pd.DataFrame()
        
        return pd.DataFrame(self.feature_importances['model_leaderboard'])
    
    def save_model(self, path: str = None):
        """
        Save the trained predictor
        
        Args:
            path: Path to save the model (uses default if None)
        """
        if self.predictor is None:
            raise ValueError("No model to save. Train a model first.")
        
        if path:
            self.predictor.save(path)
            print(f"Model saved to {path}")
        else:
            print(f"Model already saved at {self.model_path}")
    
    def load_model(self, path: str):
        """
        Load a saved predictor
        
        Args:
            path: Path to the saved model
        """
        try:
            from autogluon.tabular import TabularPredictor
            self.predictor = TabularPredictor.load(path)
            self.model_path = path
            print(f"Model loaded from {path}")
        except ImportError:
            raise ImportError("AutoGluon not installed. Please run: pip install autogluon")
        except Exception as e:
            raise Exception(f"Failed to load model: {str(e)}")