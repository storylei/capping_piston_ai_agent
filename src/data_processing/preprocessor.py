"""
Data Preprocessor Module
Handle missing values, encode categorical variables, feature scaling, etc.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os


class DataPreprocessor:
    """Data preprocessor class"""
    
    def __init__(self):
        self.processed_data_dir = "data/processed"
        self.scalers = {}
        self.encoders = {}
        
    def create_ok_ko_labels(self, df: pd.DataFrame, label_col: str, 
                           ok_values: Union[str, List[str]], 
                           drop_original: bool = True) -> pd.DataFrame:
        """
        Create OK/KO labels
        Support multi-class to binary-class conversion
        
        Args:
            df: Input DataFrame
            label_col: Name of the original label column
            ok_values: Value(s) to be classified as 'OK'
            drop_original: Whether to drop the original label column (default True)
        
        Returns:
            DataFrame with OK_KO_Label column (and original column removed if drop_original=True)
        """
        df_copy = df.copy()
        
        if isinstance(ok_values, str):
            ok_values = [ok_values]
            
        # Create binary classification labels
        df_copy['OK_KO_Label'] = df_copy[label_col].apply(
            lambda x: 'OK' if x in ok_values else 'KO'
        )
        
        # Drop original label column to avoid data leakage
        if drop_original and label_col in df_copy.columns and label_col != 'OK_KO_Label':
            df_copy = df_copy.drop(columns=[label_col])
            print(f"ℹ️  Dropped original label column: '{label_col}' (to prevent data leakage)")
        
        print(f"✅ OK/KO label creation completed")
        print(f"OK samples: {sum(df_copy['OK_KO_Label'] == 'OK')}")
        print(f"KO samples: {sum(df_copy['OK_KO_Label'] == 'KO')}")
        
        return df_copy
    
    def handle_missing_values(self, df: pd.DataFrame, 
                            strategy: Dict[str, str] = None) -> pd.DataFrame:
        """
        Handle missing values
        
        strategy: dictionary with column names as keys and processing strategies as values
        Available strategies: 'mean', 'median', 'mode', 'drop', 'forward_fill', 'custom_value'
        """
        df_copy = df.copy()
        
        if strategy is None:
            strategy = self._get_default_strategy(df_copy)
            
        for col, method in strategy.items():
            if col not in df_copy.columns:
                continue
                
            if method == 'mean' and df_copy[col].dtype in ['int64', 'float64']:
                df_copy[col].fillna(df_copy[col].mean(), inplace=True)
            elif method == 'median' and df_copy[col].dtype in ['int64', 'float64']:
                df_copy[col].fillna(df_copy[col].median(), inplace=True)
            elif method == 'mode':
                df_copy[col].fillna(df_copy[col].mode()[0], inplace=True)
            elif method == 'forward_fill':
                df_copy[col].fillna(method='ffill', inplace=True)
            elif method == 'drop':
                df_copy.dropna(subset=[col], inplace=True)
            elif method.startswith('custom_'):
                custom_value = method.split('_', 1)[1]
                df_copy[col].fillna(custom_value, inplace=True)
                
        return df_copy
    
    def encode_categorical_variables(self, df: pd.DataFrame, 
                                   encoding_methods: Dict[str, str] = None) -> pd.DataFrame:
        """
        Encode categorical variables
        
        encoding_methods: dictionary with column names as keys and encoding methods as values
        Available methods: 'label', 'onehot', 'target'
        """
        df_copy = df.copy()
        categorical_cols = df_copy.select_dtypes(include=['object']).columns
        
        if encoding_methods is None:
            encoding_methods = {col: 'label' for col in categorical_cols 
                              if col != 'OK_KO_Label'}
        
        for col, method in encoding_methods.items():
            if col not in df_copy.columns or col == 'OK_KO_Label':
                continue
                
            if method == 'label':
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                df_copy[col] = self.encoders[col].fit_transform(
                    df_copy[col].astype(str)
                )
            elif method == 'onehot':
                # One-hot encoding
                dummies = pd.get_dummies(df_copy[col], prefix=col)
                df_copy = pd.concat([df_copy.drop(col, axis=1), dummies], axis=1)
                
        return df_copy
    
    def scale_numerical_features(self, df: pd.DataFrame, 
                               method: str = 'standard') -> pd.DataFrame:
        """
        Scale numerical features
        
        method: 'standard' or 'minmax'
        """
        df_copy = df.copy()
        numerical_cols = df_copy.select_dtypes(include=['int64', 'float64']).columns
        numerical_cols = [col for col in numerical_cols if col != 'OK_KO_Label']
        
        if method == 'standard':
            for col in numerical_cols:
                if col not in self.scalers:
                    self.scalers[col] = StandardScaler()
                df_copy[col] = self.scalers[col].fit_transform(
                    df_copy[col].values.reshape(-1, 1)
                ).flatten()
                
        return df_copy
    
    def save_processed_data(self, df: pd.DataFrame, filename: str) -> str:
        """Save preprocessed data"""
        if not os.path.exists(self.processed_data_dir):
            os.makedirs(self.processed_data_dir)
            
        filepath = os.path.join(self.processed_data_dir, filename)
        df.to_csv(filepath, index=False)
        
        print(f"✅ Preprocessed data saved: {filepath}")
        return filepath
    
    def get_preprocessing_summary(self, original_df: pd.DataFrame, 
                                processed_df: pd.DataFrame) -> Dict:
        """Get preprocessing summary"""
        summary = {
            'original_shape': original_df.shape,
            'processed_shape': processed_df.shape,
            'missing_values_before': original_df.isnull().sum().sum(),
            'missing_values_after': processed_df.isnull().sum().sum(),
            'new_columns': list(set(processed_df.columns) - set(original_df.columns)),
            'removed_columns': list(set(original_df.columns) - set(processed_df.columns))
        }
        return summary
    
    def _get_default_strategy(self, df: pd.DataFrame) -> Dict[str, str]:
        """Get default missing value handling strategy"""
        strategy = {}
        
        for col in df.columns:
            if df[col].isnull().sum() == 0:
                continue
                
            if df[col].dtype in ['int64', 'float64']:
                # Use mean for numerical columns
                strategy[col] = 'mean'
            else:
                # Use mode for categorical columns
                strategy[col] = 'mode'
                
        return strategy