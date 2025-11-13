"""
Data Loader Module
Supports loading and basic validation for multiple data formats
"""

import pandas as pd
import os
from typing import Tuple, List, Optional


class DataLoader:
    """Data loader class"""
    
    def __init__(self):
        self.raw_data_dir = "data/raw"
        self.processed_data_dir = "data/processed"
        
    def load_csv(self, filename: str) -> pd.DataFrame:
        """Load CSV file"""
        filepath = os.path.join(self.raw_data_dir, filename)
        try:
            df = pd.read_csv(filepath)
            print(f"✅ Successfully loaded data: {filename}")
            print(f"Data shape: {df.shape}")
            return df
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {filepath}")
        except Exception as e:
            raise Exception(f"Error loading file: {str(e)}")
    
    def get_available_datasets(self) -> List[str]:
        """Get list of available datasets"""
        if not os.path.exists(self.raw_data_dir):
            return []
        
        csv_files = [f for f in os.listdir(self.raw_data_dir) 
                     if f.endswith('.csv')]
        return csv_files
    
    def validate_data_for_analysis(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """Validate if data is suitable for OK/KO analysis"""
        if df.empty:
            return False, "Data is empty"
        
        if df.shape[0] < 10:
            return False, "Too few rows (need at least 10 rows)"
        
        if df.shape[1] < 2:
            return False, "Too few columns (need at least 2 columns)"
        
        return True, "Data validation passed"
    
    def get_column_info(self, df: pd.DataFrame) -> dict:
        """Get column information for label selection"""
        column_info = {}
        
        for col in df.columns:
            dtype = df[col].dtype
            unique_vals = df[col].dropna().unique()
            
            column_info[col] = {
                'dtype': str(dtype),
                'unique_count': len(unique_vals),
                'unique_values': list(unique_vals)[:10],  # Show only first 10 values
                'missing_count': df[col].isnull().sum(),
                'missing_ratio': df[col].isnull().mean()
            }
            
        return column_info
    
    def suggest_label_columns(self, df: pd.DataFrame) -> List[str]:
        """Suggest possible label columns"""
        suggestions = []
        
        for col in df.columns:
            unique_count = df[col].dropna().nunique()
            
            # Suggest columns with 2-10 unique values as possible label columns
            if 2 <= unique_count <= 10:
                suggestions.append(col)
                
        return suggestions