"""
Data Loader Module
Supports loading and basic validation for multiple data formats
Including NASA C-MAPSS Turbofan Engine Degradation dataset
"""

import pandas as pd
import numpy as np
import os
from typing import Tuple, List, Optional


class DataLoader:
    """Data loader class"""
    
    # C-MAPSS dataset column names (26 columns)
    CMAPSS_COLUMNS = [
        'unit_id',      # Engine unit number
        'time_cycles',  # Time in cycles
        'op_setting_1', # Operational setting 1
        'op_setting_2', # Operational setting 2
        'op_setting_3', # Operational setting 3
        'sensor_1',     # Sensor measurements 1-21
        'sensor_2',
        'sensor_3',
        'sensor_4',
        'sensor_5',
        'sensor_6',
        'sensor_7',
        'sensor_8',
        'sensor_9',
        'sensor_10',
        'sensor_11',
        'sensor_12',
        'sensor_13',
        'sensor_14',
        'sensor_15',
        'sensor_16',
        'sensor_17',
        'sensor_18',
        'sensor_19',
        'sensor_20',
        'sensor_21',
    ]
    
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
    
    def load_cmapss_data(self, train_file: str, rul_file: str = None, 
                         rul_threshold: int = 30) -> pd.DataFrame:
        """
        Load NASA C-MAPSS Turbofan Engine Degradation dataset
        
        Args:
            train_file: Training data file (e.g., 'train_FD001.txt')
            rul_file: RUL file for test data (e.g., 'RUL_FD001.txt'), optional
            rul_threshold: Cycles threshold for OK/KO classification
                          (RUL <= threshold = KO/Failing, RUL > threshold = OK/Normal)
        
        Returns:
            DataFrame with computed RUL and OK/KO labels
        """
        filepath = os.path.join(self.raw_data_dir, train_file)
        
        try:
            # Load space-separated txt file without headers
            df = pd.read_csv(filepath, sep=r'\s+', header=None)
            
            # Assign column names based on actual number of columns
            if df.shape[1] == 26:
                df.columns = self.CMAPSS_COLUMNS
            else:
                # Handle case with different number of columns
                col_names = self.CMAPSS_COLUMNS[:df.shape[1]]
                if len(col_names) < df.shape[1]:
                    col_names.extend([f'extra_{i}' for i in range(df.shape[1] - len(col_names))])
                df.columns = col_names
            
            # Compute RUL (Remaining Useful Life) for each sample
            df = self._compute_rul(df)
            
            # Create OK/KO labels based on RUL threshold
            # KO = engine is degrading/near failure (low RUL)
            # OK = engine is healthy (high RUL)
            df['OK_KO_Label'] = df['RUL'].apply(
                lambda x: 'KO' if x <= rul_threshold else 'OK'
            )
            
            # IMPORTANT: Drop RUL column to prevent data leakage!
            # RUL is used to create OK_KO_Label, so it cannot be used as a feature
            df = df.drop(columns=['RUL'])
            
            # Drop unit_id as it's just an identifier
            # Keep time_cycles for time-series visualization (but exclude from ML features)
            df = df.drop(columns=['unit_id'])
            
            # Mark time_cycles as the time axis for plotting (metadata)
            df.attrs['time_column'] = 'time_cycles'
            df.attrs['exclude_from_ml'] = ['time_cycles']  # Exclude from ML analysis
            
            print(f"✅ Successfully loaded C-MAPSS data: {train_file}")
            print(f"Data shape: {df.shape}")
            print(f"RUL threshold for OK/KO: {rul_threshold} cycles")
            print(f"OK samples: {(df['OK_KO_Label'] == 'OK').sum()}")
            print(f"KO samples: {(df['OK_KO_Label'] == 'KO').sum()}")
            print(f"ℹ️  Removed RUL and unit_id; kept time_cycles for visualization")
            
            return df
            
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {filepath}")
        except Exception as e:
            raise Exception(f"Error loading C-MAPSS file: {str(e)}")
    
    def _compute_rul(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Remaining Useful Life (RUL) for each sample
        RUL = max_cycle_for_unit - current_cycle
        """
        df = df.copy()
        
        # Get max cycle for each unit (this is the failure point)
        max_cycles = df.groupby('unit_id')['time_cycles'].max().reset_index()
        max_cycles.columns = ['unit_id', 'max_cycle']
        
        # Merge and compute RUL
        df = df.merge(max_cycles, on='unit_id')
        df['RUL'] = df['max_cycle'] - df['time_cycles']
        df = df.drop('max_cycle', axis=1)
        
        return df
    
    def load_file(self, filename: str, **kwargs) -> pd.DataFrame:
        """
        Universal file loader - auto-detects file type
        
        Args:
            filename: File name in data/raw directory
            **kwargs: Additional arguments for specific loaders
        
        Returns:
            Loaded DataFrame
        """
        # Check if it's a C-MAPSS file
        if filename.startswith(('train_FD', 'test_FD')) and filename.endswith('.txt'):
            rul_threshold = kwargs.get('rul_threshold', 30)
            return self.load_cmapss_data(filename, rul_threshold=rul_threshold)
        elif filename.endswith('.txt'):
            # Try to load as space-separated txt
            filepath = os.path.join(self.raw_data_dir, filename)
            df = pd.read_csv(filepath, sep=r'\s+')
            print(f"✅ Successfully loaded txt data: {filename}")
            print(f"Data shape: {df.shape}")
            return df
        elif filename.endswith('.csv'):
            return self.load_csv(filename)
        else:
            raise ValueError(f"Unsupported file format: {filename}")
    
    def get_available_datasets(self) -> List[str]:
        """Get list of available datasets (CSV and C-MAPSS txt files)"""
        if not os.path.exists(self.raw_data_dir):
            return []
        
        # Support both CSV and C-MAPSS txt files
        supported_files = []
        for f in os.listdir(self.raw_data_dir):
            if f.endswith('.csv'):
                supported_files.append(f)
            # C-MAPSS training files (train_FD001.txt, etc.)
            elif f.startswith('train_FD') and f.endswith('.txt'):
                supported_files.append(f)
        
        return sorted(supported_files)
    
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