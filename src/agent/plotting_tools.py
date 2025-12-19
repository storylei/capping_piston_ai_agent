"""
Plotting Tools Module
Provides visualization capabilities for statistical analysis
Supports time series, frequency spectrum, and distribution plots
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal, stats
from typing import Dict, List, Optional, Tuple, Any
import io
import base64


class PlottingTools:
    """Tools for generating various plots based on natural language requests"""
    
    def __init__(self):
        # Set default plot style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 10
    
    def plot_time_series(self, df: pd.DataFrame, 
                        column: str,
                        group_by: str = 'OK_KO_Label',
                        title: str = None,
                        separate_groups: bool = True) -> Dict[str, Any]:
        """
        Plot time series data
        
        Args:
            df: DataFrame with data
            column: Column name to plot
            group_by: Column to group by (default 'OK_KO_Label')
            title: Plot title
            separate_groups: Whether to plot OK/KO separately
            
        Returns:
            Dictionary with plot figure and metadata
        """
        try:
            if column not in df.columns:
                return {'error': f"Column '{column}' not found in dataset"}
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            if separate_groups and group_by in df.columns:
                groups = df[group_by].unique()
                for group in groups:
                    group_data = df[df[group_by] == group][column]
                    ax.plot(group_data.index, group_data.values, 
                           label=f'{group}', alpha=0.7, linewidth=1.5)
                ax.legend()
            else:
                ax.plot(df.index, df[column].values, linewidth=1.5)
            
            ax.set_xlabel('Sample Index')
            ax.set_ylabel(column)
            ax.set_title(title or f'Time Series Plot: {column}')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            return {
                'figure': fig,
                'plot_type': 'time_series',
                'column': column,
                'success': True
            }
            
        except Exception as e:
            return {'error': str(e), 'success': False}
    
    def plot_frequency_spectrum(self, df: pd.DataFrame,
                               column: str,
                               group_by: str = 'OK_KO_Label',
                               sampling_rate: float = 1.0,
                               title: str = None) -> Dict[str, Any]:
        """
        Plot frequency spectrum (FFT)
        
        Args:
            df: DataFrame with data
            column: Column name to analyze
            group_by: Column to group by
            sampling_rate: Sampling rate in Hz
            title: Plot title
            
        Returns:
            Dictionary with plot figure and metadata
        """
        try:
            if column not in df.columns:
                return {'error': f"Column '{column}' not found in dataset"}
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            if group_by in df.columns:
                groups = df[group_by].unique()
                for group in groups:
                    group_data = df[df[group_by] == group][column].dropna().values
                    
                    # Compute FFT
                    n = len(group_data)
                    fft_values = np.fft.fft(group_data)
                    fft_freq = np.fft.fftfreq(n, d=1/sampling_rate)
                    
                    # Only plot positive frequencies
                    positive_freq_idx = fft_freq > 0
                    frequencies = fft_freq[positive_freq_idx]
                    magnitude = np.abs(fft_values[positive_freq_idx])
                    
                    ax.plot(frequencies, magnitude, label=f'{group}', alpha=0.7, linewidth=1.5)
                
                ax.legend()
            else:
                data = df[column].dropna().values
                n = len(data)
                fft_values = np.fft.fft(data)
                fft_freq = np.fft.fftfreq(n, d=1/sampling_rate)
                
                positive_freq_idx = fft_freq > 0
                frequencies = fft_freq[positive_freq_idx]
                magnitude = np.abs(fft_values[positive_freq_idx])
                
                ax.plot(frequencies, magnitude, linewidth=1.5)
            
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Magnitude')
            ax.set_title(title or f'Frequency Spectrum: {column}')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            return {
                'figure': fig,
                'plot_type': 'frequency_spectrum',
                'column': column,
                'success': True
            }
            
        except Exception as e:
            return {'error': str(e), 'success': False}
    
    def plot_distribution_comparison(self, df: pd.DataFrame,
                                    column: str,
                                    group_by: str = 'OK_KO_Label',
                                    plot_type: str = 'histogram',
                                    title: str = None) -> Dict[str, Any]:
        """
        Plot distribution comparison between groups
        
        Args:
            df: DataFrame with data
            column: Column name to plot
            group_by: Column to group by
            plot_type: 'histogram', 'kde', 'boxplot', or 'violin'
            title: Plot title
            
        Returns:
            Dictionary with plot figure and metadata
        """
        try:
            if column not in df.columns:
                return {'error': f"Column '{column}' not found in dataset"}
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if group_by not in df.columns:
                return {'error': f"Group column '{group_by}' not found"}
            
            # Check if column is categorical or numerical
            is_categorical = df[column].dtype == 'object' or df[column].nunique() < 10
            
            if plot_type == 'histogram':
                if is_categorical:
                    # For categorical data, use countplot
                    df_clean = df[[column, group_by]].dropna()
                    # Create grouped bar chart
                    cross_tab = pd.crosstab(df_clean[column], df_clean[group_by])
                    cross_tab.plot(kind='bar', ax=ax, alpha=0.7, edgecolor='black')
                    ax.set_ylabel('Count')
                    ax.set_xlabel(column)
                    ax.legend(title=group_by)
                    plt.xticks(rotation=45)
                else:
                    # For numerical data, use histogram
                    groups = df[group_by].unique()
                    for group in groups:
                        group_data = df[df[group_by] == group][column].dropna()
                        ax.hist(group_data, label=f'{group}', alpha=0.6, bins=30, edgecolor='black')
                    ax.legend()
                    ax.set_ylabel('Frequency')
                
            elif plot_type == 'kde':
                groups = df[group_by].unique()
                for group in groups:
                    group_data = df[df[group_by] == group][column].dropna()
                    group_data.plot(kind='kde', ax=ax, label=f'{group}', linewidth=2)
                ax.legend()
                ax.set_ylabel('Density')
                
            elif plot_type == 'boxplot':
                df_clean = df[[column, group_by]].dropna()
                df_clean.boxplot(column=column, by=group_by, ax=ax)
                plt.suptitle('')  # Remove default title
                
            elif plot_type == 'violin':
                df_clean = df[[column, group_by]].dropna()
                sns.violinplot(data=df_clean, x=group_by, y=column, ax=ax)
            
            ax.set_xlabel(column)
            ax.set_title(title or f'Distribution Comparison: {column}')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            return {
                'figure': fig,
                'plot_type': f'distribution_{plot_type}',
                'column': column,
                'success': True
            }
            
        except Exception as e:
            return {'error': str(e), 'success': False}
    
    def plot_feature_comparison(self, df: pd.DataFrame,
                               columns: List[str],
                               group_by: str = 'OK_KO_Label',
                               title: str = None) -> Dict[str, Any]:
        """
        Plot comparison of multiple features between groups
        
        Args:
            df: DataFrame with data
            columns: List of column names to compare
            group_by: Column to group by
            title: Plot title
            
        Returns:
            Dictionary with plot figure and metadata
        """
        try:
            if not columns:
                return {'error': 'No columns specified'}
            
            # Filter valid columns
            valid_columns = [col for col in columns if col in df.columns]
            if not valid_columns:
                return {'error': 'None of the specified columns found in dataset'}
            
            n_cols = len(valid_columns)
            n_rows = (n_cols + 1) // 2
            
            fig, axes = plt.subplots(n_rows, 2, figsize=(14, 5*n_rows))
            if n_cols == 1:
                axes = np.array([axes])
            axes = axes.flatten()
            
            for idx, col in enumerate(valid_columns):
                ax = axes[idx]
                
                if group_by in df.columns:
                    groups = df[group_by].unique()
                    for group in groups:
                        group_data = df[df[group_by] == group][col].dropna()
                        ax.hist(group_data, label=f'{group}', alpha=0.6, bins=20)
                    ax.legend()
                else:
                    ax.hist(df[col].dropna(), bins=20, alpha=0.7)
                
                ax.set_xlabel(col)
                ax.set_ylabel('Frequency')
                ax.set_title(f'{col}')
                ax.grid(True, alpha=0.3)
            
            # Hide unused subplots
            for idx in range(n_cols, len(axes)):
                axes[idx].set_visible(False)
            
            fig.suptitle(title or 'Feature Comparison', fontsize=14, y=1.00)
            plt.tight_layout()
            
            return {
                'figure': fig,
                'plot_type': 'feature_comparison',
                'columns': valid_columns,
                'success': True
            }
            
        except Exception as e:
            return {'error': str(e), 'success': False}
    
    def plot_correlation_heatmap(self, df: pd.DataFrame,
                                columns: List[str] = None,
                                title: str = None) -> Dict[str, Any]:
        """
        Plot correlation heatmap
        
        Args:
            df: DataFrame with data
            columns: Specific columns to include (None for all numerical)
            title: Plot title
            
        Returns:
            Dictionary with plot figure and metadata
        """
        try:
            if columns:
                df_subset = df[columns]
            else:
                df_subset = df.select_dtypes(include=[np.number])
            
            if df_subset.empty:
                return {'error': 'No numerical columns found'}
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            corr_matrix = df_subset.corr()
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                       center=0, ax=ax, square=True)
            
            ax.set_title(title or 'Correlation Heatmap')
            
            plt.tight_layout()
            
            return {
                'figure': fig,
                'plot_type': 'correlation_heatmap',
                'success': True
            }
            
        except Exception as e:
            return {'error': str(e), 'success': False}
    
    def fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string"""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode()
        plt.close(fig)
        return img_base64
