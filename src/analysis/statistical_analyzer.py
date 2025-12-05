"""
Statistical Analyzer Module
Performs comprehensive statistical analysis between OK and KO groups
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


class StatisticalAnalyzer:
    """Statistical analyzer for OK/KO data comparison"""
    
    def __init__(self):
        self.analysis_results = {}
        self.significance_level = 0.05
        
    def analyze_all_features(self, df: pd.DataFrame, target_col: str = 'OK_KO_Label') -> Dict[str, Any]:
        """
        Comprehensive statistical analysis for all features
        
        Args:
            df: DataFrame with preprocessed data
            target_col: Column name containing OK/KO labels
            
        Returns:
            Dictionary containing complete analysis results
        """
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame")
        
        # Separate OK and KO data
        ok_data = df[df[target_col] == 'OK']
        ko_data = df[df[target_col] == 'KO']
        
        if len(ok_data) == 0 or len(ko_data) == 0:
            raise ValueError("Both OK and KO groups must contain data")
        
        # Get numerical and categorical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Remove target column from analysis
        if target_col in numerical_cols:
            numerical_cols.remove(target_col)
        if target_col in categorical_cols:
            categorical_cols.remove(target_col)
        
        results = {
            'summary': {
                'total_samples': len(df),
                'ok_samples': len(ok_data),
                'ko_samples': len(ko_data),
                'numerical_features': len(numerical_cols),
                'categorical_features': len(categorical_cols)
            },
            'numerical_analysis': self._analyze_numerical_features(ok_data, ko_data, numerical_cols),
            'categorical_analysis': self._analyze_categorical_features(ok_data, ko_data, categorical_cols),
            'feature_ranking': None  # Will be filled later
        }
        
        # Rank features by statistical significance
        results['feature_ranking'] = self._rank_features_by_significance(results)
        
        self.analysis_results = results
        return results
    
    def _analyze_numerical_features(self, ok_data: pd.DataFrame, ko_data: pd.DataFrame, 
                                  numerical_cols: List[str]) -> Dict[str, Any]:
        """Analyze numerical features between OK and KO groups"""
        
        numerical_results = {}
        
        for col in numerical_cols:
            if col not in ok_data.columns or col not in ko_data.columns:
                continue
                
            ok_values = ok_data[col].dropna()
            ko_values = ko_data[col].dropna()
            
            if len(ok_values) == 0 or len(ko_values) == 0:
                continue
            
            # Basic statistics
            ok_stats = self._calculate_basic_stats(ok_values)
            ko_stats = self._calculate_basic_stats(ko_values)
            
            # Statistical tests
            test_results = self._perform_statistical_tests(ok_values, ko_values)
            
            # Effect size (Cohen's d)
            effect_size = self._calculate_cohens_d(ok_values, ko_values)
            
            numerical_results[col] = {
                'ok_stats': ok_stats,
                'ko_stats': ko_stats,
                'statistical_tests': test_results,
                'effect_size': effect_size,
                'difference_ratio': abs(ok_stats['mean'] - ko_stats['mean']) / max(abs(ok_stats['mean']), abs(ko_stats['mean']), 1e-8)
            }
        
        return numerical_results
    
    def _analyze_categorical_features(self, ok_data: pd.DataFrame, ko_data: pd.DataFrame,
                                    categorical_cols: List[str]) -> Dict[str, Any]:
        """Analyze categorical features between OK and KO groups"""
        
        categorical_results = {}
        
        for col in categorical_cols:
            if col not in ok_data.columns or col not in ko_data.columns:
                continue
                
            ok_values = ok_data[col].dropna()
            ko_values = ko_data[col].dropna()
            
            if len(ok_values) == 0 or len(ko_values) == 0:
                continue
            
            # Value counts and distributions
            ok_dist = ok_values.value_counts(normalize=True).to_dict()
            ko_dist = ko_values.value_counts(normalize=True).to_dict()
            
            # Chi-square test
            combined_data = pd.concat([ok_data[[col]], ko_data[[col]]], 
                                    keys=['OK', 'KO']).reset_index(level=0)
            combined_data.columns = ['Group', col]
            
            contingency_table = pd.crosstab(combined_data['Group'], combined_data[col])
            
            try:
                chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
                chi2_result = {
                    'chi2_statistic': chi2_stat,
                    'p_value': p_value,
                    'degrees_of_freedom': dof,
                    'significant': p_value < self.significance_level
                }
            except ValueError:
                chi2_result = {
                    'chi2_statistic': np.nan,
                    'p_value': np.nan,
                    'degrees_of_freedom': np.nan,
                    'significant': False
                }
            
            # Calculate Cramér's V (effect size for categorical variables)
            cramers_v = self._calculate_cramers_v(contingency_table)
            
            categorical_results[col] = {
                'ok_distribution': ok_dist,
                'ko_distribution': ko_dist,
                'chi2_test': chi2_result,
                'cramers_v': cramers_v,
                'contingency_table': contingency_table.to_dict()
            }
        
        return categorical_results
    
    def _calculate_basic_stats(self, values: pd.Series) -> Dict[str, float]:
        """Calculate basic statistical measures"""
        return {
            'count': len(values),
            'mean': float(values.mean()),
            'median': float(values.median()),
            'std': float(values.std()),
            'var': float(values.var()),
            'min': float(values.min()),
            'max': float(values.max()),
            'q1': float(values.quantile(0.25)),
            'q3': float(values.quantile(0.75)),
            'skewness': float(stats.skew(values)),
            'kurtosis': float(stats.kurtosis(values))
        }
    
    def _perform_statistical_tests(self, ok_values: pd.Series, ko_values: pd.Series) -> Dict[str, Any]:
        """Perform various statistical tests"""
        
        results = {}
        
        # T-test (assuming normal distribution)
        try:
            t_stat, t_p_value = stats.ttest_ind(ok_values, ko_values)
            results['t_test'] = {
                't_statistic': float(t_stat),
                'p_value': float(t_p_value),
                'significant': t_p_value < self.significance_level
            }
        except Exception:
            results['t_test'] = {'t_statistic': np.nan, 'p_value': np.nan, 'significant': False}
        
        # Mann-Whitney U test (non-parametric)
        try:
            u_stat, u_p_value = stats.mannwhitneyu(ok_values, ko_values, alternative='two-sided')
            results['mannwhitney_test'] = {
                'u_statistic': float(u_stat),
                'p_value': float(u_p_value),
                'significant': u_p_value < self.significance_level
            }
        except Exception:
            results['mannwhitney_test'] = {'u_statistic': np.nan, 'p_value': np.nan, 'significant': False}
        
        # Kolmogorov-Smirnov test
        try:
            ks_stat, ks_p_value = stats.ks_2samp(ok_values, ko_values)
            results['ks_test'] = {
                'ks_statistic': float(ks_stat),
                'p_value': float(ks_p_value),
                'significant': ks_p_value < self.significance_level
            }
        except Exception:
            results['ks_test'] = {'ks_statistic': np.nan, 'p_value': np.nan, 'significant': False}
        
        return results
    
    def _calculate_cohens_d(self, ok_values: pd.Series, ko_values: pd.Series) -> float:
        """Calculate Cohen's d effect size"""
        try:
            mean_diff = ok_values.mean() - ko_values.mean()
            pooled_std = np.sqrt(((len(ok_values) - 1) * ok_values.var() + 
                                (len(ko_values) - 1) * ko_values.var()) / 
                               (len(ok_values) + len(ko_values) - 2))
            return float(mean_diff / pooled_std) if pooled_std != 0 else 0.0
        except Exception:
            return 0.0
    
    def _calculate_cramers_v(self, contingency_table: pd.DataFrame) -> float:
        """Calculate Cramér's V effect size for categorical variables"""
        try:
            chi2_stat, _, _, _ = stats.chi2_contingency(contingency_table)
            n = contingency_table.sum().sum()
            min_dim = min(contingency_table.shape) - 1
            return float(np.sqrt(chi2_stat / (n * min_dim))) if min_dim > 0 else 0.0
        except Exception:
            return 0.0
    
    def _rank_features_by_significance(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Rank features by their discriminative power"""
        
        feature_scores = []
        
        # Process numerical features
        for feature, analysis in results['numerical_analysis'].items():
            # Combine p-value and effect size for scoring
            mannwhitney_p = analysis['statistical_tests'].get('mannwhitney_test', {}).get('p_value', 1.0)
            effect_size = abs(analysis.get('effect_size', 0.0))
            difference_ratio = analysis.get('difference_ratio', 0.0)
            
            # Calculate composite score (lower p-value and higher effect size = higher score)
            if mannwhitney_p > 0:
                significance_score = -np.log10(mannwhitney_p)
            else:
                significance_score = 10  # Very high significance
            
            composite_score = significance_score * effect_size * difference_ratio
            
            feature_scores.append({
                'feature': feature,
                'type': 'numerical',
                'p_value': mannwhitney_p,
                'effect_size': effect_size,
                'difference_ratio': difference_ratio,
                'composite_score': composite_score,
                'significant': mannwhitney_p < self.significance_level
            })
        
        # Process categorical features
        for feature, analysis in results['categorical_analysis'].items():
            chi2_p = analysis['chi2_test'].get('p_value', 1.0)
            cramers_v = analysis.get('cramers_v', 0.0)
            
            if chi2_p > 0:
                significance_score = -np.log10(chi2_p)
            else:
                significance_score = 10
            
            composite_score = significance_score * cramers_v
            
            feature_scores.append({
                'feature': feature,
                'type': 'categorical',
                'p_value': chi2_p,
                'effect_size': cramers_v,
                'difference_ratio': cramers_v,  # Use Cramér's V as difference measure
                'composite_score': composite_score,
                'significant': chi2_p < self.significance_level
            })
        
        # Sort by composite score (descending)
        feature_scores.sort(key=lambda x: x['composite_score'], reverse=True)
        
        return feature_scores
    
    def get_top_discriminative_features(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get top N most discriminative features"""
        if not self.analysis_results or 'feature_ranking' not in self.analysis_results:
            return []
        
        return self.analysis_results['feature_ranking'][:n]
    
    def get_feature_analysis_summary(self) -> Dict[str, Any]:
        """Get a summary of the feature analysis"""
        if not self.analysis_results:
            return {}
        
        ranking = self.analysis_results.get('feature_ranking', [])
        significant_features = [f for f in ranking if f['significant']]
        
        return {
            'total_features_analyzed': len(ranking),
            'significant_features': len(significant_features),
            'top_5_features': [f['feature'] for f in ranking[:5]],
            'most_discriminative': ranking[0] if ranking else None,
            'numerical_features_count': len([f for f in ranking if f['type'] == 'numerical']),
            'categorical_features_count': len([f for f in ranking if f['type'] == 'categorical'])
        }