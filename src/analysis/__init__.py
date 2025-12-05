"""
Analysis Module Initialization
Provides unified access to statistical and ML-based analysis components
"""

from .statistical_analyzer import StatisticalAnalyzer
from .feature_importance import FeatureImportanceAnalyzer

__all__ = ['StatisticalAnalyzer', 'FeatureImportanceAnalyzer', 'AnalysisEngine']


class AnalysisEngine:
    """Unified analysis engine combining statistical and ML approaches"""
    
    def __init__(self, random_state: int = 42):
        self.statistical_analyzer = StatisticalAnalyzer()
        self.feature_analyzer = FeatureImportanceAnalyzer(random_state=random_state)
        self.results = {}
    
    def analyze_all(self, df, target_col: str = 'OK_KO_Label', time_limit: int = 120, preset: str = 'medium_quality'):
        """
        Run complete analysis pipeline combining statistical and ML approaches
        
        Args:
            df: DataFrame with preprocessed data
            target_col: Target column name
            time_limit: Time limit for AutoGluon training (seconds)
            preset: AutoGluon preset quality ('fast_training', 'medium_quality', 'good_quality', 'high_quality', 'best_quality')
            
        Returns:
            Combined analysis results
        """
        # Statistical analysis
        print("Running statistical analysis...")
        statistical_results = self.statistical_analyzer.analyze_all_features(df, target_col)
        
        # Feature importance analysis using AutoGluon
        print("Running AutoGluon feature importance analysis...")
        importance_results = self.feature_analyzer.analyze_feature_importance(
            df, 
            target_col=target_col,
            time_limit=time_limit,
            preset=preset
        )
        
        # Combine results
        self.results = {
            'statistical_analysis': statistical_results,
            'feature_importance': importance_results,
            'summary': self._create_summary(statistical_results, importance_results)
        }
        
        return self.results
    
    def _create_summary(self, statistical_results, importance_results):
        """Create a summary combining both analysis types"""
        
        # Get top features from statistical analysis
        stat_ranking = statistical_results.get('feature_ranking', [])[:10]
        
        # Get top features from AutoGluon analysis
        ml_feature_importance = importance_results.get('feature_importance', {})
        ml_ranking = ml_feature_importance.get('feature_ranking', [])[:10] if ml_feature_importance else []
        
        # Find overlapping features
        stat_features = set([f['feature'] for f in stat_ranking])
        ml_features = set([f['feature'] for f in ml_ranking])
        overlap = stat_features.intersection(ml_features)
        
        # Get best model info
        best_model = importance_results.get('best_model', {})
        
        # Get model leaderboard count
        leaderboard = importance_results.get('model_leaderboard', [])
        
        return {
            'top_statistical_features': stat_ranking,
            'top_ml_features': ml_ranking,
            'consensus_features': list(overlap),
            'total_features_analyzed': len(statistical_results.get('numerical_analysis', {})) + len(statistical_results.get('categorical_analysis', {})),
            'best_model_info': best_model,
            'analysis_summary': {
                'features_with_statistical_significance': len([f for f in stat_ranking if f.get('significant', False)]),
                'ml_models_evaluated': len(leaderboard),
                'cross_validated_accuracy': best_model.get('score_val', 0) if best_model else 0,
                'training_time': importance_results.get('training_time', 0)
            }
        }