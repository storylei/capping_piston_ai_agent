"""
Advanced Analysis Tab - AutoGluon feature importance analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from analysis import StatisticalAnalyzer
from analysis.feature_importance import FeatureImportanceAnalyzer


def display():
    """Display advanced analysis tab"""
    if 'processed_df' in st.session_state:
        processed_df = st.session_state['processed_df']
        
        if 'OK_KO_Label' in processed_df.columns:
            st.info("🎯 This module automatically identifies features that best distinguish between OK and KO cases using statistical tests and machine learning algorithms.")
            
            # Analysis configuration
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("⚙️ Analysis Settings")
                
                # Analysis method selection
                analysis_types = st.multiselect(
                    "Select analysis methods:",
                    options=['Statistical Tests', 'Machine Learning Feature Importance', 'Combined Analysis'],
                    default=['Combined Analysis'],
                    help="Choose which analysis methods to apply"
                )
                
                # Top features to display
                top_n = st.slider("Top N features to display:", min_value=5, max_value=min(20, len(processed_df.columns)-1), value=10)
            
            with col2:
                st.subheader("📊 Data Summary")
                ok_count = len(processed_df[processed_df['OK_KO_Label'] == 'OK'])
                ko_count = len(processed_df[processed_df['OK_KO_Label'] == 'KO'])
                total_features = len(processed_df.columns) - 1  # Exclude label column
                
                st.metric("OK Samples", ok_count)
                st.metric("KO Samples", ko_count)  
                st.metric("Total Features", total_features)
            
            # Start Analysis Button
            if st.button("🚀 Run Advanced Analysis", type="primary"):
                if analysis_types:
                    with st.spinner("Running comprehensive feature analysis... This may take a few moments."):
                        try:
                            # Run statistical analysis
                            analysis_results = st.session_state.analysis_engine.analyze_all_features(processed_df)
                            
                            # Run ML feature importance if needed
                            ml_results = None
                            if 'Machine Learning Feature Importance' in analysis_types or 'Combined Analysis' in analysis_types:
                                try:
                                    st.info("🔄 Training AutoGluon models for feature importance... (this may take a while)")
                                    ml_analyzer = FeatureImportanceAnalyzer(random_state=42)
                                    ml_results = ml_analyzer.analyze_feature_importance(
                                        df=processed_df,
                                        target_col='OK_KO_Label',
                                        time_limit=120,
                                        preset='medium_quality'
                                    )
                                    # Merge ML results into analysis_results
                                    analysis_results['ml_feature_importance'] = ml_results
                                except Exception as ml_e:
                                    st.warning(f"⚠️ ML Feature Importance analysis failed: {str(ml_e)}")
                                    analysis_results['ml_feature_importance'] = None
                            
                            # Ensure feature_ranking is accessible from top level (for Model Training tab)
                            # Priority: ML > Statistical
                            if not analysis_results.get('feature_ranking'):
                                ml_fi = analysis_results.get('ml_feature_importance', {})
                                if ml_fi and ml_fi.get('feature_importance', {}).get('feature_ranking'):
                                    analysis_results['feature_ranking'] = ml_fi['feature_importance']['feature_ranking']
                            
                            st.session_state['analysis_results'] = analysis_results
                            st.success("✅ Advanced analysis completed!")
                        except Exception as e:
                            st.error(f"Analysis failed: {str(e)}")
                            st.exception(e)
                else:
                    st.warning("Please select at least one analysis method")
            
            # Display Results
            if 'analysis_results' in st.session_state:
                results = st.session_state['analysis_results']
                
                st.subheader("📋 Analysis Results")
                
                # Summary metrics
                summary = results.get('summary', {})
                
                if summary:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        significant_count = len([f for f in results.get('feature_ranking', []) if f.get('significant', False)])
                        st.metric("Significant Features", significant_count)
                    with col2:
                        st.metric("Total Features Analyzed", 
                                 summary.get('numerical_features', 0) + summary.get('categorical_features', 0))
                    with col3:
                        st.metric("OK Samples", summary.get('ok_samples', 0))
                    with col4:
                        st.metric("KO Samples", summary.get('ko_samples', 0))
                
                # Feature Rankings Tabs - dynamically create based on selected methods
                tab_names = []
                if 'Combined Analysis' in analysis_types:
                    tab_names.extend(["🏆 Combined Ranking", "📊 Statistical Analysis", "🤖 ML Feature Importance"])
                else:
                    if 'Statistical Tests' in analysis_types:
                        tab_names.append("📊 Statistical Analysis")
                    if 'Machine Learning Feature Importance' in analysis_types:
                        tab_names.append("🤖 ML Feature Importance")
                
                # Create tabs based on selected methods
                if not tab_names:
                    st.warning("Please select at least one analysis method")
                else:
                    tabs = st.tabs(tab_names)
                    
                    # Map tabs to content
                    tab_idx = 0
                    feature_ranking = results.get('feature_ranking', [])[:top_n]
                    
                    # Combined Ranking (only if Combined Analysis selected)
                    if 'Combined Analysis' in analysis_types:
                        with tabs[tab_idx]:
                            st.subheader("🏆 Combined Feature Ranking")
                            st.write("Features ranked by combining statistical significance and composite scores")
                            
                            if feature_ranking:
                                ranking_df = pd.DataFrame(feature_ranking)
                                # Format numeric columns
                                for col in ['p_value', 'effect_size', 'composite_score', 'difference_ratio']:
                                    if col in ranking_df.columns:
                                        ranking_df[col] = ranking_df[col].round(6)
                                st.dataframe(ranking_df, height=400, use_container_width=True)
                                
                                # Visualization
                                fig, ax = plt.subplots(figsize=(10, 6))
                                features = [item['feature'] for item in feature_ranking]
                                scores = [item.get('composite_score', 0) for item in feature_ranking]
                                
                                bars = ax.barh(features[::-1], scores[::-1])
                                ax.set_xlabel('Composite Score')
                                ax.set_title(f'Top {len(features)} Features by Combined Score')
                                
                                # Color bars by score
                                if max(scores) > 0:
                                    for i, bar in enumerate(bars):
                                        bar.set_color(plt.cm.RdYlBu_r(scores[::-1][i] / max(scores)))
                                
                                plt.tight_layout()
                                st.pyplot(fig, use_container_width=True)
                                plt.close(fig)
                            else:
                                st.info("No combined ranking available")
                        tab_idx += 1
                    
                    # Statistical Analysis
                    if 'Statistical Tests' in analysis_types or 'Combined Analysis' in analysis_types:
                        with tabs[tab_idx]:
                            st.subheader("📊 Statistical Analysis Results")
                            
                            if feature_ranking:
                                st.write("Features ranked by statistical significance (p-value and effect size)")
                                
                                stat_df = pd.DataFrame(feature_ranking)
                                # Display relevant columns
                                display_cols = ['feature', 'type', 'p_value', 'effect_size', 'significant']
                                available_cols = [col for col in display_cols if col in stat_df.columns]
                                display_df = stat_df[available_cols].copy()
                                
                                for col in ['p_value', 'effect_size']:
                                    if col in display_df.columns:
                                        display_df[col] = display_df[col].round(6)
                                
                                st.dataframe(display_df, height=400, use_container_width=True)
                                
                                # P-value visualization
                                fig, ax = plt.subplots(figsize=(10, 6))
                                features = [item['feature'] for item in feature_ranking]
                                p_values = [item.get('p_value', 1) for item in feature_ranking]
                                
                                # Use negative log p-values for better visualization
                                neg_log_p = [-np.log10(max(p, 1e-16)) for p in p_values]
                                
                                bars = ax.barh(features[::-1], neg_log_p[::-1])
                                ax.set_xlabel('-log10(p-value)')
                                ax.set_title('Statistical Significance of Features')
                                ax.axvline(-np.log10(0.05), color='red', linestyle='--', alpha=0.7, label='p=0.05 threshold')
                                ax.legend()
                                
                                plt.tight_layout()
                                st.pyplot(fig, use_container_width=True)
                                plt.close(fig)
                            else:
                                st.info("No statistical analysis results available")
                        tab_idx += 1
                    
                    # ML Feature Importance
                    if 'Machine Learning Feature Importance' in analysis_types or 'Combined Analysis' in analysis_types:
                        with tabs[tab_idx]:
                            st.subheader("🤖 AutoGluon ML Feature Importance")
                            
                            ml_results = results.get('ml_feature_importance', {})
                            
                            if ml_results:
                                # Display best model info
                                best_model_info = ml_results.get('best_model', {})
                                if best_model_info:
                                    st.success(f"🏆 Best Model: **{best_model_info.get('name', 'Unknown')}** | Validation Score: **{best_model_info.get('score_val', 0):.4f}**")
                                    
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Validation Accuracy", f"{best_model_info.get('score_val', 0):.4f}")
                                    with col2:
                                        if best_model_info.get('fit_time'):
                                            st.metric("Training Time", f"{best_model_info.get('fit_time', 0):.2f}s")
                                    with col3:
                                        if best_model_info.get('pred_time_val'):
                                            st.metric("Prediction Time", f"{best_model_info.get('pred_time_val', 0):.4f}s")
                                
                                # Model leaderboard
                                leaderboard = ml_results.get('model_leaderboard', [])
                                if leaderboard:
                                    st.write("**AutoGluon Model Leaderboard:**")
                                    leaderboard_df = pd.DataFrame(leaderboard)
                                    
                                    # Select relevant columns for display
                                    display_cols = ['model', 'score_val', 'pred_time_val', 'fit_time', 'stack_level']
                                    available_cols = [col for col in display_cols if col in leaderboard_df.columns]
                                    
                                    if available_cols:
                                        display_df = leaderboard_df[available_cols].copy()
                                        # Round numeric columns
                                        for col in display_df.select_dtypes(include=[np.number]).columns:
                                            display_df[col] = display_df[col].round(4)
                                        st.dataframe(display_df, height=300, use_container_width=True)
                                
                                # Feature importance
                                feature_importance_info = ml_results.get('feature_importance', {})
                                feature_ranking_ml = feature_importance_info.get('feature_ranking', [])[:top_n]
                                
                                if feature_ranking_ml:
                                    st.write(f"**Top {len(feature_ranking_ml)} Most Important Features (AutoGluon):**")
                                    
                                    importance_df = pd.DataFrame(feature_ranking_ml)
                                    if not importance_df.empty:
                                        # Display table
                                        display_cols_ml = ['feature', 'importance', 'rank']
                                        available_cols_ml = [col for col in display_cols_ml if col in importance_df.columns]
                                        if available_cols_ml:
                                            display_df = importance_df[available_cols_ml].copy()
                                            for col in display_df.select_dtypes(include=[np.number]).columns:
                                                display_df[col] = display_df[col].round(6)
                                            st.dataframe(display_df, height=300, use_container_width=True)
                                            
                                            # Feature importance visualization
                                            fig, ax = plt.subplots(figsize=(10, 6))
                                            features = display_df['feature'].tolist()
                                            importances = display_df['importance'].tolist()
                                            
                                            bars = ax.barh(features[::-1], importances[::-1])
                                            ax.set_xlabel('Feature Importance Score')
                                            ax.set_title('AutoGluon Feature Importance (Permutation-based)')
                                            
                                            # Color gradient
                                            if max(importances) > 0:
                                                for i, bar in enumerate(bars):
                                                    bar.set_color(plt.cm.viridis(importances[::-1][i] / max(importances)))
                                            
                                            plt.tight_layout()
                                            st.pyplot(fig, use_container_width=True)
                                            plt.close(fig)
                                else:
                                    st.info("No feature importance data available")
                            else:
                                st.info("🔄 ML Feature importance analysis not yet completed. Select 'Machine Learning Feature Importance' or 'Combined Analysis' and run again.")
        else:
            st.warning("OK/KO labels not found. Please complete preprocessing first.")
    else:
        st.info("Please complete data preprocessing to access advanced analysis features.")
