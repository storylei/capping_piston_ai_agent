import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data_processing import DataLoader, DataPreprocessor
from analysis import AnalysisEngine
from agent import StatisticalAgent

# Initialize app components
st.set_page_config(page_title="Statistical AI Agent", layout="wide")

# Initialize processors
if 'data_loader' not in st.session_state:
    st.session_state.data_loader = DataLoader()
if 'data_preprocessor' not in st.session_state:
    st.session_state.data_preprocessor = DataPreprocessor()
if 'analysis_engine' not in st.session_state:
    st.session_state.analysis_engine = AnalysisEngine()

# Initialize AI Agent
if 'agent' not in st.session_state:
    # Default to Ollama for local deployment (as per project requirements)
    backend = os.getenv("LLM_BACKEND", "ollama")
    api_key = os.getenv("OPENAI_API_KEY", None)
    st.session_state.agent = StatisticalAgent(llm_backend=backend, api_key=api_key)

# Initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def load_data_with_loader(filename):
    """Load data using DataLoader"""
    try:
        loader = DataLoader()
        df = loader.load_csv(filename)
        st.success(f"✅ Successfully loaded: {filename} ({df.shape[0]} rows × {df.shape[1]} cols)")
        return df
    except Exception as e:
        st.error(f"Failed to load data: {str(e)}")
        return pd.DataFrame()

def load_default_data():
    """Load default dataset or let user choose"""
    loader = DataLoader()
    available_datasets = loader.get_available_datasets()
    
    if not available_datasets:
        st.error("No CSV files found in data/raw/ directory")
        return pd.DataFrame()
    
    # Initialize session state for selected file
    if 'selected_file' not in st.session_state:
        st.session_state.selected_file = available_datasets[0]
    if 'current_data' not in st.session_state:
        st.session_state.current_data = pd.DataFrame()
    
    # Let user choose dataset in sidebar
    with st.sidebar:
        st.header("📁 Data Selection")
        selected_file = st.selectbox(
            "Choose Dataset:", 
            available_datasets,
            help="Select a CSV file from data/raw/ directory",
            key="dataset_selector"
        )
        
        # Load data when file changes or button clicked
        if (selected_file != st.session_state.selected_file) or st.button("🔄 Load Data", key="load_data_btn"):
            st.session_state.selected_file = selected_file
            st.session_state.current_data = load_data_with_loader(selected_file)
            # Clear preprocessing state when new data is loaded
            for key in ['label_col', 'ok_values', 'ko_values', 'processed_df', 'preprocessing_summary']:
                if key in st.session_state:
                    del st.session_state[key]
    
    # Return current data
    if st.session_state.current_data.empty:
        st.session_state.current_data = load_data_with_loader(st.session_state.selected_file)
    
    return st.session_state.current_data

def display_sidebar(df):
    """Configure all sidebar controls"""
    with st.sidebar:
        st.header("⚙️ Control Panel")
        
        # Dataset information
        if not df.empty:
            st.info(f"📊 Current Data: {df.shape[0]} rows × {df.shape[1]} columns")
            
            # Data validation
            is_valid, msg = st.session_state.data_loader.validate_data_for_analysis(df)
            if is_valid:
                st.success(f"✅ {msg}")
            else:
                st.error(f"❌ {msg}")
                return
        
        # OK/KO Label Configuration
        with st.expander("🏷️ OK/KO Label Configuration", expanded=True):
            # Suggested label columns
            suggested_cols = st.session_state.data_loader.suggest_label_columns(df)
            if suggested_cols:
                st.info(f"💡 Suggested label columns: {', '.join(suggested_cols)}")
            
            label_col = st.selectbox(
                "Select Label Column:", 
                options=df.columns,
                help="Choose the column containing classification labels"
            )
            
            if label_col:
                unique_vals = df[label_col].dropna().unique().tolist()
                st.write(f"Unique values in '{label_col}': {unique_vals}")
                
                # OK values selection (multi-select support)
                ok_values = st.multiselect(
                    "Select values as 'OK':",
                    options=unique_vals,
                    help="You can select multiple values as OK category"
                )
                
                if ok_values:
                    ko_values = [v for v in unique_vals if v not in ok_values]
                    st.write(f"**OK Category**: {ok_values}")
                    st.write(f"**KO Category**: {ko_values}")
                    
                    if st.button("✅ Confirm OK/KO Configuration", key="confirm_okko"):
                        st.session_state['label_col'] = label_col
                        st.session_state['ok_values'] = ok_values
                        st.session_state['ko_values'] = ko_values
                        st.success("Configuration saved!")
        
        # Data Preprocessing Configuration
        if 'label_col' in st.session_state:
            with st.expander("🔧 Data Preprocessing", expanded=False):
                st.subheader("Missing Value Handling")
                
                # Show missing value information
                missing_info = df.isnull().sum()
                cols_with_missing = missing_info[missing_info > 0]
                
                if len(cols_with_missing) > 0:
                    st.write("Columns with missing values:")
                    for col, count in cols_with_missing.items():
                        ratio = count / len(df) * 100
                        st.write(f"- {col}: {count} ({ratio:.1f}%)")
                        
                    # Processing strategy selection
                    use_auto_strategy = st.checkbox("Use automatic handling strategy", value=True)
                    
                    if not use_auto_strategy:
                        st.write("Manual configuration:")
                        # Manual configuration options can be added here
                        st.info("Manual configuration feature under development...")
                else:
                    st.success("✅ No missing values")
                
                st.subheader("Categorical Variable Encoding")
                categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
                if categorical_cols:
                    encoding_method = st.selectbox(
                        "Encoding method:",
                        options=['label', 'onehot'],
                        help="label: Label encoding, onehot: One-hot encoding"
                    )
                else:
                    st.info("No categorical variables to encode")
                
                st.subheader("Numerical Feature Scaling")
                scale_features = st.checkbox("Scale numerical features", value=False)
                if scale_features:
                    scale_method = st.selectbox(
                        "Scaling method:",
                        options=['standard', 'minmax'],
                        help="standard: Standardization, minmax: Min-max scaling"
                    )
                
                                
                # Important note about data leakage
                st.info("ℹ️ Note: The original label column will be removed after creating OK_KO_Label to prevent data leakage in analysis.")
                
                # Start preprocessing
                if st.button("🚀 Start Preprocessing", key="start_preprocessing"):
                    with st.spinner("Processing data..."):
                        try:
                            # Create OK/KO labels (will automatically drop original label column)
                            processed_df = st.session_state.data_preprocessor.create_ok_ko_labels(
                                df, st.session_state['label_col'], st.session_state['ok_values'],
                                drop_original=True  # Explicitly drop original label to prevent data leakage
                            )
                            
                            # Handle missing values
                            if len(cols_with_missing) > 0:
                                processed_df = st.session_state.data_preprocessor.handle_missing_values(processed_df)
                            
                            # Encode categorical variables
                            if categorical_cols:
                                processed_df = st.session_state.data_preprocessor.encode_categorical_variables(processed_df)
                            
                            # Scale numerical features
                            if scale_features:
                                processed_df = st.session_state.data_preprocessor.scale_numerical_features(
                                    processed_df, method=scale_method
                                )
                            
                            # Save processed data
                            filename = f"processed_{st.session_state['label_col']}_data.csv"
                            saved_path = st.session_state.data_preprocessor.save_processed_data(processed_df, filename)
                            
                            # Save to session state
                            st.session_state['processed_df'] = processed_df
                            st.session_state['preprocessing_summary'] = st.session_state.data_preprocessor.get_preprocessing_summary(df, processed_df)
                            
                            st.success(f"✅ Data preprocessing completed! Saved to: {saved_path}")
                            st.info(f"ℹ️  Original label column '{st.session_state['label_col']}' has been removed to prevent data leakage.")
                            
                        except Exception as e:
                            st.error(f"Preprocessing failed: {str(e)}")

def display_data_section(df):
    """Upper main area with tab-based data exploration"""
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Raw Data", "🔍 Preprocessing Results", "📈 Data Analysis", "🔬 Advanced Analysis", "🤖 AI Agent Chat"])

    with tab1:
        st.subheader("Raw Data Preview")
        st.dataframe(df, height=300)
        st.caption(f"Showing {len(df)} rows × {len(df.columns)} columns")
        
        # Basic statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Data Types")
            dtype_info = pd.DataFrame({
                'Column': df.columns,
                'Data Type': df.dtypes.astype(str),
                'Non-Null': df.count(),
                'Unique Values': [df[col].nunique() for col in df.columns]
            })
            st.dataframe(dtype_info, height=200)
        
        with col2:
            st.subheader("Missing Values Summary")
            missing_df = pd.DataFrame({
                'Missing Count': df.isnull().sum(),
                'Missing Ratio (%)': (df.isnull().mean() * 100).round(2)
            })
            missing_df = missing_df[missing_df['Missing Count'] > 0]
            if not missing_df.empty:
                st.dataframe(missing_df, height=200)
            else:
                st.success("✅ No missing values found")

    with tab2:
        if 'processed_df' in st.session_state:
            st.subheader("Preprocessed Data")
            processed_df = st.session_state['processed_df']
            st.dataframe(processed_df, height=300)
            
            # Show preprocessing summary
            if 'preprocessing_summary' in st.session_state:
                summary = st.session_state['preprocessing_summary']
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Original Shape", f"{summary['original_shape'][0]} × {summary['original_shape'][1]}")
                with col2:
                    st.metric("Processed Shape", f"{summary['processed_shape'][0]} × {summary['processed_shape'][1]}")
                with col3:
                    st.metric("Missing Values", f"{summary['missing_values_before']} → {summary['missing_values_after']}")
                
                if summary['new_columns']:
                    st.info(f"New columns added: {', '.join(summary['new_columns'])}")
                if summary['removed_columns']:
                    st.warning(f"Columns removed: {', '.join(summary['removed_columns'])}")
            
            # OK/KO Distribution
            if 'OK_KO_Label' in processed_df.columns:
                st.subheader("OK/KO Distribution")
                distribution = processed_df['OK_KO_Label'].value_counts()
                
                col1, col2 = st.columns(2)
                with col1:
                    fig, ax = plt.subplots(figsize=(4, 4))
                    ax.pie(distribution.values, labels=distribution.index, autopct='%1.1f%%')
                    ax.set_title("OK/KO Distribution")
                    st.pyplot(fig)
                    plt.close(fig)  # Fix: Close figure to prevent memory leak
                
                with col2:
                    st.bar_chart(distribution)
                    
        else:
            st.info("Please complete data preprocessing first")

    with tab3:
        st.subheader("Data Exploration Analysis")
        if 'processed_df' in st.session_state:
            processed_df = st.session_state['processed_df']
            
            # Select features to analyze - exclude non-meaningful numerical columns
            numerical_cols = processed_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            # Define columns that should be excluded (IDs, categorical variables, etc.)
            exclude_patterns = ['ticket', 'cabin', 'embarked', 'sex', 'name', 'passengerid']
            
            # Filter out columns that are IDs, encoded categoricals, or categorical variables
            meaningful_numerical = []
            for col in numerical_cols:
                if col == 'OK_KO_Label':
                    continue
                # Skip columns matching exclude patterns
                if any(pattern in col.lower() for pattern in exclude_patterns):
                    continue
                # Skip columns with very high cardinality (likely IDs)
                if processed_df[col].nunique() > len(processed_df) * 0.9:
                    continue
                # Skip columns with very few unique values (likely encoded categories)
                # But keep Pclass-like features (1,2,3) which are actually ordinal
                if processed_df[col].nunique() <= 3 and col.lower() not in ['pclass', 'sibsp', 'parch']:
                    continue
                meaningful_numerical.append(col)
            
            selected_features = st.multiselect(
                "Select numerical features to analyze:", 
                meaningful_numerical,
                help="Only continuous numerical features are shown. Categorical variables and IDs are excluded."
            )
            
            if selected_features:
                if 'OK_KO_Label' in processed_df.columns:
                    # Analyze by OK/KO groups
                    st.subheader("OK vs KO Feature Comparison")
                    
                    ok_data = processed_df[processed_df['OK_KO_Label'] == 'OK']
                    ko_data = processed_df[processed_df['OK_KO_Label'] == 'KO']
                    
                    for feature in selected_features:
                        st.write(f"**{feature}** Statistical Comparison:")
                        
                        comparison_df = pd.DataFrame({
                            'OK': [
                                ok_data[feature].mean(),
                                ok_data[feature].std(),
                                ok_data[feature].median()
                            ],
                            'KO': [
                                ko_data[feature].mean(),
                                ko_data[feature].std(),
                                ko_data[feature].median()
                            ]
                        }, index=['Mean', 'Std Dev', 'Median'])
                        
                        st.dataframe(comparison_df.round(4))
                        
                        # Distribution plot
                        fig, ax = plt.subplots(figsize=(8, 4))
                        ax.hist(ok_data[feature].dropna(), alpha=0.5, label='OK', bins=20)
                        ax.hist(ko_data[feature].dropna(), alpha=0.5, label='KO', bins=20)
                        ax.set_xlabel(feature)
                        ax.set_ylabel('Frequency')
                        ax.set_title(f'{feature} Distribution Comparison')
                        ax.legend()
                        st.pyplot(fig)
                        plt.close(fig)  # Fix: Close figure to prevent memory leak
                else:
                    # Basic statistical analysis
                    st.dataframe(processed_df[selected_features].describe())
        else:
            st.info("Please complete data preprocessing first")

    with tab4:
        st.subheader("🔬 Advanced Feature Analysis")
        
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
                                # Run analysis
                                analysis_results = st.session_state.analysis_engine.analyze_all(processed_df)
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
                            st.metric("Significant Features", 
                                     summary['analysis_summary'].get('features_with_statistical_significance', 0))
                        with col2:
                            st.metric("ML Models Tested", 
                                     summary['analysis_summary'].get('ml_models_evaluated', 0))
                        with col3:
                            st.metric("Best CV Accuracy", 
                                     f"{summary['analysis_summary'].get('cross_validated_accuracy', 0):.3f}")
                        with col4:
                            st.metric("Consensus Features", 
                                     len(summary.get('consensus_features', [])))
                    
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
                        
                        # Combined Ranking (only if Combined Analysis selected)
                        if 'Combined Analysis' in analysis_types:
                            with tabs[tab_idx]:
                                st.subheader("🏆 Combined Feature Ranking")
                                st.write("Features ranked by combining statistical significance and ML importance scores")
                                
                                combined_ranking = summary.get('top_ml_features', [])[:top_n]
                                if combined_ranking:
                                    ranking_df = pd.DataFrame(combined_ranking)
                                    # Handle both 'importance' (from AutoGluon) and 'combined_score' fields
                                    score_field = 'importance' if 'importance' in ranking_df.columns else 'combined_score'
                                    if score_field in ranking_df.columns:
                                        ranking_df[score_field] = ranking_df[score_field].round(4)
                                    st.dataframe(ranking_df, height=400)
                                    
                                    # Visualization
                                    if combined_ranking:
                                        fig, ax = plt.subplots(figsize=(10, 6))
                                        features = [item['feature'] for item in combined_ranking]
                                        # Get scores from the appropriate field
                                        scores = [item.get('importance', item.get('combined_score', 0)) for item in combined_ranking]
                                        
                                        bars = ax.barh(features[::-1], scores[::-1])
                                        ax.set_xlabel('Feature Importance Score')
                                        ax.set_title(f'Top {len(features)} Features by Combined Ranking')
                                        
                                        # Color bars by score
                                        if max(scores) > 0:
                                            for i, bar in enumerate(bars):
                                                bar.set_color(plt.cm.RdYlBu_r(scores[::-1][i] / max(scores)))
                                        
                                        plt.tight_layout()
                                        st.pyplot(fig)
                                        plt.close(fig)
                                else:
                                    st.info("No combined ranking available")
                            tab_idx += 1
                        
                        # Statistical Analysis
                        if 'Statistical Tests' in analysis_types or 'Combined Analysis' in analysis_types:
                            with tabs[tab_idx]:
                                st.subheader("📊 Statistical Analysis Results")
                                
                                statistical_results = results.get('statistical_analysis', {})
                                stat_ranking = summary.get('top_statistical_features', [])[:top_n]
                                
                                if stat_ranking:
                                    st.write("Features ranked by statistical significance (p-value and effect size)")
                                    
                                    stat_df = pd.DataFrame(stat_ranking)
                                    if 'p_value' in stat_df.columns:
                                        stat_df['p_value'] = stat_df['p_value'].round(6)
                                    if 'effect_size' in stat_df.columns:
                                        stat_df['effect_size'] = stat_df['effect_size'].round(4)
                                    st.dataframe(stat_df, height=400)
                                    
                                    # P-value visualization
                                    fig, ax = plt.subplots(figsize=(10, 6))
                                    features = [item['feature'] for item in stat_ranking]
                                    p_values = [item.get('p_value', 1) for item in stat_ranking]
                                    
                                    # Use negative log p-values for better visualization
                                    neg_log_p = [-np.log10(max(p, 1e-16)) for p in p_values]
                                    
                                    bars = ax.barh(features[::-1], neg_log_p[::-1])
                                    ax.set_xlabel('-log10(p-value)')
                                    ax.set_title('Statistical Significance of Features')
                                    ax.axvline(-np.log10(0.05), color='red', linestyle='--', alpha=0.7, label='p=0.05 threshold')
                                    ax.legend()
                                    
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                    plt.close(fig)
                                else:
                                    st.info("No statistical analysis results available")
                            tab_idx += 1
                        
                        # ML Feature Importance
                        if 'Machine Learning Feature Importance' in analysis_types or 'Combined Analysis' in analysis_types:
                            with tabs[tab_idx]:
                                st.subheader("🤖 AutoGluon ML Feature Importance")
                                
                                ml_results = results.get('feature_importance', {})
                                
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
                                        st.dataframe(display_df, height=300)
                                
                                # Feature importance
                                feature_importance_info = ml_results.get('feature_importance', {})
                                if feature_importance_info:
                                    feature_ranking = feature_importance_info.get('feature_ranking', [])[:top_n]
                                    
                                    if feature_ranking:
                                        st.write(f"**Top {len(feature_ranking)} Most Important Features (AutoGluon):**")
                                        
                                        importance_df = pd.DataFrame(feature_ranking)
                                        if not importance_df.empty:
                                            # Display table
                                            display_df = importance_df[['feature', 'importance', 'rank']].copy()
                                            display_df['importance'] = display_df['importance'].round(4)
                                            st.dataframe(display_df, height=300)
                                            
                                            # Feature importance visualization
                                            fig, ax = plt.subplots(figsize=(10, 6))
                                            features = display_df['feature'].tolist()
                                            importances = display_df['importance'].tolist()
                                            
                                            bars = ax.barh(features[::-1], importances[::-1])
                                            ax.set_xlabel('Feature Importance Score')
                                            ax.set_title('AutoGluon Feature Importance (Permutation-based)')
                                            
                                            # Color gradient
                                            for i, bar in enumerate(bars):
                                                bar.set_color(plt.cm.viridis(importances[::-1][i] / max(importances)))
                                            
                                            plt.tight_layout()
                                            st.pyplot(fig)
                                            plt.close(fig)
                                    else:
                                        st.info("No feature importance data available")
                                else:
                                    st.info("Feature importance analysis not yet completed")
                    
                    # Consensus Features
                    if summary.get('consensus_features'):
                        st.subheader("🎯 Consensus Features")
                        st.write("Features that appear in both statistical and ML top rankings:")
                        
                        consensus_features = summary['consensus_features']
                        for i, feature in enumerate(consensus_features, 1):
                            st.write(f"{i}. **{feature}**")
                        
                        if len(consensus_features) > 0:
                            st.success(f"Found {len(consensus_features)} features with strong consensus across methods!")
                        else:
                            st.info("No strong consensus features found. Consider reviewing analysis parameters.")
            else:
                st.warning("OK/KO labels not found. Please complete preprocessing first.")
        else:
            st.info("Please complete data preprocessing to access advanced analysis features.")
    
    with tab5:
        st.subheader("🤖 AI Agent - Natural Language Data Analysis")
        st.markdown("Ask questions about your data or request plots in natural language!")
        
        if 'processed_df' not in st.session_state or st.session_state['processed_df'] is None:
            st.warning("⚠️ Please complete data preprocessing first to use the AI Agent.")
            st.info("The AI Agent needs preprocessed data with OK/KO labels to perform analysis.")
        else:
            processed_df = st.session_state['processed_df']
            
            # AI Agent Configuration in Sidebar
            with st.sidebar:
                st.header("🤖 AI Agent Settings")
                
                st.info("🎯 **Local Deployment Mode**\n\nThis project uses local LLAMA3 via Ollama (as per project requirements)")
                
                # LLM Backend Selection
                llm_backend = st.selectbox(
                    "LLM Backend:",
                    options=["ollama", "openai"],
                    index=0,  # Default to ollama
                    help="Ollama = Local LLAMA3 (recommended). OpenAI = Cloud API (backup option)."
                )
                
                if llm_backend == "ollama":
                    st.success("✅ Using Local LLM (LLAMA3)")
                    st.markdown("""
                    **Setup Instructions:**
                    1. Download: https://ollama.ai/download
                    2. Install Ollama
                    3. Run: `ollama pull llama3`
                    4. Ollama service should start automatically
                    """)
                else:
                    st.warning("⚠️ Using OpenAI API (requires API key)")
                    api_key_input = st.text_input(
                        "OpenAI API Key:",
                        type="password",
                        value=os.getenv("OPENAI_API_KEY", ""),
                        help="Backup option - Enter API key"
                    )
                    if api_key_input:
                        os.environ["OPENAI_API_KEY"] = api_key_input
                
                # Update agent backend if changed
                if st.button("🔄 Update Agent Backend"):
                    try:
                        st.session_state.agent = StatisticalAgent(
                            llm_backend=llm_backend,
                            api_key=os.getenv("OPENAI_API_KEY")
                        )
                        st.success(f"✅ Agent backend updated to {llm_backend}")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                
                if st.button("🗑️ Clear Chat History"):
                    st.session_state.chat_history = []
                    st.session_state.agent.clear_conversation()
                    st.success("Chat history cleared!")
            
            # Update Agent's data context
            if st.session_state.agent.current_data is None or \
               st.session_state.agent.current_data.shape != processed_df.shape:
                st.session_state.agent.set_data_context(processed_df, {
                    'filename': st.session_state.get('selected_file', 'Unknown'),
                    'label_col': st.session_state.get('label_col', 'Unknown')
                })
                
                # Set analysis results if available
                if 'analysis_results' in st.session_state:
                    st.session_state.agent.set_analysis_results(st.session_state['analysis_results'])
            
            # Example queries
            st.markdown("### 💡 Example Queries:")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **Statistical Analysis:**
                - "Show me the statistical summary for all features"
                - "What's the mean and std for Age in OK vs KO?"
                - "Get feature importance ranking"
                """)
            
            with col2:
                st.markdown("""
                **Visualization:**
                - "Show the distribution of Age"
                - "Compare Fare between OK and KO groups"
                - "Plot boxplot for Pclass"
                """)
            
            st.markdown("---")
            
            # Chat Input (put BEFORE chat history for better UX)
            user_input = st.chat_input("Ask a question about your data or request a plot...")
            
            # Process user input first
            if user_input:
                # Add user message to history
                st.session_state.chat_history.append({
                    'role': 'user',
                    'content': user_input
                })
                
                # Get AI response
                with st.spinner("🤔 Thinking..."):
                    try:
                        response = st.session_state.agent.chat(user_input)
                        
                        # Add assistant response to history
                        st.session_state.chat_history.append({
                            'role': 'assistant',
                            'content': response['response'],
                            'plots': response.get('plots', [])
                        })
                        
                    except Exception as e:
                        error_msg = f"❌ Error: {str(e)}"
                        st.session_state.chat_history.append({
                            'role': 'assistant',
                            'content': error_msg
                        })
            
            # Display Chat History (newest at bottom, near input box)
            chat_container = st.container()
            with chat_container:
                # Show chat history from newest to oldest (reversed)
                for msg in reversed(st.session_state.chat_history):
                    with st.chat_message(msg['role']):
                        st.markdown(msg['content'])
                        
                        # Display plots if any
                        if msg.get('plots'):
                            for plot_fig in msg['plots']:
                                st.pyplot(plot_fig)

def main():
    st.title("🤖 Statistical AI Agent - Data Analysis Platform")
    st.markdown("Automated OK/KO data analysis, feature identification and model training")
    
    # Load data
    df = load_default_data()
    
    if not df.empty:
        # Build interface
        display_sidebar(df)
        display_data_section(df) 
    else:
        st.warning("Please place your dataset files in data/raw/ folder")
        st.info("Supported file formats: CSV")

if __name__ == "__main__":
    main()