"""
Step 3: Preprocessing Settings
"""

import streamlit as st
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from data_processing import DataPreprocessor


def display():
    """Display Step 3: Preprocessing Settings"""
    st.subheader("🔧 Step 3: Preprocessing Data")
    st.markdown("Configure and apply data preprocessing")
    
    if 'current_data' not in st.session_state or 'label_col' not in st.session_state:
        st.warning("⚠️ Please complete Steps 1-2 first")
        if st.button("← Back to Step 1"):
            st.session_state.config_step = 1
            st.rerun()
        return
    
    df = st.session_state.current_data
    
    # Show current config - conditional based on classification method
    if 'confirmed_threshold_value' in st.session_state:
        # Threshold-based method
        threshold_col = st.session_state.label_col
        threshold_value = st.session_state.confirmed_threshold_value
        
        # Calculate preview
        ok_count = (df[threshold_col] > threshold_value).sum()
        ko_count = (df[threshold_col] <= threshold_value).sum()
        
        st.info(
            f"**Data**: {df.shape[0]} rows × {df.shape[1]} cols\n"
            f"**Classification Method**: By Threshold\n"
            f"**Label Column**: {st.session_state.label_col}\n"
            f"**Threshold Value**: {st.session_state.confirmed_threshold_value:.2f}"
        )
        
        # Show OK/KO counts
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("✅ OK Samples", ok_count)
        with col2:
            st.metric("❌ KO Samples", ko_count)
        with col3:
            st.metric("Total", len(df))
            
    else:
        # Values-based method
        st.info(
            f"**Data**: {df.shape[0]} rows × {df.shape[1]} cols\n"
            f"**Classification Method**: By Values\n"
            f"**Label Column**: {st.session_state.label_col}\n"
            f"**OK Values**: {st.session_state.get('ok_values', [])}\n"
            f"**KO Values**: {st.session_state.get('ko_values', [])}"
        )
    
    # Preprocessing options (3 columns)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Missing Value Handling**")
        missing_strategy_label = st.radio(
            "How to handle missing values:",
            [
                "No processing",
                "Auto (per column)",
                "Fill with mean",
                "Fill with median",
                "Fill with mode",
                "Drop rows",
                "Forward fill",
            ],
            key="missing_strategy",
        )

        strategy_map = {
            "No processing": "none",
            "Auto (per column)": "auto",
            "Fill with mean": "mean",
            "Fill with median": "median",
            "Fill with mode": "mode",
            "Drop rows": "drop",
            "Forward fill": "forward_fill",
        }
        selected_missing_strategy = strategy_map[missing_strategy_label]
        st.caption("Auto (per column): Numeric columns use mean; categorical columns use mode.")
    
    with col2:
        st.markdown("**Categorical Encoding**")
        encoding_method_label = st.radio(
            "Encoding method:",
            [
                "No processing",
                "One-hot",
                "Label encoding",
            ],
            key="encoding_method",
        )

        encoding_map = {
            "No processing": "none",
            "One-hot": "onehot",
            "Label encoding": "label",
        }
        selected_encoding_method = encoding_map[encoding_method_label]

    with col3:
        st.markdown("**Feature Scaling**")
        scaling_method_label = st.radio(
            "Scaling method:",
            [
                "No scaling",
                "Standard",
                "Min-Max",
            ],
            key="scaling_method",
        )
        scaling_map = {
            "No scaling": "none",
            "Standard": "standard",
            "Min-Max": "minmax",
        }
        selected_scaling_method = scaling_map[scaling_method_label]
    
    # Preprocess button
    if st.button("🚀 Start Preprocessing", type="primary"):
        with st.spinner("Preprocessing data..."):
            try:
                # Prefer shared preprocessor from session state for consistency
                preprocessor = st.session_state.get('data_preprocessor')
                if preprocessor is None:
                    preprocessor = DataPreprocessor()
                    st.session_state.data_preprocessor = preprocessor
                
                # Handle two classification methods
                # Method 1: threshold-based (if confirmed_threshold_value is in session_state)
                if 'confirmed_threshold_value' in st.session_state:
                    # Create labels based on threshold
                    threshold_col = st.session_state.label_col
                    threshold_value = st.session_state.confirmed_threshold_value
                    processed_df = df.copy()
                    processed_df['OK_KO_Label'] = (processed_df[threshold_col] > threshold_value).astype(str)
                    processed_df['OK_KO_Label'] = processed_df['OK_KO_Label'].map({'True': 'OK', 'False': 'KO'})
                    processed_df = processed_df.drop(columns=[threshold_col])
                else:
                    # Method 2: Create labels based on selected values
                    processed_df = preprocessor.create_ok_ko_labels(
                        df=df,
                        label_col=st.session_state.label_col,
                        ok_values=st.session_state.ok_values,
                        drop_original=True
                    )
                
                # Step 2: Handle missing values - map selection to per-column strategy
                if selected_missing_strategy == 'auto':
                    # Use default per-column strategy from preprocessor
                    processed_df = preprocessor.handle_missing_values(
                        df=processed_df,
                        strategy=None,
                    )
                elif selected_missing_strategy != 'none':
                    strategy_dict = {}
                    if selected_missing_strategy == 'drop':
                        strategy_dict = {
                            col: 'drop'
                            for col in processed_df.columns
                            if processed_df[col].isnull().any()
                        }
                    elif selected_missing_strategy == 'median':
                        strategy_dict = {
                            col: 'median'
                            for col in processed_df.columns
                            if processed_df[col].dtype in ['int64', 'float64']
                            and processed_df[col].isnull().any()
                        }
                    elif selected_missing_strategy == 'mean':
                        strategy_dict = {
                            col: 'mean'
                            for col in processed_df.columns
                            if processed_df[col].dtype in ['int64', 'float64']
                            and processed_df[col].isnull().any()
                        }
                    elif selected_missing_strategy == 'mode':
                        strategy_dict = {
                            col: 'mode'
                            for col in processed_df.columns
                            if processed_df[col].isnull().any()
                        }
                    elif selected_missing_strategy == 'forward_fill':
                        strategy_dict = {
                            col: 'forward_fill'
                            for col in processed_df.columns
                            if processed_df[col].isnull().any()
                        }

                    if strategy_dict:
                        processed_df = preprocessor.handle_missing_values(
                            df=processed_df,
                            strategy=strategy_dict,
                        )
                
                # Step 3: Encode categorical variables
                if selected_encoding_method != 'none':
                    categorical_cols = processed_df.select_dtypes(include=['object']).columns
                    encoding_methods = {
                        col: selected_encoding_method
                        for col in categorical_cols
                        if col != 'OK_KO_Label'
                    }
                    processed_df = preprocessor.encode_categorical_variables(
                        df=processed_df,
                        encoding_methods=encoding_methods,
                    )
                
                # Step 4: Scale numerical features
                if selected_scaling_method != 'none':
                    processed_df = preprocessor.scale_numerical_features(
                        df=processed_df,
                        method=selected_scaling_method,
                    )
                
                st.session_state.processed_df = processed_df
                st.session_state.preprocessing_summary = preprocessor.get_preprocessing_summary(df, processed_df)
                st.session_state.config_step = 4
                
                st.success("✅ Preprocessing completed! Proceeding to AI Agent configuration...")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Preprocessing failed: {str(e)}")
    
    # Back button
    if st.button("← Back"):
        st.session_state.config_step = 2
        st.rerun()
