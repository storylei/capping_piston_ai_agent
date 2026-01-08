"""
Step 2: Configure OK/KO Labels
"""

import streamlit as st
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from data_processing import DataLoader


def display():
    """Display Step 2: Configure OK/KO Labels"""
    st.subheader("🏷️ Step 2: Configure OK/KO Labels")
    st.markdown("Define which values represent OK and KO states")
    
    if 'current_data' not in st.session_state:
        st.warning("⚠️ Please complete Step 1 first")
        if st.button("← Back to Step 1"):
            st.session_state.config_step = 1
            st.rerun()
        return
    
    df = st.session_state.current_data
    
    # Classification method selection (without key to allow modification in button)
    st.markdown("**🔄 Select Classification Method**")
    classification_method = st.radio(
        "How to classify OK/KO:",
        options=['by_values', 'by_threshold'],
        format_func=lambda x: "📋 By Values" if x == 'by_values' else "📊 By Threshold"
    )
    
    st.markdown("---")
    
    # ========== METHOD 1: BY VALUES ==========
    if classification_method == 'by_values':
        st.markdown("**Select column values that represent OK state**")
        
        loader = DataLoader()
        suggested_cols = loader.suggest_label_columns(df)
        
        if suggested_cols:
            st.info(f"💡 Suggested label columns: {', '.join(suggested_cols)}")
        
        # Select label column
        label_col = st.selectbox(
            "Select Label Column:",
            options=df.columns,
            help="Column containing OK/KO classification",
            key="label_col_values"
        )
        
        if label_col:
            unique_vals = df[label_col].dropna().unique().tolist()
            st.write(f"**Unique values in '{label_col}**: {unique_vals}")
            
            # Multi-select for OK values
            ok_values = st.multiselect(
                "Select values as 'OK':",
                options=unique_vals,
                help="Can select multiple values as OK category",
                key="ok_values_select"
            )
            
            if ok_values:
                ko_values = [v for v in unique_vals if v not in ok_values]
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**✅ OK values**: {ok_values}")
                with col2:
                    st.write(f"**❌ KO values**: {ko_values}")
                
                # Confirm button
                if st.button("✅ Confirm Configuration", type="primary", key="confirm_values"):
                    # Convert to Python native types to avoid serialization issues
                    ok_values_native = [str(v) if not isinstance(v, (str, int, float, bool)) else v for v in ok_values]
                    ko_values_native = [str(v) if not isinstance(v, (str, int, float, bool)) else v for v in ko_values]
                    
                    st.session_state.label_col = label_col
                    st.session_state.ok_values = ok_values_native
                    st.session_state.ko_values = ko_values_native
                    st.session_state.config_step = 3
                    st.success("Configuration saved! Proceeding to Step 3...")
                    st.rerun()
    
    # ========== METHOD 2: BY THRESHOLD ==========
    else:  # by_threshold
        st.markdown("**Select a numerical column and set threshold**")
        st.info("📌 Values above threshold = OK, values below/equal = KO")
        
        # Get numerical columns (including int, float, and numeric types)
        numerical_cols = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32', np.integer, np.floating]).columns.tolist()
        
        if not numerical_cols:
            st.error("❌ No numerical columns found. Please use 'By Values' method.")
        else:
            threshold_col = st.selectbox(
                "Select Numerical Column:",
                options=numerical_cols,
                help="Column to use for threshold-based classification",
                key="threshold_col"
            )
            
            if threshold_col:
                col_min = df[threshold_col].min()
                col_max = df[threshold_col].max()
                col_mean = df[threshold_col].mean()
                
                st.caption(f"📊 Column stats: min={col_min:.2f}, mean={col_mean:.2f}, max={col_max:.2f}")
                
                # Set threshold
                threshold_value = st.slider(
                    f"Set threshold for '{threshold_col}':",
                    min_value=float(col_min),
                    max_value=float(col_max),
                    value=float(col_mean),
                    key="threshold_value"
                )
                
                # Show preview
                ok_count = (df[threshold_col] > threshold_value).sum()
                ko_count = (df[threshold_col] <= threshold_value).sum()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("✅ OK Samples", ok_count)
                with col2:
                    st.metric("❌ KO Samples", ko_count)
                with col3:
                    st.metric("Total", len(df))
                
                st.write(f"**Classification rule**: {threshold_col} > {threshold_value:.2f} = OK")
                
                # Confirm button
                if st.button("✅ Confirm Configuration", type="primary", key="confirm_threshold"):
                    # Calculate RUL if threshold_col is 'time_cycles' (special case for C-MAPSS)
                    df_for_step3 = df.copy()
                    actual_threshold_col = threshold_col
                    
                    if threshold_col == 'time_cycles' and 'unit_id' in df.columns:
                        # Use loader's _compute_rul method
                        loader = DataLoader()
                        df_for_step3 = loader._compute_rul(df_for_step3)
                        actual_threshold_col = 'RUL'
                    
                    # Update current_data with calculated columns (e.g., RUL if computed)
                    st.session_state.current_data = df_for_step3
                    st.session_state.label_col = actual_threshold_col
                    st.session_state.confirmed_threshold_value = threshold_value
                    st.session_state.config_step = 3
                    st.success("Configuration saved! Proceeding to Step 3...")
                    st.rerun()
    
    # Back button
    if st.button("← Back to Step 1", key="back_step1_from_step2"):
        st.session_state.config_step = 1
        st.rerun()
