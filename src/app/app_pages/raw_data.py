"""
Raw Data Tab - Display raw dataset overview and statistics
"""

import streamlit as st
import pandas as pd
import numpy as np


def display():
    """Display raw data tab"""
    if 'current_data' not in st.session_state or st.session_state.current_data is None:
        st.info("No data loaded. Please complete configuration first.")
        return
    
    df = st.session_state.current_data
    
    st.subheader("Raw Data Preview")
    st.dataframe(df, height=300, use_container_width=True)
    st.caption(f"Showing {len(df)} rows × {len(df.columns)} columns")
    
    # Basic statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Data Types")
        dtype_info = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes.astype(str),
            'Non-Null': df.count(),
            'Unique Values': [df[col].nunique() for col in df.columns]
        })
        st.dataframe(dtype_info, height=300, use_container_width=True)
    
    with col2:
        st.subheader("⚠️ Missing Values Summary")
        missing_df = pd.DataFrame({
            'Missing Count': df.isnull().sum(),
            'Missing Ratio (%)': (df.isnull().mean() * 100).round(2)
        })
        missing_df = missing_df[missing_df['Missing Count'] > 0]
        if not missing_df.empty:
            st.dataframe(missing_df, height=300, use_container_width=True)
        else:
            st.success("✅ No missing values found")
