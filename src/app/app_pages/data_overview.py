"""
Data Overview Tab - Display raw and preprocessed data with comparison
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def display():
    """Display data overview with raw and preprocessed tabs"""
    
    st.subheader("📊 Data Overview")
    st.markdown("View and compare raw data with preprocessed data")
    
    # Create tabs
    tab1, tab2 = st.tabs(["📋 Raw Data", "✅ Preprocessed Data"])
    
    # ===== TAB 1: Raw Data =====
    with tab1:
        if 'current_data' not in st.session_state or st.session_state.current_data is None:
            st.info("No data loaded. Please complete configuration first.")
            return
        
        df = st.session_state.current_data
        
        st.subheader("Raw Dataset Preview")
        st.dataframe(df, height=300, use_container_width=True)
        st.caption(f"Showing {len(df)} rows × {len(df.columns)} columns")
        
        st.markdown("---")
        
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
    
    # ===== TAB 2: Preprocessed Data =====
    with tab2:
        # Check if preprocessing has been done
        if 'processed_df' not in st.session_state:
            st.warning("⚠️ No preprocessed data available")
            st.info(
                "📋 **Please complete the following steps first:**\n"
                "1. Go to Configuration tab → Step 1: Load Data\n"
                "2. Go to Configuration tab → Step 2: Configure Labels\n"
                "3. Go to Configuration tab → Step 3: Preprocessing Data\n"
                "4. Click '🚀 Start Preprocessing'\n\n"
                "Once preprocessing is complete, you can view the results here."
            )
            return
        
        processed_df = st.session_state['processed_df']
        
        # Display the preprocessed dataframe
        st.subheader("Preprocessed Dataset")
        st.dataframe(processed_df, height=300, use_container_width=True)
        
        st.markdown("---")
        
        # Show preprocessing summary
        if 'preprocessing_summary' in st.session_state:
            summary = st.session_state['preprocessing_summary']
            
            st.subheader("📈 Preprocessing Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Original Shape", f"{summary['original_shape'][0]} × {summary['original_shape'][1]}")
            with col2:
                st.metric("Processed Shape", f"{summary['processed_shape'][0]} × {summary['processed_shape'][1]}")
            with col3:
                st.metric("Missing Values", f"{summary['missing_values_before']} → {summary['missing_values_after']}")
            
            st.markdown("---")
            
            # Show new and removed columns
            col1, col2 = st.columns(2)
            with col1:
                if summary['new_columns']:
                    st.success(f"✨ **New columns added:** {len(summary['new_columns'])}")
                    st.write(", ".join(summary['new_columns']))
                else:
                    st.info("ℹ️ No new columns added")
            
            with col2:
                if summary['removed_columns']:
                    st.warning(f"🗑️ **Columns removed:** {len(summary['removed_columns'])}")
                    st.write(", ".join(summary['removed_columns']))
                else:
                    st.info("ℹ️ No columns removed")
        
        st.markdown("---")
        
        # OK/KO Distribution
        if 'OK_KO_Label' in processed_df.columns:
            st.subheader("🎯 OK/KO Distribution")
            distribution = processed_df['OK_KO_Label'].value_counts()
            
            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.pie(distribution.values, labels=distribution.index, autopct='%1.1f%%', 
                      colors=['#2ecc71', '#e74c3c'], startangle=90)
                ax.set_title("OK/KO Distribution")
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            
            with col2:
                st.bar_chart(distribution)
                # Show counts
                st.metric("OK Samples", distribution.get('OK', 0))
                st.metric("KO Samples", distribution.get('KO', 0))
