"""
Data Analysis Tab - Feature exploration and comparison
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def display():
    """Display data exploration/analysis tab"""
    if 'processed_df' not in st.session_state:
        st.info("Please complete configuration first.")
        return
    
    processed_df = st.session_state['processed_df']
    
    st.subheader("Data Exploration Analysis")
    st.markdown("Compare feature distributions between OK and KO groups")
    
    # Select features
    numerical_cols = processed_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    meaningful_numerical = [
        col for col in numerical_cols 
        if col != 'OK_KO_Label' and 
        processed_df[col].nunique() <= len(processed_df) * 0.9 and
        processed_df[col].nunique() > 1
    ]
    
    selected_features = st.multiselect(
        "Select numerical features to analyze:",
        meaningful_numerical,
        max_selections=5
    )
    
    if selected_features and 'OK_KO_Label' in processed_df.columns:
        ok_data = processed_df[processed_df['OK_KO_Label'] == 'OK']
        ko_data = processed_df[processed_df['OK_KO_Label'] == 'KO']
        
        for feature in selected_features:
            st.write(f"### {feature} - Statistical Comparison")
            
            comparison_df = pd.DataFrame({
                'OK': [ok_data[feature].mean(), ok_data[feature].std(), ok_data[feature].median()],
                'KO': [ko_data[feature].mean(), ko_data[feature].std(), ko_data[feature].median()]
            }, index=['Mean', 'Std Dev', 'Median'])
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.dataframe(comparison_df.round(4), use_container_width=True)
            
            with col2:
                # Distribution plot
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.hist(ok_data[feature].dropna(), alpha=0.6, label='OK', bins=20, color='#2ecc71')
                ax.hist(ko_data[feature].dropna(), alpha=0.6, label='KO', bins=20, color='#e74c3c')
                ax.set_xlabel(feature, fontweight='bold')
                ax.set_ylabel('Frequency', fontweight='bold')
                ax.set_title(f'{feature} Distribution Comparison')
                ax.legend()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            
            st.markdown("---")
