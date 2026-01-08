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
    
    st.subheader("📈 Data Analysis")
    st.markdown("Compare feature distributions between OK and KO groups")
    
    if 'OK_KO_Label' not in processed_df.columns:
        st.warning("⚠️ OK_KO_Label not found in data")
        return
    
    # Split features by type
    numerical_cols = processed_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = processed_df.select_dtypes(include=['object']).columns.tolist()
    
    # Filter meaningful features
    meaningful_numerical = [
        col for col in numerical_cols 
        if col != 'OK_KO_Label' and 
        processed_df[col].nunique() <= len(processed_df) * 0.9 and
        processed_df[col].nunique() > 1
    ]
    
    meaningful_categorical = [
        col for col in categorical_cols 
        if col != 'OK_KO_Label' and 
        processed_df[col].nunique() > 1 and
        processed_df[col].nunique() <= 100
    ]
    
    # Build feature summary table
    feature_summary = []
    for col in processed_df.columns:
        dtype = str(processed_df[col].dtype)
        nunique = processed_df[col].nunique()
        
        # Determine feature type and availability
        if col == 'OK_KO_Label':
            feature_type = "Label"
            available = "🎯"
            reason = "Target variable"
        elif col in meaningful_numerical:
            feature_type = "Numerical"
            available = "✅"
            reason = "Available"
        elif col in meaningful_categorical:
            feature_type = "Categorical"
            available = "✅"
            reason = "Available"
        elif col in numerical_cols:
            feature_type = "Numerical"
            available = "❌"
            if nunique == 1:
                reason = "Only 1 unique value"
            elif nunique > len(processed_df) * 0.9:
                reason = f"Too many unique ({nunique})"
            else:
                reason = "Filtered"
        elif col in categorical_cols:
            feature_type = "Categorical"
            available = "❌"
            if nunique == 1:
                reason = "Only 1 unique value"
            elif nunique > 100:
                reason = f"Too many categories ({nunique})"
            else:
                reason = "Filtered"
        else:
            feature_type = "Other"
            available = "❌"
            reason = "Unknown type"
        
        feature_summary.append({
            'Column': col,
            'Type': feature_type,
            'Data Type': dtype,
            'Unique': nunique,
            'Available': available,
            'Status': reason
        })
    
    summary_df = pd.DataFrame(feature_summary)
        # Count filtered features (exclude OK_KO_Label)
    filtered_count = len(summary_df[(summary_df['Available'] == '❌') & (summary_df['Column'] != 'OK_KO_Label')])
    
    
    # Show summary table
    with st.expander("📊 Dataset Summary & Feature Availability", expanded=True):
        st.write(f"**Total Columns:** {len(processed_df.columns)} | "
             f"**Numerical:** {len(meaningful_numerical)} | "
             f"**Categorical:** {len(meaningful_categorical)} | "
             f"**Filtered:** {filtered_count}")
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    # Create tabs for different feature types
    tab1, tab2 = st.tabs(["📊 Numerical Features", "🏷️ Categorical Features"])
    
    # ===== TAB 1: Numerical Features =====
    with tab1:
        if not meaningful_numerical:
            st.info("No meaningful numerical features available for analysis.")
        else:
            selected_numerical = st.multiselect(
                "Select numerical features to analyze:",
                meaningful_numerical,
                max_selections=5,
                key="numerical_features"
            )
            
            if selected_numerical:
                ok_data = processed_df[processed_df['OK_KO_Label'] == 'OK']
                ko_data = processed_df[processed_df['OK_KO_Label'] == 'KO']
                
                for feature in selected_numerical:
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
    
    # ===== TAB 2: Categorical Features =====
    with tab2:
        if not meaningful_categorical:
            st.info("No meaningful categorical features available for analysis.")
        else:
            selected_categorical = st.multiselect(
                "Select categorical features to analyze:",
                meaningful_categorical,
                max_selections=5,
                key="categorical_features"
            )
            
            if selected_categorical:
                ok_data = processed_df[processed_df['OK_KO_Label'] == 'OK']
                ko_data = processed_df[processed_df['OK_KO_Label'] == 'KO']
                
                for feature in selected_categorical:
                    st.write(f"### {feature} - Category Distribution")
                    
                    # Create crosstab
                    crosstab = pd.crosstab(
                        processed_df[feature], 
                        processed_df['OK_KO_Label'],
                        margins=False
                    )
                    
                    # Calculate percentages
                    crosstab_pct = pd.crosstab(
                        processed_df[feature], 
                        processed_df['OK_KO_Label'],
                        normalize='columns'
                    ) * 100
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.write("**Count Table:**")
                        st.dataframe(crosstab, use_container_width=True)
                        st.write("**Percentage Table:**")
                        st.dataframe(crosstab_pct.round(2).astype(str) + '%', use_container_width=True)
                    
                    with col2:
                        # Bar plot comparison
                        fig, ax = plt.subplots(figsize=(8, 4))
                        x = np.arange(len(crosstab.index))
                        width = 0.35
                        
                        ax.bar(x - width/2, crosstab['OK'], width, label='OK', color='#2ecc71', alpha=0.8)
                        ax.bar(x + width/2, crosstab['KO'], width, label='KO', color='#e74c3c', alpha=0.8)
                        
                        ax.set_xlabel(feature, fontweight='bold')
                        ax.set_ylabel('Count', fontweight='bold')
                        ax.set_title(f'{feature} Distribution by OK/KO')
                        ax.set_xticks(x)
                        ax.set_xticklabels(crosstab.index, rotation=45, ha='right')
                        ax.legend()
                        ax.grid(axis='y', alpha=0.3)
                        
                        plt.tight_layout()
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)
                    
                    st.markdown("---")
