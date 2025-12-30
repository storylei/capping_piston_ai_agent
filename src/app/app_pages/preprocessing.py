"""
Preprocessing Results Tab - Display preprocessing details and configure preprocessing
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def display():
    """Display preprocessing results and configuration tab"""
    
    # Check if we have data loaded
    if 'current_data' not in st.session_state or st.session_state['current_data'].empty:
        st.warning("⚠️ Please load data first in Configuration tab")
        return
    
    df = st.session_state['current_data']
    
    # Two modes: Configuration or Results
    if 'processed_df' not in st.session_state:
        # Configuration mode - allow user to configure and run preprocessing
        st.subheader("🔧 Data Preprocessing Configuration")
        st.markdown("Configure preprocessing settings and apply transformations")
        
        # Check if label configuration is done
        if 'label_col' not in st.session_state:
            st.warning("⚠️ Please configure OK/KO labels first in Configuration tab")
            return
        
        # Show current config
        st.info(
            f"📊 **Data**: {df.shape[0]} rows × {df.shape[1]} columns\n"
            f"🏷️ **Label Column**: {st.session_state['label_col']}\n"
            f"✅ **OK Values**: {st.session_state['ok_values']}\n"
            f"❌ **KO Values**: {st.session_state['ko_values']}"
        )
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 💧 Missing Value Handling")
            
            # Show missing value information
            missing_info = df.isnull().sum()
            cols_with_missing = missing_info[missing_info > 0]
            
            if len(cols_with_missing) > 0:
                st.write("**Columns with missing values:**")
                for col, count in cols_with_missing.items():
                    ratio = count / len(df) * 100
                    st.write(f"- `{col}`: {count} ({ratio:.1f}%)")
                
                use_auto_strategy = st.checkbox(
                    "Use automatic handling strategy",
                    value=True,
                    help="Automatically fill numerical columns with mean, categorical with mode"
                )
            else:
                st.success("✅ No missing values detected")
                use_auto_strategy = True
        
        with col2:
            st.markdown("### 🏷️ Categorical Encoding")
            
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            # Exclude the label column
            categorical_cols = [c for c in categorical_cols if c != st.session_state.get('label_col')]
            
            if categorical_cols:
                st.write(f"**Found {len(categorical_cols)} categorical columns:**")
                for col in categorical_cols[:5]:  # Show first 5
                    st.write(f"- `{col}`")
                if len(categorical_cols) > 5:
                    st.write(f"... and {len(categorical_cols) - 5} more")
                
                encoding_method = st.radio(
                    "Encoding method:",
                    options=['label', 'onehot'],
                    index=0,
                    help="**label**: Convert to numeric codes\n**onehot**: Create binary columns"
                )
            else:
                st.success("✅ No categorical variables to encode")
                encoding_method = 'label'
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Numerical Feature Scaling")
            scale_features = st.checkbox(
                "Scale numerical features",
                value=False,
                help="Standardize numerical features to have mean=0, std=1"
            )
            
            if scale_features:
                scale_method = st.radio(
                    "Scaling method:",
                    options=['standard', 'minmax'],
                    help="**standard**: Mean=0, Std=1\n**minmax**: Scale to [0, 1]"
                )
            else:
                scale_method = 'standard'
        
        with col2:
            st.markdown("### ⚠️ Important Notes")
            st.warning(
                "🛡️ **Data Leakage Prevention**\n\n"
                "The original label column will be removed after creating OK_KO_Label "
                "to prevent data leakage during analysis and model training."
            )
        
        st.markdown("---")
        
        # Start preprocessing button
        if st.button("🚀 Start Preprocessing", type="primary", use_container_width=True):
            with st.spinner("🔄 Processing data..."):
                try:
                    preprocessor = st.session_state.data_preprocessor
                    
                    # Step 1: Create OK/KO labels (automatically drops original label column)
                    processed_df = preprocessor.create_ok_ko_labels(
                        df=df,
                        label_col=st.session_state['label_col'],
                        ok_values=st.session_state['ok_values'],
                        drop_original=True  # Prevent data leakage
                    )
                    
                    # Step 2: Handle missing values
                    if len(cols_with_missing) > 0:
                        processed_df = preprocessor.handle_missing_values(
                            df=processed_df,
                            strategy=None  # Use default strategy
                        )
                    
                    # Step 3: Encode categorical variables
                    if categorical_cols:
                        # Pass None to auto-encode all categorical columns
                        processed_df = preprocessor.encode_categorical_variables(
                            df=processed_df,
                            encoding_methods=None  # Auto-encode
                        )
                    
                    # Step 4: Scale numerical features
                    if scale_features:
                        processed_df = preprocessor.scale_numerical_features(
                            df=processed_df,
                            method=scale_method
                        )
                    
                    # Save processed data
                    filename = f"processed_{st.session_state['label_col']}_data.csv"
                    saved_path = preprocessor.save_processed_data(processed_df, filename)
                    
                    # Save to session state
                    st.session_state['processed_df'] = processed_df
                    st.session_state['preprocessing_summary'] = preprocessor.get_preprocessing_summary(df, processed_df)
                    
                    st.success("✅ Preprocessing completed successfully!")
                    st.info(f"💾 Saved to: `{saved_path}`")
                    st.balloons()
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Preprocessing failed: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc(), language="python")
    
    else:
        # Results mode - show preprocessing results
        st.subheader("✅ Preprocessed Data")
        
        processed_df = st.session_state['processed_df']
    
    st.subheader("Preprocessed Data")
    st.dataframe(processed_df, height=300, use_container_width=True)
    
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
            st.info(f"✨ New columns added: {', '.join(summary['new_columns'])}")
        if summary['removed_columns']:
            st.warning(f"🗑️ Columns removed: {', '.join(summary['removed_columns'])}")
    
    # OK/KO Distribution
    if 'OK_KO_Label' in processed_df.columns:
        st.subheader("OK/KO Distribution")
        distribution = processed_df['OK_KO_Label'].value_counts()
        
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.pie(distribution.values, labels=distribution.index, autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'])
            ax.set_title("OK/KO Distribution")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        
        with col2:
            st.bar_chart(distribution)
