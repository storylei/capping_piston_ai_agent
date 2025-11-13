import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data_processing import DataLoader, DataPreprocessor

# Initialize app components
st.set_page_config(page_title="Statistical AI Agent", layout="wide")

# Initialize processors
if 'data_loader' not in st.session_state:
    st.session_state.data_loader = DataLoader()
if 'data_preprocessor' not in st.session_state:
    st.session_state.data_preprocessor = DataPreprocessor()

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
                
                                
                # Start preprocessing
                if st.button("🚀 Start Preprocessing", key="start_preprocessing"):
                    with st.spinner("Processing data..."):
                        try:
                            # Create OK/KO labels
                            processed_df = st.session_state.data_preprocessor.create_ok_ko_labels(
                                df, st.session_state['label_col'], st.session_state['ok_values']
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
                            
                        except Exception as e:
                            st.error(f"Preprocessing failed: {str(e)}")

def display_data_section(df):
    """Upper main area with tab-based data exploration"""
    tab1, tab2, tab3 = st.tabs(["📊 Raw Data", "🔍 Preprocessing Results", "📈 Data Analysis"])

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
                
                with col2:
                    st.bar_chart(distribution)
                    
        else:
            st.info("Please complete data preprocessing first")

    with tab3:
        st.subheader("Data Exploration Analysis")
        if 'processed_df' in st.session_state:
            processed_df = st.session_state['processed_df']
            
            # Select features to analyze
            numerical_cols = processed_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            selected_features = st.multiselect("Select numerical features to analyze:", numerical_cols)
            
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
                else:
                    # Basic statistical analysis
                    st.dataframe(processed_df[selected_features].describe())
        else:
            st.info("Please complete data preprocessing first")

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