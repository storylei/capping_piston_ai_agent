"""
Step 1: Load Data
"""

import streamlit as st
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from data_processing import DataLoader


def display():
    """Display Step 1: Load Data"""
    st.subheader("📁 Step 1: Load Data")
    st.markdown("Select and load your dataset")
    
    loader = DataLoader()
    available_datasets = loader.get_available_datasets()
    
    if not available_datasets:
        st.error("❌ No data files found in `data/raw/` directory")
        st.info("Please add CSV or C-MAPSS txt files to `data/raw/`")
        return
    
    # Initialize selected file
    if 'selected_file' not in st.session_state:
        st.session_state.selected_file = available_datasets[0]
    
    # File selection
    selected_file = st.selectbox(
        "Choose Dataset:",
        available_datasets,
        index=available_datasets.index(st.session_state.selected_file),
        help="Select a CSV or C-MAPSS txt file"
    )
    st.session_state.selected_file = selected_file
    
    # Load data
    if st.button("📥 Load Data", type="primary"):
        with st.spinner("Loading data..."):
            try:
                # Load without rul_threshold to preserve RUL column for Step 2
                df = loader.load_file(selected_file)
                st.session_state.current_data = df
                
                st.success(f"✅ Loaded: {selected_file} ({df.shape[0]} rows × {df.shape[1]} cols)")
                
                # Show preview
                # st.subheader("Data Preview")
                # st.dataframe(df.head(), height=200)
                
                st.session_state.config_step = 2  # Proceed to Step 2
                st.rerun()
            except Exception as e:
                st.error(f"❌ Failed to load data: {str(e)}")
