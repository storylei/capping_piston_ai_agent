"""
Step 5: Configuration Complete
"""

import streamlit as st


def display():
    """Display Step 5: Configuration Complete"""
    st.subheader("✅ Step 5: Configuration Complete")
    
    if 'processed_df' not in st.session_state:
        st.warning("⚠️ Configuration incomplete")
        return
    
    processed_df = st.session_state.processed_df
    summary = st.session_state.get('preprocessing_summary', {})
    
    # Success message
    st.success("🎉 Configuration complete! Data is ready for analysis.")
    
    # Summary statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Rows", processed_df.shape[0])
    with col2:
        st.metric("Total Features", processed_df.shape[1])
    with col3:
        if 'OK_KO_Label' in processed_df.columns:
            ok_count = (processed_df['OK_KO_Label'] == 'OK').sum()
            ko_count = (processed_df['OK_KO_Label'] == 'KO').sum()
            st.metric("OK / KO", f"{ok_count} / {ko_count}")
    
    # Next steps
    st.markdown("---")
    st.subheader("📋 Next Steps")
    st.markdown("""
    ✅ Configuration complete! You can now:
    
    1. **📊 Data Overview** → View raw data and preprocessing results
    2. **📈 Data Analysis** → Explore features and distributions
    3. **🔬 Advanced Analysis** → Run AutoGluon feature importance
    4. **🎯 Model Training** → Train discriminative models
    5. **🤖 AI Agent Chat** → Ask questions in natural language
    """)
    
    st.markdown("---")
    st.subheader("🔄 Edit Configuration")
    st.markdown("Need to change something? You can return to any step:")
    
    # Navigation buttons to each step
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📁 Step 1: Load Data", use_container_width=True):
            st.session_state.config_step = 1
            st.rerun()
    
    with col2:
        if st.button("🏷️ Step 2: Labels", use_container_width=True):
            st.session_state.config_step = 2
            st.rerun()
    
    with col3:
        if st.button("🔧 Step 3: Preprocess", use_container_width=True):
            st.session_state.config_step = 3
            st.rerun()
    
    with col4:
        if st.button("🤖 Step 4: AI Settings", use_container_width=True):
            st.session_state.config_step = 4
            st.rerun()
