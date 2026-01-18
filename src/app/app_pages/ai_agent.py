"""
AI Agent Chat Tab - Natural language data analysis
"""

import streamlit as st
import os
import matplotlib.pyplot as plt


def display():
    """Display AI Agent chat tab"""
    if 'processed_df' not in st.session_state:
        st.warning("⚠️ Please complete configuration first.")
        st.info("The AI Agent needs preprocessed data with OK/KO labels to perform analysis.")
        return
    
    processed_df = st.session_state['processed_df']
    
    st.subheader("🤖 AI Agent - Natural Language Data Analysis")
    st.markdown("Ask questions about your data or request plots in natural language!")
    
    # AI Settings status and controls at the top
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        # Show current AI settings
        if 'llm_backend' in st.session_state:
            backend = st.session_state.get('llm_backend', 'ollama')
            st.info(f"🔌 **Backend**: {backend.upper()}")
    
    with col2:
        if 'enable_llm_interpretation' in st.session_state:
            interpretation = st.session_state.get('enable_llm_interpretation', False)
            st.info(f"🧠 **Interpretation**: {'Enabled' if interpretation else 'Disabled'}")
    
    with col3:
        if st.button("🗑️ Clear Chat", width="stretch"):
            st.session_state.chat_history = []
            if hasattr(st.session_state.agent, 'clear_conversation'):
                st.session_state.agent.conversation.clear_history()
            st.success("Chat cleared!")
            st.rerun()
    
    st.caption("💡 To change AI settings, go to **Configuration** tab (Step 4: AI Settings)")
    st.markdown("---")
    
    # Update agent context
    data_changed = False
    if st.session_state.agent.current_data is None or \
       st.session_state.agent.current_data.shape != processed_df.shape:
        # Data changed - clear chat history to avoid confusion
        st.session_state.chat_history = []
        st.session_state.agent.conversation.clear_history()
        st.session_state.agent.set_data_context(processed_df)
        data_changed = True
    
    if data_changed:
        st.info("🔄 New dataset detected. Chat history has been cleared.")
        # Force rerun to refresh the interface
        st.rerun()
    
    if 'analysis_results' in st.session_state:
        st.session_state.agent.set_analysis_results(st.session_state['analysis_results'])  
    else:
        st.warning("⚠️ No analysis results in session state. Please run Advanced Analysis first.")
    
    # Example queries - dynamic based on actual columns
    st.markdown("### 💡 Example Queries:")
    
    # Get actual column names, prefer the second column when the first is time
    cols = list(processed_df.columns)
    start_idx = 0
    # To avoid 'time' and 'OK_KO_Label' being selected
    # Todo maybe will other columns should be skipped?
    if len(cols) >= 2 and cols[0].lower() == 'time' and cols[1] != 'OK_KO_Label':
        start_idx = 1
    available_cols = [col for col in cols[start_idx:] if col != 'OK_KO_Label']
    example_col = available_cols[0] if available_cols else 'feature_name'
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        **Statistical Analysis:**
        - "Show statistical summary for all features"
        - "What's the mean {example_col} difference between OK and KO?"
        - "Get feature importance ranking"
        """)
    
    with col2:
        st.markdown(f"""
        **Visualization:**
        - "Plot histogram of {example_col}"
        - "Show time series for {example_col}"
        - "Plot FFT for {example_col}"
        - "Compare distribution between OK and KO"
        """)
    
    st.markdown("---")
    
    # Chat input
    user_input = st.chat_input("Ask a question about your data...")
    
    if user_input:
        st.session_state.chat_history.append({'role': 'user', 'content': user_input})
        
        with st.spinner("🤔 Thinking..."):
            try:
                response = st.session_state.agent.chat(user_input)
                
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': response['response'],
                    'plots': response.get('plots', [])
                })
            except Exception as e:
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': f"❌ Error: {str(e)}"
                })
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for msg in reversed(st.session_state.chat_history):
            with st.chat_message(msg['role']):
                st.markdown(msg['content'])
                # Display plots if any
                plots = msg.get('plots', [])
                if plots:
                    for plot_fig in plots:
                        if plot_fig is not None:
                            try:
                                st.pyplot(plot_fig, use_container_width=False)
                            except Exception as e:
                                st.error(f"Failed to display plot: {e}")
