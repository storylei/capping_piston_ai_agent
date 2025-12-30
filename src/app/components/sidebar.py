"""
Sidebar Navigation Component - Simple tab switcher
"""

import streamlit as st


def display_sidebar():
    """
    Display sidebar with tab navigation
    Returns selected tab name
    """
    with st.sidebar:
        st.title("📑 Navigation")
        
        tabs = [
            ("⚙️ Configuration", "configuration"),
            ("📋 Raw Data", "raw_data"),
            ("🔧 Preprocessing", "preprocessing"),
            ("📊 Data Analysis", "data_analysis"),
            ("🔬 Advanced Analysis", "advanced_analysis"),
            ("🎯 Model Training", "model_training"),
            ("🤖 AI Agent Chat", "ai_agent"),
        ]
        
        # Create buttons for each tab
        selected_tab = None
        for tab_label, tab_id in tabs:
            if st.button(tab_label, use_container_width=True, key=f"tab_btn_{tab_id}"):
                selected_tab = tab_id
                st.session_state.nav_tab = tab_id
        
        # Use session state if no button just clicked
        if selected_tab is None:
            selected_tab = st.session_state.get('nav_tab', 'configuration')
        
        return selected_tab
