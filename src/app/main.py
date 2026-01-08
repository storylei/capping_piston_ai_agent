"""
Main Application Entry Point - Route between pages
"""

import streamlit as st
import os
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data_processing import DataLoader, DataPreprocessor
from analysis import StatisticalAnalyzer
from agent import StatisticalAgent
from components.sidebar import display_sidebar
from app_pages import configuration, raw_data, preprocessing, data_analysis, advanced_analysis, model_training, ai_agent



def initialize_session_state():
    """Initialize all session state variables"""
    if 'data_preprocessor' not in st.session_state:
        st.session_state.data_preprocessor = DataPreprocessor()
    
    if 'analysis_engine' not in st.session_state:
        st.session_state.analysis_engine = StatisticalAnalyzer()
    
    if 'agent' not in st.session_state:
        llm_backend = os.getenv('LLM_BACKEND', 'ollama')
        st.session_state.agent = StatisticalAgent(
            llm_backend=llm_backend,
            api_key=os.getenv('OPENAI_API_KEY'),
            enable_llm_interpretation=False
        )
    
    if 'enable_llm_interpretation' not in st.session_state:
        st.session_state.enable_llm_interpretation = False
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'config_complete' not in st.session_state:
        st.session_state.config_complete = False
    
    if 'config_step' not in st.session_state:
        st.session_state.config_step = 1
    
    if 'nav_tab' not in st.session_state:
        st.session_state.nav_tab = 'configuration'


def main():
    """Main application entry point"""
    # Page configuration
    st.set_page_config(
        page_title="Capping Piston AI Agent",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .stMain { padding-top: 0; }
    [data-testid="stSidebar"] { background-color: #f8f9fa; }
    [data-testid="stMetricValue"] { font-size: 24px; }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize
    initialize_session_state()
    
    # Sidebar navigation
    selected_tab = display_sidebar()
    
    # Route to selected tab
    if selected_tab == 'configuration':
        configuration.display()
    elif selected_tab == 'raw_data':
        raw_data.display()
    elif selected_tab == 'preprocessing':
        preprocessing.display()
    elif selected_tab == 'data_analysis':
        data_analysis.display()
    elif selected_tab == 'advanced_analysis':
        advanced_analysis.display()
    elif selected_tab == 'model_training':
        model_training.display()
    elif selected_tab == 'ai_agent':
        ai_agent.display()
    else:
        st.error(f"Unknown tab: {selected_tab}")


if __name__ == "__main__":
    main()