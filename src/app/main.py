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
from analysis.feature_importance import FeatureImportanceAnalyzer
from agent import StatisticalAgent
from components.sidebar import display_sidebar
from app_pages import configuration, data_overview, data_analysis, advanced_analysis, model_training, ai_agent



def initialize_session_state():
    """Initialize all session state variables"""
    if 'data_preprocessor' not in st.session_state:
        st.session_state.data_preprocessor = DataPreprocessor()
    
    if 'analysis_engine' not in st.session_state:
        st.session_state.analysis_engine = StatisticalAnalyzer()
    
    # Initialize ML feature importance analyzer separately
    if 'ml_analyzer' not in st.session_state:
        st.session_state.ml_analyzer = FeatureImportanceAnalyzer(random_state=42)
    
    if 'agent' not in st.session_state:
        # Get configuration from session state or environment variables
        llm_backend = st.session_state.get('llm_backend', os.getenv('LLM_BACKEND', 'ollama'))
        llm_model = st.session_state.get('llm_model', None)
        
        # Get API key based on backend
        api_key = st.session_state.get('llm_api_key', None)
        if not api_key:
            # Try environment variables
            env_keys = {
                'openai': 'OPENAI_API_KEY',
                'claude': 'ANTHROPIC_API_KEY',
                'gemini': 'GOOGLE_API_KEY',
                'deepseek': 'DEEPSEEK_API_KEY'
            }
            if llm_backend in env_keys:
                api_key = os.getenv(env_keys[llm_backend])
        
        st.session_state.agent = StatisticalAgent(
            llm_backend=llm_backend,
            llm_model=llm_model,
            api_key=api_key,
            enable_llm_interpretation=st.session_state.get('enable_llm_interpretation', False)
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
    
    # Initialize LLM configuration state
    if 'llm_backend' not in st.session_state:
        st.session_state.llm_backend = 'ollama'
    
    if 'llm_model' not in st.session_state:
        st.session_state.llm_model = None
    
    if 'llm_api_key' not in st.session_state:
        st.session_state.llm_api_key = None


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
    elif selected_tab == 'data_overview':
        data_overview.display()
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