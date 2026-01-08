"""
Configuration Page - Wizard-style setup for data processing
"""

import streamlit as st
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from components.config import (
    display_load_data,
    display_configure_labels,
    display_preprocessing,
    display_ai_settings,
    display_complete
)


def _render_progress_indicator():
    """Render wizard progress indicator"""
    progress_col1, progress_col2, progress_col3, progress_col4, progress_col5 = st.columns(5)
    steps = [
        ("📁 Load Data", 1),
        ("🏷️ OK/KO Labels", 2),
        ("🔧 Preprocess", 3),
        ("🤖 AI Settings", 4),
        ("✅ Complete", 5)
    ]
    
    for i, (step_name, step_num) in enumerate(steps):
        col = [progress_col1, progress_col2, progress_col3, progress_col4, progress_col5][i]
        with col:
            if step_num < st.session_state.config_step:
                st.markdown(f"""
                <div style="background-color: #d4edda; border: 1px solid #c3e6cb; border-radius: 5px; 
                           padding: 12px; text-align: center; font-weight: 500; color: #155724;
                           display: flex; align-items: center; justify-content: center; min-height: 50px;">
                    {step_name}
                </div>
                """, unsafe_allow_html=True)
            elif step_num == st.session_state.config_step:
                st.markdown(f"""
                <div style="background-color: #d1ecf1; border: 1px solid #bee5eb; border-radius: 5px; 
                           padding: 12px; text-align: center; font-weight: 600; color: #0c5460;
                           display: flex; align-items: center; justify-content: center; min-height: 50px;">
                    {step_name}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background-color: #e9ecef; border: 1px solid #dee2e6; border-radius: 5px; 
                           padding: 12px; text-align: center; font-weight: 500; color: #6c757d;
                           display: flex; align-items: center; justify-content: center; min-height: 50px;">
                    {step_name}
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("---")


def display():
    """Display configuration wizard"""
    st.title("⚙️ Configuration Wizard")
    st.markdown("Complete the setup steps to prepare your data for analysis")
    
    # Initialize config state if needed
    if 'config_step' not in st.session_state:
        st.session_state.config_step = 1
    if 'config_complete' not in st.session_state:
        st.session_state.config_complete = False
    
    # Render progress indicator
    _render_progress_indicator()
    
    # Route to appropriate step
    if st.session_state.config_step == 1:
        display_load_data()
    elif st.session_state.config_step == 2:
        display_configure_labels()
    elif st.session_state.config_step == 3:
        display_preprocessing()
    elif st.session_state.config_step == 4:
        display_ai_settings()
    elif st.session_state.config_step == 5:
        display_complete()
    else:
        st.error(f"Unknown step: {st.session_state.config_step}")
