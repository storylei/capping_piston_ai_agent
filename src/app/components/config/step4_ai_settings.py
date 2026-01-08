"""
Step 4: AI Agent Configuration
"""

import streamlit as st
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from agent import StatisticalAgent


def display():
    """Display Step 4: AI Agent Configuration"""
    st.subheader("🤖 Step 4: AI Agent Configuration")
    st.markdown("Configure the AI Agent backend and interpretation settings")
    
    if 'processed_df' not in st.session_state:
        st.warning("⚠️ Please complete preprocessing first")
        if st.button("← Back to Step 1"):
            st.session_state.config_step = 1
            st.rerun()
        return
    
    # Show data status
    st.info(
        f"✅ Data ready: {st.session_state.processed_df.shape[0]} rows × "
        f"{st.session_state.processed_df.shape[1]} columns"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🔌 LLM Backend Selection**")
        llm_backend = st.radio(
            "Choose LLM Backend:",
            options=["ollama", "openai"],
            index=0,
            help="Ollama = Local LLAMA3 (recommended for project). OpenAI = Cloud API."
        )
        
        if llm_backend == "ollama":
            st.success("✅ Using Local LLM (LLAMA3)")
            st.markdown("""
            **Setup Instructions:**
            - Download: https://ollama.ai/download
            - Run: `ollama pull llama3`
            - Start: `ollama serve`
            """)
        else:
            st.warning("⚠️ Using OpenAI API")
            api_key = st.text_input(
                "OpenAI API Key:",
                type="password",
                value=os.getenv("OPENAI_API_KEY", ""),
                help="Enter your OpenAI API key"
            )
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
                st.success("✅ API key saved")
    
    with col2:
        st.markdown("**🧠 LLM Interpretation Settings**")
        enable_interpretation = st.checkbox(
            "Enable LLM Interpretation",
            value=st.session_state.get('enable_llm_interpretation', False),
            help="AI will explain analysis results in natural language (slower but more insightful)"
        )
        
        if enable_interpretation:
            st.info("""
            **When enabled:**
            - Analysis results include AI explanations
            - Responses are more detailed
            - Processing takes longer
            """)
        else:
            st.info("""
            **Fast mode:**
            - Direct tool outputs only
            - Quick responses
            - No AI interpretation
            """)
    
    st.markdown("---")
    
    # Save configuration
    if st.button("💾 Save AI Configuration", type="primary"):
        try:
            # Update session state
            st.session_state.enable_llm_interpretation = enable_interpretation
            st.session_state.llm_backend = llm_backend
            
            # Create/update agent
            st.session_state.agent = StatisticalAgent(
                llm_backend=llm_backend,
                api_key=os.getenv("OPENAI_API_KEY"),
                enable_llm_interpretation=enable_interpretation
            )
            
            st.success(f"✅ AI Agent configured with {llm_backend.upper()} backend")
            st.session_state.config_complete = True
            st.session_state.config_step = 5
            st.balloons()
            st.rerun()
        except Exception as e:
            st.error(f"❌ Failed to configure AI Agent: {str(e)}")
            st.info("💡 Tip: Make sure Ollama is running if you selected it as backend")
    
    # Skip button
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("⏭️ Skip (Use Default)"):
            st.session_state.enable_llm_interpretation = False
            st.session_state.llm_backend = "ollama"
            st.session_state.config_complete = True
            st.session_state.config_step = 5
            st.info("Using default settings: Ollama backend, interpretation disabled")
            st.rerun()
    with col2:
        if st.button("← Back to Preprocessing"):
            st.session_state.config_step = 3
            st.rerun()
