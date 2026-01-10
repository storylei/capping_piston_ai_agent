"""
Step 4: AI Agent Configuration
"""

import streamlit as st
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from agent import StatisticalAgent


# Backend configuration
LLM_BACKENDS = {
    "ollama": {"name": "Ollama (Local)", "requires_api_key": False, "default_model": "llama3:latest"},
    "openai": {"name": "OpenAI", "requires_api_key": True, "default_model": "gpt-4"},
    "claude": {"name": "Claude (Anthropic)", "requires_api_key": True, "default_model": "claude-3-sonnet-20240229"},
    "gemini": {"name": "Gemini (Google)", "requires_api_key": True, "default_model": "gemini-pro"},
    "deepseek": {"name": "DeepSeek", "requires_api_key": True, "default_model": "deepseek-chat"}
}


def display():
    """Display Step 4: AI Agent Configuration"""
    st.subheader("🤖 Step 4: AI Agent Configuration")
    
    if 'processed_df' not in st.session_state:
        st.warning("⚠️ Please complete preprocessing first")
        if st.button("← Back to Step 1"):
            st.session_state.config_step = 1
            st.rerun()
        return
    
    st.info(f"✅ Data ready: {st.session_state.processed_df.shape[0]} rows × {st.session_state.processed_df.shape[1]} columns")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🔌 LLM Backend**")
        
        backend_options = list(LLM_BACKENDS.keys())
        backend_names = [LLM_BACKENDS[b]["name"] for b in backend_options]
        
        selected_idx = st.selectbox(
            "Choose Backend:",
            options=range(len(backend_options)),
            format_func=lambda x: backend_names[x],
            index=0
        )
        
        llm_backend = backend_options[selected_idx]
        backend_config = LLM_BACKENDS[llm_backend]
        
        # API key input
        api_key = None
        if backend_config["requires_api_key"]:
            env_key_name = {
                "openai": "OPENAI_API_KEY",
                "claude": "ANTHROPIC_API_KEY", 
                "gemini": "GOOGLE_API_KEY",
                "deepseek": "DEEPSEEK_API_KEY"
            }.get(llm_backend, "")
            
            api_key = st.text_input(
                "API Key:",
                type="password",
                value=st.session_state.get(f'{llm_backend}_api_key', os.getenv(env_key_name, ""))
            )
            if api_key:
                st.session_state[f'{llm_backend}_api_key'] = api_key
    
    with col2:
        st.markdown("**🧠 Interpretation**")
        enable_interpretation = st.checkbox(
            "Enable LLM Interpretation",
            value=st.session_state.get('enable_llm_interpretation', False),
            help="AI explains results (slower)"
        )
        
        if enable_interpretation:
            st.info("AI explains results in natural language")
        else:
            st.info("Fast mode: Direct tool outputs only")
    
    st.markdown("---")
    
    # Validation
    can_save = not backend_config["requires_api_key"] or api_key
    
    # Save button
    if st.button("💾 Save Configuration", type="primary", disabled=not can_save):
        try:
            st.session_state.enable_llm_interpretation = enable_interpretation
            st.session_state.llm_backend = llm_backend
            st.session_state.llm_model = backend_config['default_model']
            st.session_state.llm_api_key = api_key
            
            st.session_state.agent = StatisticalAgent(
                llm_backend=llm_backend,
                llm_model=backend_config['default_model'],
                api_key=api_key,
                enable_llm_interpretation=enable_interpretation
            )
            
            st.success(f"✅ Configured: {backend_config['name']}")
            st.session_state.config_complete = True
            st.session_state.config_step = 5
            st.balloons()
            st.rerun()
        except Exception as e:
            st.error(f"❌ Failed: {str(e)}")
    
    if not can_save:
        st.warning("⚠️ API key required")
    
    # Navigation
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("⏭️ Skip (Use Ollama)"):
            st.session_state.enable_llm_interpretation = False
            st.session_state.llm_backend = "ollama"
            st.session_state.llm_model = "llama3:latest"
            st.session_state.llm_api_key = None
            st.session_state.config_complete = True
            st.session_state.config_step = 5
            st.rerun()
    with col2:
        if st.button("← Back"):
            st.session_state.config_step = 3
            st.rerun()