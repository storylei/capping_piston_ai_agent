"""
Configuration Page - Wizard-style setup for data processing
Step 1: Load Data
Step 2: Configure OK/KO Labels
Step 3: Preprocessing Settings
Step 4: Ready to proceed
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from data_processing import DataLoader, DataPreprocessor


def display():
    """Display configuration wizard"""
    st.title("⚙️ Configuration Wizard")
    st.markdown("Complete the setup steps to prepare your data for analysis")
    
    # Initialize config state if needed
    if 'config_step' not in st.session_state:
        st.session_state.config_step = 1
    if 'config_complete' not in st.session_state:
        st.session_state.config_complete = False
    
    # Progress indicator
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
                st.success(step_name)
            elif step_num == st.session_state.config_step:
                st.info(step_name)
            else:
                st.write(f"⏳ {step_name}")
    
    st.markdown("---")
    
    # ========== STEP 1: Load Data ==========
    if st.session_state.config_step == 1:
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
        
        # C-MAPSS specific settings
        is_cmapss = selected_file.startswith('train_FD') and selected_file.endswith('.txt')
        
        rul_threshold = 30
        if is_cmapss:
            st.markdown("**🛩️ C-MAPSS Dataset Settings**")
            rul_threshold = st.slider(
                "RUL Threshold (cycles):",
                min_value=10,
                max_value=100,
                value=30,
                help="Samples with RUL ≤ threshold labeled as KO"
            )
            st.caption(f"• **OK**: RUL > {rul_threshold} cycles (healthy)")
            st.caption(f"• **KO**: RUL ≤ {rul_threshold} cycles (degrading)")
        
        # Load data
        if st.button("📥 Load Data", type="primary"):
            with st.spinner("Loading data..."):
                try:
                    df = loader.load_file(selected_file, rul_threshold=rul_threshold)
                    st.session_state.current_data = df
                    st.session_state.rul_threshold = rul_threshold
                    st.session_state.is_cmapss = is_cmapss
                    
                    st.success(f"✅ Loaded: {selected_file} ({df.shape[0]} rows × {df.shape[1]} cols)")
                    
                    # Show preview
                    st.subheader("Data Preview")
                    st.dataframe(df.head(), height=200)
                    
                    # For C-MAPSS, auto-configure
                    if is_cmapss and 'OK_KO_Label' in df.columns:
                        st.session_state.processed_df = df.copy()
                        st.session_state.label_col = 'OK_KO_Label'
                        st.session_state.config_step = 3  # Skip to Step 3 (Preprocess)
                        st.info("✅ C-MAPSS data auto-configured with OK/KO labels. Proceed to Step 3.")
                    else:
                        st.session_state.config_step = 2  # Go to Step 2
                    
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Failed to load data: {str(e)}")
    
    # ========== STEP 2: Configure OK/KO Labels ==========
    elif st.session_state.config_step == 2:
        st.subheader("🏷️ Step 2: Configure OK/KO Labels")
        st.markdown("Define which values represent OK (healthy) and KO (degraded) states")
        
        if 'current_data' not in st.session_state:
            st.warning("⚠️ Please complete Step 1 first")
            if st.button("← Back to Step 1"):
                st.session_state.config_step = 1
                st.rerun()
            return
        
        df = st.session_state.current_data
        
        # Suggest label columns
        loader = DataLoader()
        suggested_cols = loader.suggest_label_columns(df)
        
        if suggested_cols:
            st.info(f"💡 Suggested label columns: {', '.join(suggested_cols)}")
        
        # Select label column
        label_col = st.selectbox(
            "Select Label Column:",
            options=df.columns,
            help="Column containing OK/KO classification"
        )
        
        if label_col:
            unique_vals = df[label_col].dropna().unique().tolist()
            st.write(f"**Unique values**: {unique_vals}")
            
            # Multi-select for OK values
            ok_values = st.multiselect(
                "Select values as 'OK':",
                options=unique_vals,
                help="Can select multiple values as OK category"
            )
            
            if ok_values:
                ko_values = [v for v in unique_vals if v not in ok_values]
                st.write(f"**OK**: {ok_values}")
                st.write(f"**KO**: {ko_values}")
                
                # Confirm button
                if st.button("✅ Confirm Configuration", type="primary"):
                    st.session_state.label_col = label_col
                    st.session_state.ok_values = ok_values
                    st.session_state.ko_values = ko_values
                    st.session_state.config_step = 3
                    st.success("Configuration saved! Proceeding to Step 3...")
                    st.rerun()
        
        # Back button
        if st.button("← Back to Step 1"):
            st.session_state.config_step = 1
            st.rerun()
    
    # ========== STEP 3: Preprocessing ==========
    elif st.session_state.config_step == 3:
        st.subheader("🔧 Step 3: Preprocessing Data")
        st.markdown("Configure and apply data preprocessing")
        
        if 'current_data' not in st.session_state or 'label_col' not in st.session_state:
            st.warning("⚠️ Please complete Steps 1-2 first")
            if st.button("← Back to Step 1"):
                st.session_state.config_step = 1
                st.rerun()
            return
        
        df = st.session_state.current_data
        
        # Show current config
        st.info(
            f"**Data**: {df.shape[0]} rows × {df.shape[1]} cols\n"
            f"**Label Column**: {st.session_state.label_col}\n"
            f"**OK Values**: {st.session_state.ok_values}\n"
            f"**KO Values**: {st.session_state.ko_values}"
        )
        
        # Preprocessing options
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Missing Value Handling**")
            missing_strategy = st.radio(
                "How to handle missing values:",
                ["Drop rows", "Fill with median", "Fill with mean"],
                key="missing_strategy"
            )
        
        with col2:
            st.markdown("**Categorical Encoding**")
            encoding_method = st.radio(
                "Encoding method:",
                ["One-hot", "Label encoding"],
                key="encoding_method"
            )
        
        # Preprocess button
        if st.button("🚀 Start Preprocessing", type="primary"):
            with st.spinner("Preprocessing data..."):
                try:
                    preprocessor = DataPreprocessor()
                    
                    # Step 1: Create OK/KO labels
                    processed_df = preprocessor.create_ok_ko_labels(
                        df=df,
                        label_col=st.session_state.label_col,
                        ok_values=st.session_state.ok_values,
                        drop_original=True
                    )
                    
                    # Step 2: Handle missing values - convert strategy string to dict
                    if missing_strategy == 'drop':
                        strategy_dict = {col: 'drop' for col in processed_df.columns 
                                       if processed_df[col].isnull().any()}
                    elif missing_strategy == 'median':
                        strategy_dict = {col: 'median' for col in processed_df.columns 
                                       if processed_df[col].dtype in ['int64', 'float64'] 
                                       and processed_df[col].isnull().any()}
                    else:  # mean
                        strategy_dict = {col: 'mean' for col in processed_df.columns 
                                       if processed_df[col].dtype in ['int64', 'float64'] 
                                       and processed_df[col].isnull().any()}
                    
                    processed_df = preprocessor.handle_missing_values(
                        df=processed_df,
                        strategy=strategy_dict if strategy_dict else None
                    )
                    
                    # Step 3: Encode categorical variables - automatically encode all categorical columns
                    # Pass None to use default behavior (encode all except OK_KO_Label)
                    processed_df = preprocessor.encode_categorical_variables(
                        df=processed_df,
                        encoding_methods=None  # Auto-encode all categorical columns
                    )
                    
                    # Step 4: Scale numerical features
                    processed_df = preprocessor.scale_numerical_features(
                        df=processed_df,
                        method='standard'
                    )
                    
                    st.session_state.processed_df = processed_df
                    st.session_state.preprocessing_summary = preprocessor.get_preprocessing_summary(df, processed_df)
                    st.session_state.config_step = 4
                    
                    st.success("✅ Preprocessing completed! Proceeding to AI Agent configuration...")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Preprocessing failed: {str(e)}")
        
        # Back button
        if st.button("← Back"):
            st.session_state.config_step = 2
            st.rerun()
    
    # ========== STEP 4: AI Agent Settings ==========
    elif st.session_state.config_step == 4:
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
                from agent import StatisticalAgent
                
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
    
    # ========== STEP 5: Complete ==========
    elif st.session_state.config_step == 5:
        st.subheader("✅ Step 5: Configuration Complete")
        
        if 'processed_df' not in st.session_state:
            st.warning("⚠️ Configuration incomplete")
            return
        
        processed_df = st.session_state.processed_df
        summary = st.session_state.get('preprocessing_summary', {})
        
        # Success message
        st.success("🎉 Data is ready for analysis!")
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", processed_df.shape[0])
        with col2:
            st.metric("Columns", processed_df.shape[1])
        with col3:
            if 'OK_KO_Label' in processed_df.columns:
                ok_count = (processed_df['OK_KO_Label'] == 'OK').sum()
                st.metric("OK Samples", ok_count)
        
        # Data preview
        st.subheader("Processed Data Preview")
        st.dataframe(processed_df.head(20), height=300)
        
        # Next steps
        st.markdown("---")
        st.subheader("📋 Next Steps")
        st.markdown("""
        ✅ Configuration complete! You can now:
        
        1. **Raw Data** → View raw data statistics
        2. **Preprocessing Results** → Review preprocessing details
        3. **Data Analysis** → Explore features
        4. **Advanced Analysis** → Run AutoGluon feature importance
        5. **Model Training** → Train discriminative models
        6. **AI Agent Chat** → Ask questions in natural language
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
