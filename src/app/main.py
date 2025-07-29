import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


# Initialize app components
st.set_page_config(layout="wide")

@st.cache_data
def load_default_data():
    """Load default dataset from /data/raw/train.csv without preprocessing"""
    try:
        return pd.read_csv("data/raw/train.csv")
    except FileNotFoundError:
        st.error("Default dataset not found: data/raw/train.csv")
        return pd.DataFrame()

def build_sidebar():
    """Configure all sidebar controls"""
    with st.sidebar:
        st.header("⚙️ Control Panel")
        
        # Dataset Configuration
        with st.expander("DATA CONFIG"):
            global label_col
            label_col = st.selectbox(
                "Label Column (OK/KO)", 
                options=df.columns,
                help="Select which column contains OK/KO labels"
            )
            
        # Preprocessing Options
        with st.expander("PREPROCESSING"):
            st.radio(
                "Missing Value Strategy",
                options=["Mean", "Median", "Custom Value"],
                key="na_strategy"
            )
            if st.session_state.na_strategy == "Custom Value":
                st.number_input("Fill Value", key="fill_value")

def display_data_section():
    """Upper main area with tab-based data exploration"""
    tab1, tab2, tab3 = st.tabs(["📊 Raw Data", "📈 Basic Stats", "🔍 Advanced View"])
    
    with tab1:
        st.dataframe(df, height=300)
        st.caption(f"Showing {len(df)} rows with {len(df.columns)} features")

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.write("Numerical Features")
            st.write(df.select_dtypes(include='number').describe())
        with col2:
            st.write("Categorical Features")
            st.write(df.select_dtypes(exclude='number').describe())

    with tab3:
        fig, ax = plt.subplots()
        df.plot.scatter(x='Age', y='Fare', c=df[label_col].map({'OK':'green','KO':'red'}), ax=ax)
        st.pyplot(fig)

def build_ai_console():
    """Lower section for AI interaction""" 
    st.subheader("AI Analysis Console")
    user_query = st.text_area(
        "Enter your analysis request (e.g. 'Show correlation between Age and Survival')",
        height=100
    )
    
    if user_query:
        st.warning("AI integration pending implementation")
        # Future LLM processing will happen here

# Main app flow
if __name__ == "__main__":
    df = load_default_data()
    
    if not df.empty:
        # Build interface sections
        build_sidebar()
        display_data_section() 
        build_ai_console()
    else:
        st.warning("Please place your dataset in data/raw/ folder")