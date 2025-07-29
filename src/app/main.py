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

def display_sidebar(df):
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
            st.session_state['label_col'] = label_col
            unique_vals = df[label_col].dropna().unique()
            # TODO: Considering the case where there might be more than two unique values
            if len(unique_vals) < 2:
                st.warning("Not enough unique values for OK/KO analysis. Please select a different label column.")
                return
            if len(unique_vals) == 2:
                target_val = st.selectbox("OK value:", unique_vals, key="okko_value_select")
                btn1, btn2 = st.columns(2)
                with btn1:
                    if st.button("Confirm", key="okko_confirm"):
                        st.session_state['okko_value'] = target_val
                        st.toast(f"Set {target_val} as OK for {label_col}.")
                with btn2:
                    if st.button("Cancel", key="okko_cancel"):
                        st.session_state['okko_value'] = None

def display_data_section(df):
    """Upper main area with tab-based data exploration"""
    tab1, tab2 = st.tabs(["📊 Raw Data", "📈 Feature Stats"])

    with tab1:
        st.dataframe(df, height=300)
        st.caption(f"Showing {len(df)} rows with {len(df.columns)} features")
        st.write("Data Statistics:")
        stat_type = st.radio(
            "Select statistics type:",
            options=["All", "Numerical", "Categorical"],
            index=0,
            horizontal=True,
            help="Choose which columns to show in statistics table"
        )

        if stat_type == "All":
            desc = df.describe(include='all')
            num_cols = df.select_dtypes(include='number').columns
            desc.loc['variance', num_cols] = df[num_cols].var()
            desc.loc['mode'] = df.mode().iloc[0]
            st.dataframe(desc)
        elif stat_type == "Numerical":
            desc = df.describe(include='number')
            desc.loc['variance'] = df.var(numeric_only=True)
            desc.loc['mode'] = df.mode(numeric_only=True).iloc[0]
            st.dataframe(desc)
        elif stat_type == "Categorical":
            desc = df.describe(include='object')
            obj_cols = df.select_dtypes(include='object').columns
            desc.loc['mode', obj_cols] = df[obj_cols].mode().iloc[0]
            st.dataframe(desc)
        
    with tab2:
        okko_expanded = st.toggle("Show OK/KO Analysis", value=True)
        if okko_expanded and 'okko_value' in st.session_state and 'label_col' in st.session_state:
            st.subheader("OK/KO Analysis")
            df['__okko__'] = df[st.session_state['label_col']].apply(lambda x: 'OK' if x == st.session_state['okko_value'] else 'KO')
            pie_data = df['__okko__'].value_counts()
            st.write(f"Current OK value: {st.session_state['okko_value']}")
            col1, col2 = st.columns(2)
            with col1:
                fig1, ax1 = plt.subplots(figsize=(3,3))
                ax1.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=90)
                ax1.set_title("OK/KO Pie Chart")
                st.pyplot(fig1, use_container_width=True)
            with col2:
                fig2, ax2 = plt.subplots(figsize=(3,3))
                ax2.bar(pie_data.index, pie_data.values, color=['green','red'])
                ax2.set_title("OK/KO Bar Chart")
                st.pyplot(fig2, use_container_width=True)
        
        st.subheader("Multi-Feature Analysis")
        selected_features = st.multiselect("Select one or more features:", df.columns)
        chart_type = st.radio("Chart Type", ["Line Chart", "Bar Chart", "Crosstab"])
        if selected_features:
            st.write(f"Selected features: {selected_features}")
            if chart_type == "Line Chart":
                fig_line, ax_line = plt.subplots()
                df[selected_features].plot(ax=ax_line)
                st.pyplot(fig_line)
            elif chart_type == "Bar Chart":
                bar_data = df[selected_features[0]].value_counts()
                fig_bar, ax_bar = plt.subplots()
                ax_bar.bar(bar_data.index.astype(str), bar_data.values)
                ax_bar.set_xlabel(selected_features[0])
                ax_bar.set_ylabel("Count")
                st.pyplot(fig_bar)
            elif chart_type == "Crosstab" and len(selected_features) >= 2:
                ct = pd.crosstab(df[selected_features[0]], df[selected_features[1]])
                st.dataframe(ct)

        st.subheader("AI Analysis Console")
        user_query = st.text_area(
            "Enter your analysis request (e.g. 'Show correlation between Age and Survival')",
            height=100
        )
        if user_query:
            st.warning("AI integration pending implementation")

def main():
    # TODO: Load data from a different source if needed
    df = load_default_data()
    
    if not df.empty:
        # Build interface sections
        display_sidebar(df)
        display_data_section(df) 
    else:
        st.warning("Please place your dataset in data/raw/ folder")

# Run the app
if __name__ == "__main__":
    main()