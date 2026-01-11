"""
Step 2: Configure OK/KO Labels
"""

import streamlit as st
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from data_processing import DataLoader


def display():
    """Display Step 2: Configure OK/KO Labels"""
    st.subheader("🏷️ Step 2: Configure OK/KO Labels")
    st.markdown("Define which values represent OK and KO states")
    
    if 'current_data' not in st.session_state:
        st.warning("⚠️ Please complete Step 1 first")
        if st.button("← Back to Step 1"):
            st.session_state.config_step = 1
            st.rerun()
        return
    
    df = st.session_state.current_data
    
    st.markdown("**Select column values that represent OK state**")

    loader = DataLoader()
    suggested_cols = loader.suggest_label_columns(df)

    if suggested_cols:
        st.info(f"💡 Suggested label columns: {', '.join(suggested_cols)}")

    label_col = st.selectbox(
        "Select Label Column:",
        options=df.columns,
        help="Column containing OK/KO classification",
        key="label_col_values"
    )

    if label_col:
        unique_vals = df[label_col].dropna().unique().tolist()
        st.write(f"**Unique values in '{label_col}**: {unique_vals}")

        ok_values = st.multiselect(
            "Select values as 'OK':",
            options=unique_vals,
            help="Can select multiple values as OK category",
            key="ok_values_select"
        )

        if ok_values:
            ko_values = [v for v in unique_vals if v not in ok_values]
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**✅ OK values**: {ok_values}")
            with col2:
                st.write(f"**❌ KO values**: {ko_values}")

            if st.button("✅ Confirm Configuration", type="primary", key="confirm_values"):
                ok_values_native = [str(v) if not isinstance(v, (str, int, float, bool)) else v for v in ok_values]
                ko_values_native = [str(v) if not isinstance(v, (str, int, float, bool)) else v for v in ko_values]

                st.session_state.label_col = label_col
                st.session_state.ok_values = ok_values_native
                st.session_state.ko_values = ko_values_native
                st.session_state.pop('confirmed_threshold_value', None)
                st.session_state.config_step = 3
                st.success("Configuration saved! Proceeding to Step 3...")
                st.rerun()

    if st.button("← Back to Step 1", key="back_step1_from_step2"):
        st.session_state.config_step = 1
        st.rerun()
