"""
Model Training Tab - Train and compare discriminative models
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


def display():
    """Display model training tab"""
    if 'processed_df' not in st.session_state:
        st.info("Please complete configuration first.")
        return
    
    if 'analysis_results' not in st.session_state:
        st.warning("⚠️ Please run Advanced Analysis first to get feature importance.")
        return
    
    processed_df = st.session_state['processed_df']
    analysis_results = st.session_state['analysis_results']
    
    st.subheader("🎯 Model Training - Simple Discriminative Models")
    st.markdown("Train simple models (Logistic Regression, SVM, Decision Tree, Random Forest) using top features")
    
    # Feature source selection (Statistical vs ML)
    has_statistical = bool(analysis_results.get('feature_ranking'))
    has_ml = bool(analysis_results.get('ml_feature_importance', {})
                   .get('feature_importance', {})
                   .get('feature_ranking'))

    if not has_statistical and not has_ml:
        st.error("❌ No feature importance ranking found. Please run Advanced Analysis first.")
        st.info("📌 How to fix:\n1. Go to 'Advanced Analysis' tab\n2. Select at least one analysis method (Statistical Tests or ML Feature Importance)\n3. Click '🚀 Run Advanced Analysis'\n4. Return to this tab")
        return

    feature_source_options = []
    if has_statistical:
        feature_source_options.append("Statistical Analysis")
    if has_ml:
        feature_source_options.append("AutoGluon ML Analysis")

    default_index = 0
    if has_ml and has_statistical:
        # prefer ML as default when both available
        default_index = 1

    feature_source = st.radio(
        "Select feature ranking source for training:",
        options=feature_source_options,
        index=default_index,
        help="Choose which analysis method's feature ranking to use for model training"
    )

    feature_names = []
    if feature_source == "AutoGluon ML Analysis":
        feature_ranking_ml = analysis_results['ml_feature_importance']['feature_importance']['feature_ranking']
        feature_names = [f['feature'] for f in feature_ranking_ml]
    else:
        feature_names = [f['feature'] for f in analysis_results.get('feature_ranking', [])]

    if not feature_names:
        st.error("❌ Failed to extract feature ranking from the selected source.")
        return

    st.success(f"✅ Using {len(feature_names)} features from {feature_source}")
    
    # Configuration
    col1, col2 = st.columns(2)
    with col1:
        max_available = len(feature_names)
        
        # 从3到max_available生成所有可选项
        valid_options = list(range(1, max_available + 1))
        
        # 默认选择3个均匀分布的值
        default_counts = [3, max_available // 2, max_available] if max_available >= 3 else [max_available]
        default_counts = sorted(list(set(default_counts)))
        
        n_features_options = st.multiselect(
            "Select feature counts to test:",
            valid_options,
            default=default_counts,
            help="Choose how many top features to test (max 5 configurations)",
            max_selections=10
        )
        n_features_options = sorted(list(set(n_features_options)))
    
    with col2:
        model_options = st.multiselect(
            "Select models to train:",
            ["Logistic Regression", "SVM", "Decision Tree", "Random Forest"],
            default=["Logistic Regression", "Decision Tree", "Random Forest"],
            help="Different model types to compare"
        )
        if not model_options:
            model_options = ["Logistic Regression", "Decision Tree"]
    
    # Train button
    if st.button("🚀 Train Models", type="primary"):
        with st.spinner("Training models with different feature counts..."):
            try:
                from analysis.model_trainer import ModelTrainer
                
                trainer = ModelTrainer()
                model_map = {
                    "Logistic Regression": "logistic",
                    "SVM": "svm",
                    "Decision Tree": "dt",
                    "Random Forest": "rf"
                }
                selected_models = [model_map[m] for m in model_options]
                
                results = trainer.train_models_with_feature_selection(
                    df=processed_df,
                    feature_importance_ranking=feature_names,
                    feature_counts=n_features_options,
                    model_names=selected_models
                )
                
                if results["success"]:
                    st.session_state['training_results'] = results
                    st.success(results["message"])
                else:
                    st.error(results["message"])
            except Exception as e:
                st.error(f"❌ Training failed: {str(e)}")
    
    # Display results
    if 'training_results' in st.session_state:
        results = st.session_state['training_results']
                
        st.subheader("🏆 Best Model Details")
        best = results["best_model"]
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Model", best['name'].upper())
        with col2:
            st.metric("Features", f"{best['n_features']}/{len(feature_names)}")
        with col3:
            st.metric("Accuracy", f"{best['accuracy']:.4f}")
        with col4:
            st.metric("F1 Score", f"{best['f1']:.4f}")
        with col5:
            st.metric("Recall", f"{best['recall']:.4f}")
        
        # Plots
        plot_data = results["plot_data"]
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📈 Accuracy vs Feature Count")
            fig, ax = plt.subplots(figsize=(10, 6))
            for model, accs in plot_data["feature_vs_accuracy"].items():
                feature_vals = plot_data["feature_counts"][:len(accs)]
                ax.plot(feature_vals, accs, marker="o", label=model.upper(), linewidth=2.5, markersize=8)
            ax.set_xlabel("Number of Features", fontweight='bold')
            ax.set_ylabel("Accuracy", fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        
        with col2:
            st.subheader("🎯 Model Comparison")
            fig, ax = plt.subplots(figsize=(10, 6))
            models = [m.upper() for m in plot_data["model_comparison"].keys()]
            accs = list(plot_data["model_comparison"].values())
            bars = ax.bar(models, accs, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            ax.set_ylabel("Average Accuracy", fontweight='bold')
            ax.set_ylim([0, 1.1])
            for bar, v in zip(bars, accs):
                ax.text(bar.get_x() + bar.get_width()/2, v + 0.03, 
                       f"{v:.3f}", ha="center", fontweight='bold')
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        # 📊 Performance Summary (All Feature/Model Combinations) at bottom
        st.subheader("📊 Performance Summary (All Feature/Model Combinations)")
        perf_df = results["performance_summary"].copy()
        st.dataframe(
            perf_df.style.highlight_max(subset=['Accuracy'], color='lightgreen'),
            use_container_width=True,
            height=400,
        )
    
