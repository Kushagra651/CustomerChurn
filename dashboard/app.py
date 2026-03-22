"""
app.py
-------
PulseDirects Customer Churn Prediction Dashboard

Pages:
  1. Overview    — KPIs and churn summary
  2. EDA         — exploratory data analysis charts
  3. Model       — model comparison and PR curve
  4. Prediction  — live single customer prediction via FastAPI
"""

import json
import requests
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PulseDirects Churn Dashboard",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

FASTAPI_URL  = "http://fastapi:8000"
DATA_PATH    = Path("/app/data/preprocessed/churn_preprocessed.csv")
COMPARE_PATH = Path("/app/models/model_comparison.csv")
BEST_PATH    = Path("/app/models/best_model_info.json")
EVAL_PATH    = Path("/app/models/evaluation/evaluation_report.json")
PR_IMG_PATH  = Path("/app/models/evaluation/precision_recall.png")
CMP_IMG_PATH = Path("/app/models/evaluation/model_comparison.png")

PRIMARY = "#4F46E5"
SUCCESS = "#10B981"
DANGER  = "#EF4444"
WARNING = "#F59E0B"

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .section-header {
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #4F46E5;
    }
    .risk-high   { color: #EF4444; font-weight: 700; }
    .risk-medium { color: #F59E0B; font-weight: 700; }
    .risk-low    { color: #10B981; font-weight: 700; }
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_data():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return pd.DataFrame()

@st.cache_data
def load_model_comparison():
    if COMPARE_PATH.exists():
        return pd.read_csv(COMPARE_PATH)
    return pd.DataFrame()

@st.cache_data
def load_best_model_info():
    if BEST_PATH.exists():
        with open(BEST_PATH) as f:
            return json.load(f)
    return {}

@st.cache_data
def load_eval_report():
    if EVAL_PATH.exists():
        with open(EVAL_PATH) as f:
            return json.load(f)
    return {}

def call_predict_api(payload: dict):
    try:
        response = requests.post(
            f"{FASTAPI_URL}/predict",
            json=payload,
            timeout=10
        )
        if response.status_code == 200:
            return response.json(), None
        return None, response.text
    except Exception as e:
        return None, str(e)

def get_api_health():
    try:
        r = requests.get(f"{FASTAPI_URL}/health", timeout=5)
        return r.json() if r.status_code == 200 else None
    except:
        return None

def set_plot_style():
    plt.rcParams.update({
        'figure.facecolor' : 'white',
        'axes.facecolor'   : '#FAFAFA',
        'axes.spines.top'  : False,
        'axes.spines.right': False,
        'font.family'      : 'sans-serif',
        'axes.titlesize'   : 12,
        'axes.labelsize'   : 10,
    })


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 📡 PulseDirects")
    st.markdown("**Customer Churn Prediction**")
    st.divider()

    page = st.radio(
        "Navigate",
        ["🏠 Overview", "📊 EDA", "🤖 Model Performance", "🔮 Live Prediction"],
        label_visibility="collapsed"
    )

    st.divider()

    health = get_api_health()
    if health and health.get("model_loaded"):
        st.success("API ✅ Online")
        st.caption(f"Model: {health.get('model_name')}")
    else:
        st.error("API ❌ Offline")

    st.divider()
    st.caption("ETL Pipeline: Apache Airflow")
    st.caption("Model Serving: FastAPI")
    st.caption("Database: MySQL")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════

if page == "🏠 Overview":
    st.title("🏠 Churn Overview")
    st.markdown("High-level summary of customer churn across the PulseDirects customer base.")

    df       = load_data()
    info     = load_best_model_info()
    eval_rep = load_eval_report()

    if df.empty:
        st.warning("Data not loaded. Run the ETL pipeline first.")
        st.stop()

    total      = len(df)
    churned    = int(df['Churn Value'].sum())
    retained   = total - churned
    churn_rate = churned / total * 100

    # ── KPI Cards ─────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", f"{total:,}")
    c2.metric("Churned",         f"{churned:,}",  delta=f"{churn_rate:.1f}%", delta_color="inverse")
    c3.metric("Retained",        f"{retained:,}")
    c4.metric("Best ROC-AUC",    f"{info.get('roc_auc', 'N/A')}")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="section-header">Churn Distribution</p>', unsafe_allow_html=True)
        set_plot_style()
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.pie(
            [retained, churned],
            labels=["Retained", "Churned"],
            colors=[SUCCESS, DANGER],
            autopct='%1.1f%%',
            startangle=90,
            wedgeprops=dict(width=0.6)
        )
        ax.set_title("Churn Distribution", fontsize=13)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown('<p class="section-header">Churn Rate by Contract Type</p>', unsafe_allow_html=True)
        contract_map = {0: 'Month-to-month', 1: 'One year', 2: 'Two year'}
        df['Contract_Label'] = df['Contract'].map(contract_map).fillna('Unknown')
        contract_churn = df.groupby('Contract_Label')['Churn Value'].mean().reset_index()
        contract_churn['Churn Rate %'] = (contract_churn['Churn Value'] * 100).round(1)

        fig2, ax2 = plt.subplots(figsize=(5, 4))
        bars = ax2.bar(
            contract_churn['Contract_Label'],
            contract_churn['Churn Rate %'],
            color=[DANGER, WARNING, SUCCESS],
            edgecolor='white', linewidth=0.5
        )
        for bar in bars:
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f'{bar.get_height():.1f}%',
                ha='center', va='bottom', fontsize=9
            )
        ax2.set_ylabel('Churn Rate (%)')
        ax2.set_title('Churn Rate by Contract Type', fontsize=13)
        ax2.set_ylim(0, 60)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    st.divider()

    # ── Model summary ──────────────────────────────────────────────
    st.markdown('<p class="section-header">Best Model Summary</p>', unsafe_allow_html=True)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Model",    info.get('best_model', 'N/A'))
    m2.metric("ROC-AUC",  info.get('roc_auc',   'N/A'))
    m3.metric("Accuracy", info.get('accuracy',  'N/A'))
    m4.metric("F1 Score", info.get('f1_score',  'N/A'))
    m5.metric("Recall",   info.get('recall',    'N/A'))

    # ── Confusion matrix ───────────────────────────────────────────
    if eval_rep.get('confusion_matrix'):
        st.divider()
        st.markdown('<p class="section-header">Confusion Matrix</p>', unsafe_allow_html=True)
        cm = eval_rep['confusion_matrix']
        cm_array = np.array([
            [cm['true_negative'],  cm['false_positive']],
            [cm['false_negative'], cm['true_positive']]
        ])
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm_array, annot=True, fmt='d',
            cmap='Blues', ax=ax_cm,
            xticklabels=['Predicted: No', 'Predicted: Yes'],
            yticklabels=['Actual: No', 'Actual: Yes'],
            linewidths=0.5
        )
        ax_cm.set_title('Confusion Matrix', fontsize=13)
        plt.tight_layout()
        col_cm, _ = st.columns([1, 1])
        with col_cm:
            st.pyplot(fig_cm)
        plt.close()


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EDA
# ═════════════════════════════════════════════════════════════════════════════

elif page == "📊 EDA":
    st.title("📊 Exploratory Data Analysis")
    df = load_data()

    if df.empty:
        st.warning("Data not loaded.")
        st.stop()

    set_plot_style()

    if 'Churn Label' not in df.columns:
        df['Churn Label'] = df['Churn Value'].map({0: 'Not Churned', 1: 'Churned'})

    # ── Tenure distribution ────────────────────────────────────────
    st.markdown('<p class="section-header">Tenure Distribution by Churn</p>', unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for val, color, label in [(0, SUCCESS, 'Retained'), (1, DANGER, 'Churned')]:
        axes[0].hist(
            df[df['Churn Value'] == val]['Tenure Months'],
            bins=30, alpha=0.6, color=color, label=label
        )
    axes[0].set_title('Tenure Distribution')
    axes[0].set_xlabel('Tenure (Months)')
    axes[0].legend()

    axes[1].boxplot(
        [df[df['Churn Value'] == 0]['Tenure Months'],
         df[df['Churn Value'] == 1]['Tenure Months']],
        labels=['Retained', 'Churned'],
        patch_artist=True,
        boxprops=dict(facecolor='#E0E7FF'),
        medianprops=dict(color=PRIMARY, linewidth=2)
    )
    axes[1].set_title('Tenure Boxplot')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="section-header">Monthly Charges by Churn</p>', unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        for val, color, label in [(0, SUCCESS, 'Retained'), (1, DANGER, 'Churned')]:
            ax2.hist(
                df[df['Churn Value'] == val]['Monthly Charges'],
                bins=30, alpha=0.6, color=color, label=label
            )
        ax2.set_title('Monthly Charges Distribution')
        ax2.set_xlabel('Monthly Charges ($)')
        ax2.legend()
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    with col2:
        st.markdown('<p class="section-header">Average Charges by Churn</p>', unsafe_allow_html=True)
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        ax3.boxplot(
            [df[df['Churn Value'] == 0]['average_charges'],
             df[df['Churn Value'] == 1]['average_charges']],
            labels=['Retained', 'Churned'],
            patch_artist=True,
            boxprops=dict(facecolor='#FEF3C7'),
            medianprops=dict(color=WARNING, linewidth=2)
        )
        ax3.set_title('Average Charges Boxplot')
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

    st.divider()

    # ── Correlation heatmap ────────────────────────────────────────
    st.markdown('<p class="section-header">Correlation Heatmap</p>', unsafe_allow_html=True)
    num_cols = ['Tenure Months', 'Monthly Charges', 'Total Charges',
                'average_charges', 'Churn Value']
    num_cols = [c for c in num_cols if c in df.columns]
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        df[num_cols].corr().round(2),
        annot=True, fmt='.2f',
        cmap='RdYlGn', ax=ax4,
        linewidths=0.5, square=True
    )
    ax4.set_title('Correlation Matrix', fontsize=13)
    plt.tight_layout()
    st.pyplot(fig4)
    plt.close()

    st.divider()

    # ── High value customer ────────────────────────────────────────
    st.markdown('<p class="section-header">High Value Customer Churn Rate</p>', unsafe_allow_html=True)
    hv = df.groupby('high_value_customer')['Churn Value'].mean().reset_index()
    hv['Label'] = hv['high_value_customer'].map({0: 'Standard', 1: 'High Value'})
    hv['Churn Rate %'] = (hv['Churn Value'] * 100).round(1)

    fig5, ax5 = plt.subplots(figsize=(5, 4))
    bars5 = ax5.bar(
        hv['Label'], hv['Churn Rate %'],
        color=[PRIMARY, WARNING],
        edgecolor='white', linewidth=0.5
    )
    for bar in bars5:
        ax5.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f'{bar.get_height():.1f}%',
            ha='center', va='bottom', fontsize=10
        )
    ax5.set_ylabel('Churn Rate (%)')
    ax5.set_title('High Value vs Standard Customers', fontsize=13)
    plt.tight_layout()
    col_hv, _ = st.columns([1, 1])
    with col_hv:
        st.pyplot(fig5)
    plt.close()

    st.divider()
    st.markdown('<p class="section-header">Dataset Preview</p>', unsafe_allow_html=True)
    st.dataframe(df.head(20), use_container_width=True)
    st.caption(f"Rows: {len(df):,} | Columns: {len(df.columns)}")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL PERFORMANCE
# ═════════════════════════════════════════════════════════════════════════════

elif page == "🤖 Model Performance":
    st.title("🤖 Model Performance")

    comparison_df = load_model_comparison()
    info          = load_best_model_info()

    if comparison_df.empty:
        st.warning("Model comparison data not found. Run train_model.py first.")
        st.stop()

    st.success(
        f"🏆 Best Model: **{info.get('best_model')}** | "
        f"ROC-AUC: **{info.get('roc_auc')}** | "
        f"Accuracy: **{info.get('accuracy')}** | "
        f"F1: **{info.get('f1_score')}**"
    )

    st.divider()

    # ── Model comparison table ─────────────────────────────────────
    st.markdown('<p class="section-header">Model Comparison Table</p>', unsafe_allow_html=True)
    display_cols = ['Model', 'ROC-AUC', 'Accuracy', 'F1 Score',
                    'Precision', 'Recall', 'CV ROC-AUC Mean', 'CV ROC-AUC Std']
    display_cols = [c for c in display_cols if c in comparison_df.columns]
    st.dataframe(
        comparison_df[display_cols].style.highlight_max(
            subset=['ROC-AUC', 'Accuracy', 'F1 Score'],
            color='#D1FAE5'
        ).format(precision=4),
        use_container_width=True
    )

    st.divider()

    # ── Model comparison chart image ───────────────────────────────
    st.markdown('<p class="section-header">Model Comparison Chart</p>', unsafe_allow_html=True)
    if CMP_IMG_PATH.exists():
        st.image(str(CMP_IMG_PATH), use_column_width=True)
    else:
        # Draw from CSV if image missing
        set_plot_style()
        metric = st.selectbox("Select metric", ['ROC-AUC', 'Accuracy', 'F1 Score', 'Precision', 'Recall'])
        if metric in comparison_df.columns:
            sorted_df = comparison_df.sort_values(metric, ascending=True)
            fig, ax = plt.subplots(figsize=(10, 5))
            bars = ax.barh(sorted_df['Model'], sorted_df[metric], color=PRIMARY, alpha=0.85)
            for bar in bars:
                ax.text(
                    bar.get_width() + 0.005,
                    bar.get_y() + bar.get_height() / 2,
                    f'{bar.get_width():.4f}',
                    va='center', fontsize=9
                )
            ax.set_xlim(0, 1.1)
            ax.set_title(f'Model Comparison — {metric}', fontsize=13)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    st.divider()

    # ── PR Curve ───────────────────────────────────────────────────
    st.markdown('<p class="section-header">Precision-Recall Curve</p>', unsafe_allow_html=True)
    if PR_IMG_PATH.exists():
        st.image(str(PR_IMG_PATH), use_column_width=True)
    else:
        st.info("PR curve not found. Run evaluate_model.py first.")

    # ── CV scores bar chart ────────────────────────────────────────
    if 'CV ROC-AUC Mean' in comparison_df.columns:
        st.divider()
        st.markdown('<p class="section-header">Cross-Validation ROC-AUC</p>', unsafe_allow_html=True)
        set_plot_style()
        fig_cv, ax_cv = plt.subplots(figsize=(10, 4))
        x = np.arange(len(comparison_df))
        bars_cv = ax_cv.bar(
            comparison_df['Model'],
            comparison_df['CV ROC-AUC Mean'],
            yerr=comparison_df['CV ROC-AUC Std'],
            color=PRIMARY, alpha=0.85,
            capsize=5, edgecolor='white'
        )
        ax_cv.set_ylim(0, 1.1)
        ax_cv.set_ylabel('CV ROC-AUC')
        ax_cv.set_title('Cross-Validation ROC-AUC by Model', fontsize=13)
        plt.xticks(rotation=15, ha='right')
        plt.tight_layout()
        st.pyplot(fig_cv)
        plt.close()


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4 — LIVE PREDICTION
# ═════════════════════════════════════════════════════════════════════════════

elif page == "🔮 Live Prediction":
    st.title("🔮 Live Churn Prediction")
    st.markdown("Fill in customer details below to get an instant churn prediction.")

    health = get_api_health()
    if not health or not health.get("model_loaded"):
        st.error("FastAPI is offline. Cannot make predictions.")
        st.stop()

    with st.form("prediction_form"):
        st.markdown('<p class="section-header">Demographics</p>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            gender         = st.selectbox("Gender",         ["Female", "Male"])
            senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        with col2:
            partner        = st.selectbox("Partner",    ["No", "Yes"])
            dependents     = st.selectbox("Dependents", ["No", "Yes"])
        with col3:
            tenure_months  = st.slider("Tenure (Months)", 0, 72, 12)

        st.divider()
        st.markdown('<p class="section-header">Services</p>', unsafe_allow_html=True)
        col4, col5, col6 = st.columns(3)
        with col4:
            phone_service    = st.selectbox("Phone Service",    ["No", "Yes"])
            multiple_lines   = st.selectbox("Multiple Lines",   ["No", "Yes", "No phone service"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        with col5:
            online_security  = st.selectbox("Online Security",   ["No", "Yes", "No internet service"])
            online_backup    = st.selectbox("Online Backup",     ["No", "Yes", "No internet service"])
            device_protection= st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        with col6:
            tech_support     = st.selectbox("Tech Support",     ["No", "Yes", "No internet service"])
            streaming_tv     = st.selectbox("Streaming TV",     ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

        st.divider()
        st.markdown('<p class="section-header">Billing</p>', unsafe_allow_html=True)
        col7, col8, col9 = st.columns(3)
        with col7:
            contract          = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
        with col8:
            payment_method  = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)"
            ])
            monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0)
        with col9:
            total_charges   = st.number_input("Total Charges ($)", 0.0, 10000.0, 1500.0)
            support_tickets = st.slider("Support Tickets", 0, 20, 2)

        submitted = st.form_submit_button("🔮 Predict Churn", use_container_width=True)

    if submitted:
        yes_no   = {"No": 0, "Yes": 1, "No phone service": 0, "No internet service": 0}
        ml_map   = {"No": 0, "Yes": 1, "No phone service": 2}
        is_map   = {"DSL": 0, "Fiber optic": 1, "No": 2}
        ct_map   = {"Month-to-month": 0, "One year": 1, "Two year": 2}
        pm_map   = {
            "Bank transfer (automatic)": 0,
            "Credit card (automatic)"  : 1,
            "Electronic check"         : 2,
            "Mailed check"             : 3
        }

        avg_charges = total_charges / tenure_months if tenure_months > 0 else monthly_charges
        high_value  = 1 if monthly_charges > 89.85 else 0

        payload = {
            "Gender"              : 1 if gender == "Male" else 0,
            "Senior_Citizen"      : yes_no[senior_citizen],
            "Partner"             : yes_no[partner],
            "Dependents"          : yes_no[dependents],
            "Tenure_Months"       : tenure_months,
            "Phone_Service"       : yes_no[phone_service],
            "Multiple_Lines"      : ml_map.get(multiple_lines, 0),
            "Internet_Service"    : is_map.get(internet_service, 0),
            "Online_Security"     : yes_no[online_security],
            "Online_Backup"       : yes_no[online_backup],
            "Device_Protection"   : yes_no[device_protection],
            "Tech_Support"        : yes_no[tech_support],
            "Streaming_Tv"        : yes_no[streaming_tv],
            "Streaming_Movies"    : yes_no[streaming_movies],
            "Contract"            : ct_map[contract],
            "Paperless_Billing"   : yes_no[paperless_billing],
            "Payment_Method"      : pm_map[payment_method],
            "Monthly_Charges"     : monthly_charges,
            "Total_Charges"       : total_charges,
            "Support_Ticket_Count": support_tickets,
            "average_charges"     : round(avg_charges, 2),
            "high_value_customer" : high_value
        }

        with st.spinner("Getting prediction from API..."):
            result, error = call_predict_api(payload)

        if error:
            st.error(f"Prediction failed: {error}")
        elif result:
            st.divider()
            st.markdown("### Prediction Result")

            prob = result['churn_probability']
            risk = result['churn_risk_level']
            pred = result['churn_prediction']

            col_r1, col_r2, col_r3 = st.columns(3)
            col_r1.metric("Prediction",        "Will Churn 🚨" if pred == 1 else "Will Stay ✅")
            col_r2.metric("Churn Probability", f"{prob:.2%}")
            col_r3.metric("Risk Level",         risk)

            # Probability bar
            st.markdown(f"**Churn Probability: {prob:.1%}**")
            bar_color = DANGER if risk == "High" else WARNING if risk == "Medium" else SUCCESS
            st.progress(prob)

            risk_icon = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
            st.info(
                f"{risk_icon.get(risk,'⚪')} **{risk} Risk** — {result['message']} "
                f"| Model: {result['model_used']}"
            )

            # Probability gauge using matplotlib
            set_plot_style()
            fig_g, ax_g = plt.subplots(figsize=(6, 3))
            zones = [(0.45, SUCCESS, 'Low Risk'),
                     (0.30, WARNING, 'Medium Risk'),
                     (0.25, DANGER,  'High Risk')]
            left = 0
            for width, color, label in zones:
                ax_g.barh(0, width, left=left, color=color, alpha=0.3, height=0.4)
                left += width
            ax_g.barh(0, prob, color=bar_color, height=0.3, alpha=0.9)
            ax_g.axvline(x=prob, color='black', linewidth=2, linestyle='--')
            ax_g.set_xlim(0, 1)
            ax_g.set_yticks([])
            ax_g.set_xlabel('Churn Probability')
            ax_g.set_title(f'Risk Level: {risk} ({prob:.1%})', fontsize=12)
            patches = [mpatches.Patch(color=c, alpha=0.5, label=l)
                      for _, c, l in zones]
            ax_g.legend(handles=patches, loc='upper right', fontsize=8)
            plt.tight_layout()
            col_gauge, _ = st.columns([2, 1])
            with col_gauge:
                st.pyplot(fig_g)
            plt.close()