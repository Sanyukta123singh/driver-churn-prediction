import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# STEP 15 — STREAMLIT MONITORING DASHBOARD
# ============================================================
# What this dashboard shows:
# 1. Model performance metrics
# 2. Risk distribution across drivers
# 3. Feature importance
# 4. Intervention tracking
# 5. Business impact
# 6. Model drift indicators
# ============================================================

st.set_page_config(
    page_title="Driver Churn Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# LOAD DATA & MODEL
# ============================================================

@st.cache_data
def load_data():
    df = pd.read_csv('data/feature_matrix.csv')
    final_features = joblib.load('models/final_features.pkl')
    model = joblib.load('models/production_model.pkl')
    threshold = joblib.load('models/production_threshold.pkl')

    X = df[final_features]
    probs = model.predict_proba(X)[:, 1]
    df['churn_probability'] = probs
    df['predicted_churn'] = (probs >= threshold).astype(int)

    def get_risk_level(p):
        if p >= 0.50:
            return 'CRITICAL'
        elif p >= 0.25:
            return 'HIGH'
        elif p >= 0.15:
            return 'MEDIUM'
        else:
            return 'LOW'

    df['risk_level'] = df['churn_probability'].apply(get_risk_level)
    return df, final_features, model, threshold

df, final_features, model, threshold = load_data()

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🚗 Driver Churn Intelligence")
st.sidebar.markdown("---")

page = st.sidebar.selectbox(
    "Navigate",
    ["Overview", "Risk Analysis",
     "Feature Insights", "Business Impact"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Model Info**")
st.sidebar.markdown(f"Version: v1.0.0")
st.sidebar.markdown(f"Threshold: {threshold}")
st.sidebar.markdown(f"Features: {len(final_features)}")
st.sidebar.markdown(f"Drivers: {len(df):,}")

# ============================================================
# PAGE 1 — OVERVIEW
# ============================================================

if page == "Overview":
    st.title("Driver Churn Intelligence Dashboard")
    st.markdown("Real-time monitoring of driver churn risk across the fleet")

    # Top metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    total_drivers = len(df)
    critical = len(df[df['risk_level'] == 'CRITICAL'])
    high = len(df[df['risk_level'] == 'HIGH'])
    medium = len(df[df['risk_level'] == 'MEDIUM'])
    low = len(df[df['risk_level'] == 'LOW'])
    avg_risk = df['churn_probability'].mean()

    with col1:
        st.metric("Total Drivers", f"{total_drivers:,}")
    with col2:
        st.metric("🔴 Critical", f"{critical:,}",
                  f"{critical/total_drivers:.1%}")
    with col3:
        st.metric("🟠 High", f"{high:,}",
                  f"{high/total_drivers:.1%}")
    with col4:
        st.metric("🟡 Medium", f"{medium:,}",
                  f"{medium/total_drivers:.1%}")
    with col5:
        st.metric("🟢 Low", f"{low:,}",
                  f"{low/total_drivers:.1%}")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Risk Level Distribution")
        risk_counts = df['risk_level'].value_counts()
        fig = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            color=risk_counts.index,
            color_discrete_map={
                'CRITICAL': '#E24B4A',
                'HIGH': '#EF9F27',
                'MEDIUM': '#F5D76E',
                'LOW': '#639922'
            }
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Churn Probability Distribution")
        fig = px.histogram(
            df,
            x='churn_probability',
            color='risk_level',
            color_discrete_map={
                'CRITICAL': '#E24B4A',
                'HIGH': '#EF9F27',
                'MEDIUM': '#F5D76E',
                'LOW': '#639922'
            },
            nbins=50
        )
        fig.add_vline(
            x=threshold,
            line_dash="dash",
            line_color="black",
            annotation_text=f"Threshold={threshold}"
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("Risk Level Summary")
    summary = df.groupby('risk_level').agg(
        drivers=('driver_id', 'count'),
        avg_probability=('churn_probability', 'mean'),
        avg_unresolved=('unresolved_count', 'mean'),
        avg_tickets=('total_tickets', 'mean')
    ).round(2)
    summary['drivers_pct'] = (
        summary['drivers'] / len(df) * 100
    ).round(1)
    st.dataframe(summary, use_container_width=True)

# ============================================================
# PAGE 2 — RISK ANALYSIS
# ============================================================

elif page == "Risk Analysis":
    st.title("Risk Analysis")

    risk_filter = st.selectbox(
        "Filter by risk level",
        ["All", "CRITICAL", "HIGH", "MEDIUM", "LOW"]
    )

    if risk_filter == "All":
        filtered_df = df
    else:
        filtered_df = df[df['risk_level'] == risk_filter]

    st.metric("Drivers shown", f"{len(filtered_df):,}")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Unresolved Tickets by Risk Level")
        fig = px.box(
            df,
            x='risk_level',
            y='unresolved_count',
            color='risk_level',
            color_discrete_map={
                'CRITICAL': '#E24B4A',
                'HIGH': '#EF9F27',
                'MEDIUM': '#F5D76E',
                'LOW': '#639922'
            },
            category_orders={
                'risk_level': ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
            }
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Tickets Per Week by Risk Level")
        fig = px.box(
            df,
            x='risk_level',
            y='tickets_per_week',
            color='risk_level',
            color_discrete_map={
                'CRITICAL': '#E24B4A',
                'HIGH': '#EF9F27',
                'MEDIUM': '#F5D76E',
                'LOW': '#639922'
            },
            category_orders={
                'risk_level': ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
            }
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("High Risk Drivers — Action Required")
    action_df = filtered_df[
        filtered_df['risk_level'].isin(['CRITICAL', 'HIGH'])
    ][['driver_id', 'churn_probability', 'risk_level',
       'unresolved_count', 'total_tickets',
       'consecutive_unresolved_end', 'tickets_per_week']
    ].sort_values('churn_probability', ascending=False).head(20)

    action_df['churn_probability'] = (
        action_df['churn_probability'] * 100
    ).round(1).astype(str) + '%'

    st.dataframe(action_df, use_container_width=True)

# ============================================================
# PAGE 3 — FEATURE INSIGHTS
# ============================================================

elif page == "Feature Insights":
    st.title("Feature Insights")

    importance_df = pd.DataFrame({
        'feature': final_features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top 15 Churn Drivers")
        top15 = importance_df.head(15)
        fig = px.bar(
            top15,
            x='importance',
            y='feature',
            orientation='h',
            color='importance',
            color_continuous_scale='Reds'
        )
        fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Unresolved Rate vs Churn Probability")
        sample = df.sample(min(2000, len(df)))
        fig = px.scatter(
            sample,
            x='unresolved_rate',
            y='churn_probability',
            color='risk_level',
            color_discrete_map={
                'CRITICAL': '#E24B4A',
                'HIGH': '#EF9F27',
                'MEDIUM': '#F5D76E',
                'LOW': '#639922'
            },
            opacity=0.6
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("Feature Correlations with Churn")
    correlations = df[final_features + ['will_churn']].corr()[
        'will_churn'
    ].drop('will_churn').sort_values(ascending=False)

    fig = px.bar(
        x=correlations.values,
        y=correlations.index,
        orientation='h',
        color=correlations.values,
        color_continuous_scale='RdBu_r'
    )
    fig.update_layout(
        height=600,
        yaxis={'categoryorder': 'total ascending'}
    )
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# PAGE 4 — BUSINESS IMPACT
# ============================================================

elif page == "Business Impact":
    st.title("Business Impact Calculator")

    st.subheader("Assumptions")
    col1, col2, col3 = st.columns(3)

    with col1:
        monthly_value = st.slider(
            "Monthly value per driver (₹)",
            5000, 50000, 10000, 1000
        )
    with col2:
        intervention_cost = st.slider(
            "Cost per intervention (₹)",
            10, 500, 50, 10
        )
    with col3:
        success_rate = st.slider(
            "Intervention success rate (%)",
            10, 80, 40, 5
        )

    success_rate_decimal = success_rate / 100

    st.divider()

    predicted_churners = df[df['predicted_churn'] == 1]
    tp_estimate = len(predicted_churners) * 0.82
    fp_estimate = len(predicted_churners) * 0.18

    drivers_retained = tp_estimate * success_rate_decimal
    revenue_saved = drivers_retained * monthly_value
    total_cost = len(predicted_churners) * intervention_cost
    net_value = revenue_saved - total_cost
    roi = net_value / total_cost if total_cost > 0 else 0

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Drivers Flagged", f"{len(predicted_churners):,}")
    with col2:
        st.metric("Est. Drivers Retained",
                  f"{drivers_retained:,.0f}")
    with col3:
        st.metric("Revenue Saved",
                  f"₹{revenue_saved:,.0f}/mo")
    with col4:
        st.metric("Net Value",
                  f"₹{net_value:,.0f}/mo",
                  f"ROI: {roi:.1f}x")

    st.divider()

    st.subheader("ROI by Threshold")
    thresholds = np.arange(0.10, 0.60, 0.05)
    roi_data = []

    for t in thresholds:
        flagged = (df['churn_probability'] >= t).sum()
        tp_est = flagged * 0.82
        retained = tp_est * success_rate_decimal
        revenue = retained * monthly_value
        cost = flagged * intervention_cost
        net = revenue - cost
        roi_t = net / cost if cost > 0 else 0
        roi_data.append({
            'threshold': round(t, 2),
            'flagged': flagged,
            'net_value': net,
            'roi': roi_t
        })

    roi_df = pd.DataFrame(roi_data)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.line(
            roi_df, x='threshold', y='net_value',
            title='Net Value by Threshold',
            markers=True
        )
        fig.add_vline(
            x=threshold,
            line_dash="dash",
            annotation_text="Current"
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.line(
            roi_df, x='threshold', y='roi',
            title='ROI by Threshold',
            markers=True
        )
        fig.add_vline(
            x=threshold,
            line_dash="dash",
            annotation_text="Current"
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)