"""
Claude Startup ROI Predictor - Interactive Dashboard
====================================================

Interactive web application for analyzing Claude API credit ROI for startups.

Built for Anthropic's Startup Platform Operations team to evaluate
which startups will generate maximum value from Claude credits.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Add src to path for Streamlit Cloud
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from roi_engine import (
    ROICalculator, UseCaseConfig, CostConfig, ScenarioConfig,
    DEFAULT_SCENARIOS, AdoptionCurve, run_roi_analysis
)

# Page config
st.set_page_config(
    page_title="Claude Startup ROI Predictor",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E86AB;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .anthropic-note {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🚀 Claude Startup ROI Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">A Framework for Evaluating Claude API Credit Impact</div>', unsafe_allow_html=True)

st.markdown("""
<div class="anthropic-note">
    <strong>Built for Anthropic's Startup Ecosystem Team</strong><br>
    This tool helps identify which startups will generate the highest ROI from Claude credits,
    enabling data-driven decisions for the Startup Platform Operations and Account Manager roles.
    <br><br>
    <em>Created by Hanjong Lee - Portfolio Project for Anthropic Application</em>
</div>
""", unsafe_allow_html=True)

# Sidebar - Input Parameters
st.sidebar.header("📊 Configuration")

tab_config = st.sidebar.radio("Configuration Type", ["Quick Presets", "Custom Parameters"])

if tab_config == "Quick Presets":
    preset = st.sidebar.selectbox(
        "Select Startup Profile",
        [
            "Customer Support Automation (Series A)",
            "Sales Proposal Generation (Series B)",
            "Code Documentation (Seed)",
            "Technical Support (Series A)",
            "Custom..."
        ]
    )
    
    if preset == "Customer Support Automation (Series A)":
        use_case_config = {
            "annual_tasks": 50000,
            "minutes_saved_per_task": 2.0,
            "hourly_rate_fully_loaded": 65.0,
            "automation_potential": 0.60,
            "error_rate_before": 0.05,
            "error_rate_after": 0.01,
            "cost_per_error": 25.0,
            "revenue_units_per_year": 500,
            "conversion_rate": 0.20,
            "avg_deal_value": 50000.0,
            "margin": 0.30,
            "attribution_factor": 0.10
        }
        cost_config = {
            "api_credit_usd": 10000.0,
            "implementation_usd": 20000.0,
            "monthly_overhead_usd": 0.0
        }
    elif preset == "Sales Proposal Generation (Series B)":
        use_case_config = {
            "annual_tasks": 1200,
            "minutes_saved_per_task": 30.0,
            "hourly_rate_fully_loaded": 75.0,
            "automation_potential": 0.70,
            "error_rate_before": 0.10,
            "error_rate_after": 0.02,
            "cost_per_error": 500.0,
            "revenue_units_per_year": 1200,
            "conversion_rate": 0.20,
            "avg_deal_value": 50000.0,
            "margin": 0.20,
            "attribution_factor": 0.50
        }
        cost_config = {
            "api_credit_usd": 10000.0,
            "implementation_usd": 25000.0,
            "monthly_overhead_usd": 0.0
        }
    elif preset == "Code Documentation (Seed)":
        use_case_config = {
            "annual_tasks": 20000,
            "minutes_saved_per_task": 5.0,
            "hourly_rate_fully_loaded": 80.0,
            "automation_potential": 0.75,
            "error_rate_before": 0.15,
            "error_rate_after": 0.03,
            "cost_per_error": 200.0,
            "revenue_units_per_year": 0,
            "conversion_rate": 0.0,
            "avg_deal_value": 0.0,
            "margin": 0.0,
            "attribution_factor": 0.0
        }
        cost_config = {
            "api_credit_usd": 8000.0,
            "implementation_usd": 15000.0,
            "monthly_overhead_usd": 0.0
        }
    elif preset == "Technical Support (Series A)":
        use_case_config = {
            "annual_tasks": 54000,
            "minutes_saved_per_task": 4.0,
            "hourly_rate_fully_loaded": 70.0,
            "automation_potential": 0.65,
            "error_rate_before": 0.08,
            "error_rate_after": 0.02,
            "cost_per_error": 150.0,
            "revenue_units_per_year": 1200,
            "conversion_rate": 0.25,
            "avg_deal_value": 150000.0,
            "margin": 0.35,
            "attribution_factor": 0.15
        }
        cost_config = {
            "api_credit_usd": 10000.0,
            "implementation_usd": 25000.0,
            "monthly_overhead_usd": 0.0
        }
    else:
        tab_config = "Custom Parameters"

if tab_config == "Custom Parameters":
    st.sidebar.subheader("💰 Cost Structure")
    credit_usd = st.sidebar.number_input(
        "Claude API Credit ($)", 
        min_value=1000, 
        max_value=50000, 
        value=10000, 
        step=1000
    )
    implementation_usd = st.sidebar.number_input(
        "Implementation Cost ($)", 
        min_value=5000, 
        max_value=100000, 
        value=25000, 
        step=5000
    )
    
    st.sidebar.subheader("⚙️ Use Case Parameters")
    annual_tasks = st.sidebar.number_input(
        "Annual Tasks", 
        min_value=100, 
        max_value=500000, 
        value=54000, 
        step=1000
    )
    minutes_saved = st.sidebar.slider(
        "Minutes Saved per Task", 
        min_value=0.5, 
        max_value=60.0, 
        value=4.0, 
        step=0.5
    )
    hourly_rate = st.sidebar.number_input(
        "Fully-Loaded Hourly Rate ($)", 
        min_value=20, 
        max_value=200, 
        value=70, 
        step=5
    )
    
    automation_potential = st.sidebar.slider(
        "Automation Potential", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.65, 
        step=0.05,
        format="%.0f%%",
        help="Percentage of tasks that can be automated"
    ) 
    
    attribution_factor = st.sidebar.slider(
        "Revenue Attribution Factor", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.15, 
        step=0.05,
        format="%.0f%%",
        help="Percentage of revenue impact attributable to Claude"
    )
    
    # Advanced options in expander
    with st.sidebar.expander("Advanced Parameters"):
        error_rate_before = st.slider("Error Rate Before", 0.0, 0.30, 0.08, 0.01, format="%.0f%%")
        error_rate_after = st.slider("Error Rate After", 0.0, 0.10, 0.02, 0.01, format="%.0f%%")
        cost_per_error = st.number_input("Cost per Error ($)", 0, 1000, 150, 10)
        revenue_units = st.number_input("Revenue Units/Year", 0, 10000, 1200, 100)
        conversion_rate = st.slider("Conversion Rate", 0.0, 1.0, 0.25, 0.05, format="%.0f%%")
        avg_deal_value = st.number_input("Avg Deal Value ($)", 0, 1000000, 150000, 10000)
        margin = st.slider("Margin", 0.0, 1.0, 0.35, 0.05, format="%.0f%%")
    
    use_case_config = {
        "annual_tasks": int(annual_tasks),
        "minutes_saved_per_task": float(minutes_saved),
        "hourly_rate_fully_loaded": float(hourly_rate),
        "automation_potential": float(automation_potential),
        "error_rate_before": float(error_rate_before),
        "error_rate_after": float(error_rate_after),
        "cost_per_error": float(cost_per_error),
        "revenue_units_per_year": int(revenue_units),
        "conversion_rate": float(conversion_rate),
        "avg_deal_value": float(avg_deal_value),
        "margin": float(margin),
        "attribution_factor": float(attribution_factor)
    }
    
    cost_config = {
        "api_credit_usd": float(credit_usd),
        "implementation_usd": float(implementation_usd),
        "monthly_overhead_usd": 0.0
    }

# Run ROI Analysis
config = {
    "use_case": use_case_config,
    "costs": cost_config
}

results = run_roi_analysis(config, DEFAULT_SCENARIOS)

# Display Results
st.header("📈 ROI Analysis Results")

# Key Metrics Row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Probability-Weighted ROI",
        f"{results['probability_weighted_roi_pct']:.1f}%",
        delta="Expected Return"
    )

with col2:
    total_investment = cost_config['api_credit_usd'] + cost_config['implementation_usd']
    st.metric(
        "Total Investment",
        f"${total_investment:,.0f}",
        delta=None
    )

with col3:
    expected_scenario = next(s for s in results['scenarios'] if s['scenario'] == 'Expected')
    st.metric(
        "Expected 12M Value",
        f"${expected_scenario['total_benefit']:,.0f}",
        delta=f"{expected_scenario['roi_pct']:.0f}% ROI"
    )

with col4:
    payback = expected_scenario['payback_month']
    st.metric(
        "Payback Period",
        f"{payback} months" if payback else "Not achieved",
        delta="Expected Scenario"
    )

# Tabs for different visualizations
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Scenario Comparison", 
    "📈 ROI Curve", 
    "🎯 Sensitivity Analysis",
    "📋 Detailed Breakdown"
])

with tab1:
    st.subheader("Scenario Analysis")
    
    # Scenario comparison table
    scenario_df = pd.DataFrame([
        {
            "Scenario": s['scenario'],
            "Probability": f"{s['probability']*100:.0f}%",
            "ROI": f"{s['roi_pct']:.1f}%",
            "Total Benefit": f"${s['total_benefit']:,.0f}",
            "Payback (months)": s['payback_month'] if s['payback_month'] else "N/A",
            "Avg Adoption": f"{s['avg_adoption']*100:.0f}%"
        }
        for s in results['scenarios']
    ])
    
    st.dataframe(scenario_df, use_container_width=True, hide_index=True)
    
    # ROI comparison bar chart
    fig_scenarios = go.Figure()
    
    scenarios_data = results['scenarios']
    colors = ['#A23B72', '#2E86AB', '#06A77D']
    
    fig_scenarios.add_trace(go.Bar(
        x=[s['scenario'] for s in scenarios_data],
        y=[s['roi_pct'] for s in scenarios_data],
        marker_color=colors,
        text=[f"{s['roi_pct']:.1f}%" for s in scenarios_data],
        textposition='outside'
    ))
    
    fig_scenarios.update_layout(
        title="ROI by Scenario",
        xaxis_title="Scenario",
        yaxis_title="ROI (%)",
        height=400
    )
    
    st.plotly_chart(fig_scenarios, use_container_width=True)

with tab2:
    st.subheader("12-Month ROI Projection")
    
    # ROI curve for expected scenario
    expected_data = expected_scenario['rows']
    months = [row['month'] for row in expected_data]
    cumulative_values = [row['cumulative_value'] for row in expected_data]
    
    fig_curve = go.Figure()
    
    # Cumulative value line
    fig_curve.add_trace(go.Scatter(
        x=months,
        y=cumulative_values,
        mode='lines+markers',
        name='Cumulative Value',
        line=dict(color='#2E86AB', width=3),
        marker=dict(size=8)
    ))
    
    # Total cost line
    fig_curve.add_trace(go.Scatter(
        x=months,
        y=[expected_scenario['total_costs']] * len(months),
        mode='lines',
        name='Total Cost',
        line=dict(color='#A23B72', width=2, dash='dash')
    ))
    
    fig_curve.update_layout(
        title=f"ROI Curve - {expected_scenario['scenario']} Scenario",
        xaxis_title="Month",
        yaxis_title="Cumulative Value ($)",
        hovermode='x unified',
        height=500
    )
    
    st.plotly_chart(fig_curve, use_container_width=True)
    
    # Adoption curve
    adoption_rates = [row['adoption_rate'] * 100 for row in expected_data]
    
    fig_adoption = go.Figure()
    fig_adoption.add_trace(go.Scatter(
        x=months,
        y=adoption_rates,
        mode='lines+markers',
        fill='tozeroy',
        line=dict(color='#06A77D', width=2),
        marker=dict(size=6)
    ))
    
    fig_adoption.update_layout(
        title="Adoption Rate Over Time",
        xaxis_title="Month",
        yaxis_title="Adoption Rate (%)",
        height=400
    )
    
    st.plotly_chart(fig_adoption, use_container_width=True)

with tab3:
    st.subheader("Sensitivity Analysis")
    
    st.markdown("""
    Analyze how ROI changes with different credit costs, automation potential, 
    and attribution factors.
    """)
    
    # Sensitivity parameters
    col1, col2 = st.columns(2)
    
    with col1:
        credit_range = st.multiselect(
            "Credit Amounts to Test ($)",
            [5000, 8000, 10000, 12000, 15000, 20000],
            default=[8000, 10000, 12000, 15000, 20000]
        )
    
    with col2:
        test_scenario = st.selectbox(
            "Test Scenario",
            ["Conservative", "Expected", "Upside"],
            index=1
        )
    
    if st.button("Run Sensitivity Analysis", type="primary"):
        with st.spinner("Calculating sensitivity..."):
            from roi_engine import SensitivityAnalyzer
            
            test_scenario_config = next(s for s in DEFAULT_SCENARIOS if s.name == test_scenario)
            
            # Create sensitivity analyzer
            analyzer = SensitivityAnalyzer(config)
            
            # Define ranges
            automation_range = np.arange(0.45, 0.81, 0.05)
            attribution_range = np.arange(0.10, 0.31, 0.05)
            
            # Run analysis
            sensitivity_results = analyzer.analyze_credit_sensitivity(
                credit_range,
                automation_range,
                attribution_range,
                test_scenario_config
            )
            
            # Display heatmaps
            for result in sensitivity_results:
                credit = result['credit']
                grid_data = result['grid']
                
                # Pivot data for heatmap
                df_grid = pd.DataFrame(grid_data)
                pivot = df_grid.pivot(
                    index='attribution',
                    columns='automation',
                    values='roi'
                )
                
                fig_heatmap = go.Figure(data=go.Heatmap(
                    z=pivot.values,
                    x=[f"{int(a*100)}%" for a in pivot.columns],
                    y=[f"{int(a*100)}%" for a in pivot.index],
                    colorscale='RdYlGn',
                    colorbar=dict(title="ROI (%)"),
                    text=pivot.values.round(0),
                    texttemplate='%{text:.0f}%',
                    textfont={"size": 10}
                ))
                
                fig_heatmap.update_layout(
                    title=f"ROI Heatmap - Credit ${credit:,}",
                    xaxis_title="Automation Potential",
                    yaxis_title="Attribution Factor",
                    height=500
                )
                
                st.plotly_chart(fig_heatmap, use_container_width=True)
                
                # Find minimum viable parameters
                feasible = df_grid[df_grid['roi'] >= 0].sort_values(['automation', 'attribution'])
                if not feasible.empty:
                    min_auto = feasible['automation'].min()
                    min_attr = feasible[feasible['automation'] == min_auto]['attribution'].min()
                    st.success(
                        f"✅ For ${credit:,} credit: Minimum ROI ≥ 0% achieved with "
                        f"Automation ≥ {int(min_auto*100)}% and Attribution ≥ {int(min_attr*100)}%"
                    )
                else:
                    st.warning(f"⚠️ ROI ≥ 0% not achieved within tested parameters for ${credit:,} credit")

with tab4:
    st.subheader("Detailed Monthly Breakdown")
    
    # Select scenario to view
    selected_scenario = st.selectbox(
        "Select Scenario for Details",
        [s['scenario'] for s in results['scenarios']]
    )
    
    scenario_detail = next(s for s in results['scenarios'] if s['scenario'] == selected_scenario)
    
    # Monthly breakdown table
    monthly_df = pd.DataFrame(scenario_detail['rows'])
    monthly_df['adoption_rate'] = (monthly_df['adoption_rate'] * 100).round(1)
    monthly_df['monthly_value'] = monthly_df['monthly_value'].round(0)
    monthly_df['cumulative_value'] = monthly_df['cumulative_value'].round(0)
    monthly_df['labor_savings'] = monthly_df['labor_savings'].round(0)
    monthly_df['error_savings'] = monthly_df['error_savings'].round(0)
    monthly_df['revenue_impact'] = monthly_df['revenue_impact'].round(0)
    
    monthly_df.columns = [
        'Month', 'Adoption %', 'Monthly Value', 'Cumulative Value',
        'Labor Savings', 'Error Savings', 'Revenue Impact'
    ]
    
    st.dataframe(
        monthly_df.style.format({
            'Adoption %': '{:.1f}%',
            'Monthly Value': '${:,.0f}',
            'Cumulative Value': '${:,.0f}',
            'Labor Savings': '${:,.0f}',
            'Error Savings': '${:,.0f}',
            'Revenue Impact': '${:,.0f}'
        }),
        use_container_width=True,
        hide_index=True
    )
    
    # Value breakdown chart
    fig_breakdown = go.Figure()
    
    fig_breakdown.add_trace(go.Bar(
        x=monthly_df['Month'],
        y=monthly_df['Labor Savings'],
        name='Labor Savings',
        marker_color='#2E86AB'
    ))
    
    fig_breakdown.add_trace(go.Bar(
        x=monthly_df['Month'],
        y=monthly_df['Error Savings'],
        name='Error Savings',
        marker_color='#06A77D'
    ))
    
    fig_breakdown.add_trace(go.Bar(
        x=monthly_df['Month'],
        y=monthly_df['Revenue Impact'],
        name='Revenue Impact',
        marker_color='#F18F01'
    ))
    
    fig_breakdown.update_layout(
        title="Monthly Value Breakdown by Component",
        xaxis_title="Month",
        yaxis_title="Value ($)",
        barmode='stack',
        height=500
    )
    
    st.plotly_chart(fig_breakdown, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
### 🎯 Why This Framework Matters for Anthropic

This tool demonstrates:
- **Data-Driven Credit Allocation**: Quantify ROI before deploying credits
- **Startup Readiness Assessment**: Identify which startups are ready for Claude adoption
- **Portfolio Optimization**: Maximize value across Anthropic's startup ecosystem
- **Evidence-Based Expansion**: Build case studies showing Claude's B2B SaaS impact

**Built by**: Hanjong Lee | **For**: Anthropic Startup Platform Operations  
**Contact**: hanjong.lee@polymerize.io | **GitHub**: [View Repository](#)

*This framework is based on real Polymerize ROI analysis, demonstrating 1532% probability-weighted ROI 
from Claude implementation in B2B SaaS environments.*
""")
