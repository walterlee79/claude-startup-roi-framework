import json
from pathlib import Path
import sys
import streamlit as st

# src 경로 추가
sys.path.append(str(Path(__file__).parent / "src"))
from roi_engine import run as roi_run  # noqa: E402

st.set_page_config(page_title="Claude Startup ROI", layout="wide")
st.title("Claude Startup ROI Framework")

# 기본 입력값
default_cfg = {
    "use_case": {
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
    },
    "costs": {
        "api_credit_usd": 10000.0,
        "implementation_usd": 25000.0,
        "monthly_overhead_usd": 0.0
    }
}

with st.sidebar:
    st.header("Inputs")
    uc = default_cfg["use_case"]
    co = default_cfg["costs"]
    uc["annual_tasks"] = st.number_input("Annual tasks", 1000, 200000, uc["annual_tasks"], step=1000)
    uc["minutes_saved_per_task"] = st.number_input("Minutes saved / task", 0.0, 120.0, uc["minutes_saved_per_task"], step=0.5)
    uc["hourly_rate_fully_loaded"] = st.number_input("Hourly rate ($)", 0.0, 500.0, uc["hourly_rate_fully_loaded"], step=5.0)
    uc["automation_potential"] = st.slider("Automation potential", 0.0, 1.0, uc["automation_potential"], 0.01)
    uc["error_rate_before"] = st.slider("Error rate (before)", 0.0, 0.5, uc["error_rate_before"], 0.01)
    uc["error_rate_after"] = st.slider("Error rate (after)", 0.0, 0.5, uc["error_rate_after"], 0.01)
    uc["cost_per_error"] = st.number_input("Cost per error ($)", 0.0, 10000.0, uc["cost_per_error"], step=50.0)
    uc["revenue_units_per_year"] = st.number_input("Revenue units / year", 0, 100000, uc["revenue_units_per_year"], step=100)
    uc["conversion_rate"] = st.slider("Conversion rate", 0.0, 1.0, uc["conversion_rate"], 0.01)
    uc["avg_deal_value"] = st.number_input("Avg deal value ($)", 0.0, 1000000.0, uc["avg_deal_value"], step=1000.0)
    uc["margin"] = st.slider("Margin", 0.0, 1.0, uc["margin"], 0.01)
    uc["attribution_factor"] = st.slider("Attribution factor", 0.0, 1.0, uc["attribution_factor"], 0.01)

    co["api_credit_usd"] = st.number_input("Claude credit ($)", 0.0, 1000000.0, co["api_credit_usd"], step=1000.0)
    co["implementation_usd"] = st.number_input("Implementation ($)", 0.0, 1000000.0, co["implementation_usd"], step=1000.0)
    co["monthly_overhead_usd"] = st.number_input("Monthly overhead ($)", 0.0, 100000.0, co["monthly_overhead_usd"], step=100.0)

    run_btn = st.button("Compute ROI", use_container_width=True)

if run_btn:
    result = roi_run(default_cfg)
    st.subheader(f"Probability-weighted ROI: {result['probability_weighted_roi_pct']:.1f}%")
    for s in result["scenarios"]:
        st.write(f"**{s['scenario']}** — ROI: {s['roi_pct']:.1f}% | Payback: {s['payback_month'] or 'N/A'} | Benefit: ${s['total_benefit']:,.0f}")
else:
    st.info("좌측 사이드바 값 조정 후 **Compute ROI** 버튼을 눌러 결과를 확인하세요.")
