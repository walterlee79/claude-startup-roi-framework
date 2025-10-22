# Claude Startup ROI Framework  
_Data-driven credit allocation for Anthropic’s Startup Ecosystem_

<img width="1893" height="827" alt="Screen Shot 2025-10-22 at 1 46 35 PM" src="https://github.com/user-attachments/assets/88bc881e-58b0-46f7-8df6-1415129458f9" />

---

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-blue)](https://claude-startup-roi-framework-rclvgwxgynst2pepcm5ywm.streamlit.app/) ![License](https://img.shields.io/badge/License-MIT-green)

Why this exists (for Anthropic): Anthropic’s startup program needs to allocate limited Claude API credits** to maximize portfolio-level value**. This framework shifts from qualitative assessments to a quantitative model that quickly estimates and compares which teams convert credits into value (ROI) most efficiently.

> Live Demo: click the badge above or click the link as below
> https://claude-startup-roi-framework-rclvgwxgynst2pepcm5ywm.streamlit.app/ 

---

## 🎯 Project Goal & Background

An interactive ROI prediction framework that helps the Anthropic Startup Ecosystem team make more scientific credit allocation decisions.
This project demonstrates how the Startup Platform Operations / Account Manager role would approach the problem with a concrete deliverable.

Calibration note: The calculation engine was sanity-checked / calibrated using a B2B SaaS (Polymerize) case with real metrics/assumptions.
An illustrative result of 1,532% probability-weighted ROI was produced, but it is assumption-sensitive and presented for illustration.

## 🧮 How it works (short)

ROI = (Expected Benefit − Total Cost) ÷ Total Cost

Expected Benefit = Σᵢ [ pᵢ × benefitᵢ ] for scenarios i ∈ {Conservative, Expected, Upside} with Σ pᵢ = 1

Costs = One-time implementation + monthly run-rate (credits/headcount/tools)

Adoption = S-Curve to reflect ramp-up and onboarding periods

Outputs = Payback month, PW-ROI%, 12M cumulative value curve, sensitivity heatmaps, monthly breakdown

## 🖥️ Usage (at a glance)

In the web dashboard, adjust API credits, automation rate, revenue attribution, error cost, hourly headcount rates, etc., and observe results live.

Use the “Most sensitive levers” section to prioritize which variables to optimize first.

**📎 Example Questions it Answers
**
Which startup is most likely to create greater value/ROI with the same credits?

What is the payback month, and what is the 12-month cumulative value?

Among credit size / automation rate / revenue attribution, which lever is most sensitive?


## 🧪 Roadmap

CSV/PNG Export buttons

Side-by-side What-if comparisons

Portfolio view (compare N startups simultaneously)

Monte Carlo option (assumption distributions)

PRs welcome—everything from small README tweaks to new features.

## ❓ FAQ

Q. Is the Polymerize 1,532% figure guaranteed?
A. No. It is an illustrative example based on public assumptions. Results vary by team/domain/workflow. The model supports assumption sensitivity analysis.

Q. How do I read the sensitivity heatmap?
A. Darker areas indicate larger ROI swings. Use it to decide which levers to tune first.

Q. Can I use private data?
A. Yes for local runs. Before sharing, please remove/obfuscate sensitive details.

## 📄 License

MIT License — see LICENSE.


---


## ✨ What it does
- **📊 Interactive Streamlit Dashboard** — Adjust parameters and see **real-time** ROI simulations  
- **⚖️ Probability-Weighted Scenarios** — **Conservative / Expected / Upside** with probability-weighted ROI  
- **📈 Adoption & Payback** — 12-month cumulative value vs. cost, **S-Curve adoption** to visualize payback timing  
- **🎯 Sensitivity Heatmaps** — Analyze levers like API credits, automation potential, and revenue attribution  
- **📋 Monthly Roll-up** — Transparent month-by-month breakdown of labor savings, error reduction, and revenue impact  
- **🧰 Tech** — Python · Streamlit · Pandas · NumPy · Plotly (Deploy: Streamlit Community Cloud)

---

## 🚀 Quickstart

```bash
# 1) Clone
git clone https://github.com/walterlee79/claude-startup-roi-framework.git
cd claude-startup-roi-framework

# 2) Install
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt

# 3) Run (local)
streamlit run app.py
# → Open the localhost address printed in your terminal

---

🗂️ Project Structure
.
├── .streamlit/
│   └── config.toml           # Streamlit theme/server config
├── src/
│   └── roi_engine.py         # Core ROI calculation engine
├── app.py                    # Main Streamlit app
├── README.md                 # Docs
└── requirements.txt          # Dependencies
