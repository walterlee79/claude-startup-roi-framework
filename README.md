# claude-startup-roi-framework

This is an interactive ROI prediction framework designed for Anthropic's Startup Ecosystem team. It empowers data-driven decisions by quantifying which startups will generate the most value from Claude API credits.

🔗 Live Demo link : https://claude-startup-roi-framework-rclvgwxgynst2pepcm5ywm.streamlit.app/

A screenshot of the live dashboard is highly recommended here.

🎯 Project Goal & Background
A key challenge for Anthropic's startup program is allocating limited Claude API credits to maximize value across the entire portfolio. This requires moving beyond qualitative assessments to a quantitative framework that can forecast the potential impact of a credit grant.

I proactively built this project to address that specific challenge. It's a tangible demonstration of how I would approach the problems and responsibilities of the Startup Platform Operations and Account Manager roles at Anthropic.

To ensure real-world accuracy, the framework's entire calculation engine is validated against the Polymerize case study—a B2B SaaS startup that achieved a remarkable 1,532% probability-weighted ROI by implementing Claude.

✨ Key Features
📊 Interactive Streamlit Dashboard: Allows for real-time parameter adjustments to intuitively simulate ROI outcomes on the web.

⚖️ Probability-Weighted Scenario Analysis: Forecasts ROI based on Conservative, Expected, and Upside scenarios to support realistic, risk-adjusted decision-making.

📈 ROI Curve & Adoption Modeling: Visualizes the 12-month cumulative value against costs and models technology adoption with an S-Curve to clearly identify the payback period.

🎯 Sensitivity Analysis Heatmaps: Generates heatmaps to analyze how ROI is impacted by key variables like API Credit Amount, Automation Potential, and Revenue Attribution, identifying the most critical levers for success.

📋 Detailed Monthly Breakdown: Provides a transparent, month-by-month breakdown of value components—including Labor Savings, Error Reduction, and Revenue Impact—to substantiate the final ROI calculation.

🛠️ Tech Stack
Language: Python

Framework: Streamlit

Libraries: Pandas, NumPy, Plotly

Deployment: Streamlit Community Cloud


1. Clone the Repository
   
git clone https://github.com/walterlee79/claude-startup-roi-framework.git
cd claude-startup-roi-framework

2. Install Dependencies

pip install -r requirements.txt

3. Run the Streamlit App
This command will launch the web application on your local machine.

Bash

streamlit run app.py
You can then access the dashboard by navigating to the localhost address provided in your terminal.

📁 Project Structure
.
├── .streamlit/
│   └── config.toml      # Streamlit theme and server configuration
├── src/
│   └── roi_engine.py    # Core ROI calculation engine
├── app.py               # Main Streamlit dashboard application
├── README.md            # Project overview and documentation
└── requirements.txt     # Python dependencies

📄 License

This project is licensed under the MIT License. See the LICENSE file for more details.


