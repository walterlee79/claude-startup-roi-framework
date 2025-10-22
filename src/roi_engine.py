"""
Claude Credit ROI Prediction Engine
====================================

A comprehensive framework for predicting ROI of Claude API credits for 
startups.
Designed for Anthropic's Startup Platform Operations team.

Author: Hanjong Lee
Purpose: Portfolio project for Anthropic application
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict


@dataclass
class UseCaseConfig:
    """Configuration for startup use case parameters"""
    annual_tasks: int
    minutes_saved_per_task: float
    hourly_rate_fully_loaded: float
    automation_potential: float  # 0.0 to 1.0
    error_rate_before: float
    error_rate_after: float
    cost_per_error: float
    revenue_units_per_year: int
    conversion_rate: float
    avg_deal_value: float
    margin: float
    attribution_factor: float  # 0.0 to 1.0


@dataclass
class CostConfig:
    """Cost structure for Claude implementation"""
    api_credit_usd: float
    implementation_usd: float
    monthly_overhead_usd: float = 0.0


@dataclass
class ScenarioConfig:
    """Adoption scenario configuration"""
    name: str
    probability: float
    max_adoption: float  # K_max in logistic curve
    slope_param: float   # b in logistic curve
    inflection_month: float  # t0 in logistic curve


class AdoptionCurve:
    """S-curve adoption model using logistic function"""
    
    @staticmethod
    def calculate_adoption(month: int, k_max: float, b: float, t0: 
float) -> float:
        """Calculate adoption rate using logistic curve"""
        
        Formula: Adoption(t) = K_max / (1 + e^(-b*(t - t0)))
        
        Args:
            month: Current month (1-12)
            k_max: Maximum adoption rate (0.50-0.85)
            b: Slope parameter (0.25-0.50)
            t0: Inflection point in months (5-7)
        """
        return k_max / (1 + np.exp(-b * (month - t0)))
    
    @staticmethod
    def get_monthly_adoptions(months: int, scenario: ScenarioConfig) -> 
np.ndarray:
        """Get adoption rates for each month"""
        return np.array([
            AdoptionCurve.calculate_adoption(
                m, 
                scenario.max_adoption, 
                scenario.slope_param, 
                scenario.inflection_month
            ) for m in range(1, months + 1)
        ])


class ROICalculator:
    """Main ROI calculation engine"""
    
    def __init__(self, use_case: UseCaseConfig, costs: CostConfig):
        self.use_case = use_case
        self.costs = costs
        
    def calculate_monthly_value(self, adoption_rate: float) -> Dict[str, 
float]:
        """Calculate value generated in a single month"""
        # Labor cost savings
        tasks_automated = (
            self.use_case.annual_tasks / 12 * 
            adoption_rate * 
            self.use_case.automation_potential
        )
        
        labor_savings = (
            tasks_automated * 
            (self.use_case.minutes_saved_per_task / 60) * 
            self.use_case.hourly_rate_fully_loaded
        )
        
        # Error reduction value
        error_reduction = self.use_case.error_rate_before - 
self.use_case.error_rate_after
        error_savings = tasks_automated * error_reduction * 
self.use_case.cost_per_error
        
        # Revenue impact (for revenue-generating use cases)
        monthly_units = self.use_case.revenue_units_per_year / 12
        productivity_boost = adoption_rate * 
self.use_case.automation_potential
        
        additional_units = monthly_units * productivity_boost
        revenue_impact = (
            additional_units * 
            self.use_case.conversion_rate * 
            self.use_case.avg_deal_value * 
            self.use_case.margin * 
            self.use_case.attribution_factor
        )
        
        return {
            "labor_savings": labor_savings,
            "error_savings": error_savings,
            "revenue_impact": revenue_impact,
            "total_value": labor_savings + error_savings + revenue_impact
        }
    
    def calculate_scenario_roi(
        self, 
        scenario: ScenarioConfig, 
        months: int = 12
    ) -> Dict:
        """Calculate ROI for a specific adoption scenario"""
        
        # Get monthly adoption rates
        adoption_rates = AdoptionCurve.get_monthly_adoptions(months, 
scenario)
        
        # Calculate monthly values
        monthly_data = []
        cumulative_value = 0
        total_costs = (
            self.costs.api_credit_usd + 
            self.costs.implementation_usd + 
            self.costs.monthly_overhead_usd * months
        )
        
        payback_month = None
        
        for month in range(1, months + 1):
            adoption = adoption_rates[month - 1]
            monthly_calc = self.calculate_monthly_value(adoption)
            cumulative_value += monthly_calc["total_value"]
            
            # Check for payback
            if payback_month is None and cumulative_value >= total_costs:
                payback_month = month
            
            monthly_data.append({
                "month": month,
                "adoption_rate": float(adoption),
                "monthly_value": monthly_calc["total_value"],
                "cumulative_value": cumulative_value,
                "labor_savings": monthly_calc["labor_savings"],
                "error_savings": monthly_calc["error_savings"],
                "revenue_impact": monthly_calc["revenue_impact"]
            })
        
        total_benefit = cumulative_value
        roi_pct = ((total_benefit - total_costs) / total_costs) * 100
        
        return {
            "scenario": scenario.name,
            "probability": scenario.probability,
            "total_costs": total_costs,
            "total_benefit": total_benefit,
            "roi_pct": roi_pct,
            "payback_month": payback_month,
            "rows": monthly_data,
            "avg_adoption": float(np.mean(adoption_rates))
        }
    
    def calculate_probability_weighted_roi(
        self, 
        scenarios: List[ScenarioConfig]
    ) -> Dict:
        """Calculate probability-weighted ROI across multiple 
scenarios"""
        
        scenario_results = [
            self.calculate_scenario_roi(scenario) 
            for scenario in scenarios
        ]
        
        # Probability-weighted ROI
        weighted_roi = sum(
            result["roi_pct"] * result["probability"]
            for result in scenario_results
        )
        
        return {
            "probability_weighted_roi_pct": weighted_roi,
            "scenarios": scenario_results,
            "inputs": {
                "use_case": asdict(self.use_case),
                "costs": asdict(self.costs)
            }
        }


class SensitivityAnalyzer:
    """Sensitivity analysis for ROI parameters"""
    
    def __init__(self, base_config: Dict):
        self.base_config = base_config
    
    def analyze_credit_sensitivity(
        self,
        credit_range: List[float],
        automation_range: np.ndarray,
        attribution_range: np.ndarray,
        scenario: ScenarioConfig
    ) -> List[Dict]:
        """
        Analyze ROI sensitivity to credit cost and key parameters
        
        Returns list of results for different credit levels
        """
        results = []
        
        for credit in credit_range:
            grid_results = []
            
            for automation in automation_range:
                for attribution in attribution_range:
                    # Create modified config
                    use_case = 
UseCaseConfig(**self.base_config["use_case"])
                    use_case.automation_potential = float(automation)
                    use_case.attribution_factor = float(attribution)
                    
                    costs = CostConfig(**self.base_config["costs"])
                    costs.api_credit_usd = float(credit)
                    
                    # Calculate ROI
                    calculator = ROICalculator(use_case, costs)
                    result = calculator.calculate_scenario_roi(scenario)
                    
                    grid_results.append({
                        "automation": float(automation),
                        "attribution": float(attribution),
                        "roi": result["roi_pct"]
                    })
            
            results.append({
                "credit": credit,
                "grid": grid_results
            })
        
        return results


# Default scenario configurations
DEFAULT_SCENARIOS = [
    ScenarioConfig(
        name="Conservative",
        probability=0.30,
        max_adoption=0.50,
        slope_param=0.25,
        inflection_month=7.0
    ),
    ScenarioConfig(
        name="Expected",
        probability=0.50,
        max_adoption=0.70,
        slope_param=0.35,
        inflection_month=6.0
    ),
    ScenarioConfig(
        name="Upside",
        probability=0.20,
        max_adoption=0.85,
        slope_param=0.50,
        inflection_month=5.0
    )
]


def load_config_from_json(json_path: str) -> Dict:
    """Load configuration from JSON file"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def run_roi_analysis(config: Dict, scenarios: 
Optional[List[ScenarioConfig]] = None) -> Dict:
    """
    Main entry point for ROI analysis
    
    Args:
        config: Configuration dictionary with 'use_case' and 'costs' keys
        scenarios: Optional list of scenario configs (defaults to 
DEFAULT_SCENARIOS)
    
    Returns:
        Dictionary with probability-weighted ROI and scenario details
    """
    if scenarios is None:
        scenarios = DEFAULT_SCENARIOS
    
    use_case = UseCaseConfig(**config["use_case"])
    costs = CostConfig(**config["costs"])
    
    calculator = ROICalculator(use_case, costs)
    return calculator.calculate_probability_weighted_roi(scenarios)


if __name__ == "__main__":
    # Example usage
    example_config = {
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
    
    results = run_roi_analysis(example_config)
    print(f"Probability-Weighted ROI: 
{results['probability_weighted_roi_pct']:.1f}%")# 
---- 
Safety 
helpers & compat 
entrypoint 
-------------------------------------
def _clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, float(x)))

def _validate_and_fix(config: Dict, scenarios: List["ScenarioConfig"]) -> 
None:
    uc = config["use_case"]
    # clamp 0..1 params
    uc["automation_potential"] = _clamp(uc["automation_potential"])
    uc["attribution_factor"]   = _clamp(uc["attribution_factor"])
    uc["conversion_rate"]      = _clamp(uc["conversion_rate"])
    uc["margin"]               = _clamp(uc["margin"])
    # 기본 검증
    if uc["annual_tasks"] < 0: uc["annual_tasks"] = 0
    if uc["minutes_saved_per_task"] < 0: uc["minutes_saved_per_task"] = 
0.0
    # normalize scenario probabilities
    p_sum = sum(s.probability for s in scenarios)
    if p_sum <= 0:
        raise ValueError("Scenario probabilities must sum to > 0.")
    for s in scenarios:
        s.probability = s.probability / p_sum
def run(cfg: Dict) -> Dict:
    """
    Backwards-compatible entrypoint.
    기존 스크립트들이 mod.run(cfg)를 호출하던 흐름을 유지합니다.
    """
    return run_roi_analysis(cfg, scenarios=DEFAULT_SCENARIOS)

import json, math, argparse
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

# --- Adoption S-curve (logistic) ---
def adoption_rate(t: int, K_max: float, b: float, t0: float) -> float:
    """
    Logistic adoption curve.
    t: month (1..12)
    K_max: maximum adoption (0-1)
    b: slope parameter (0.25-0.50 typical)
    t0: inflection point month (5-7 typical)
    """
    return K_max / (1.0 + math.e ** (-b * (t - t0)))

@dataclass
class Costs:
    api_credit_usd: float = 10000.0
    implementation_usd: float = 20000.0   # integration, change mgmt, training
    other_ongoing_usd: float = 0.0

    def total(self) -> float:
        return float(self.api_credit_usd + self.implementation_usd + self.other_ongoing_usd)

@dataclass
class UseCaseParams:
    # Work volume
    annual_tasks: int = 50000                  # e.g., support tickets / proposals / docs
    minutes_saved_per_task: float = 2.0
    hourly_rate_fully_loaded: float = 65.0     # $/hour

    # Automation & utilization
    automation_potential: float = 0.60         # max fraction of tasks that can be automated at maturity
    utilization_rate: float = 0.75             # fraction of eligible tasks actually using AI at maturity

    # Error reduction
    error_rate_before: float = 0.05
    error_rate_after: float = 0.01
    cost_per_error: float = 25.0

    # Revenue impact (optional, set any to 0 to ignore)
    revenue_units_per_year: int = 0            # e.g., proposals or content pieces created
    conversion_rate: float = 0.0               # 0-1
    avg_deal_value: float = 0.0                # $
    margin: float = 0.0                        # 0-1 margin
    attribution_factor: float = 0.5            # % of revenue credibly attributed to AI

@dataclass
class Scenario:
    name: str
    K_max: float
    b: float
    t0: float
    probability: float

DEFAULT_SCENARIOS = [
    Scenario("Conservative", K_max=0.50, b=0.25, t0=7, probability=0.30),
    Scenario("Expected"    , K_max=0.70, b=0.35, t0=6, probability=0.50),
    Scenario("Upside"      , K_max=0.85, b=0.50, t0=5, probability=0.20),
]

def monthly_projection(use: UseCaseParams, costs: Costs, scenario: Scenario) -> Dict:
    # Convert annual volumes to monthly baseline
    tasks_per_month = use.annual_tasks / 12.0
    revenue_units_per_month = use.revenue_units_per_year / 12.0 if use.revenue_units_per_year else 0.0

    rows = []
    cumulative_value = 0.0

    # Error rate improvement (value per automated task)
    error_improvement = max(0.0, use.error_rate_before - use.error_rate_after)

    for t in range(1, 13):
        adopt = adoption_rate(t, scenario.K_max, scenario.b, scenario.t0)  # 0..K_max
        # adoption drives portion of the "potential" we actually realize in month t
        realized_fraction = adopt * use.automation_potential * use.utilization_rate

        automated_tasks = tasks_per_month * realized_fraction
        minutes_saved = automated_tasks * use.minutes_saved_per_task
        labor_savings = (minutes_saved / 60.0) * use.hourly_rate_fully_loaded

        error_savings = automated_tasks * error_improvement * use.cost_per_error

        # Revenue impact scaled by adoption as well (if configured)
        revenue_impact = 0.0
        if use.revenue_units_per_year and use.conversion_rate and use.avg_deal_value and use.margin:
            extra_units = revenue_units_per_month * realized_fraction
            revenue_impact = extra_units * use.conversion_rate * use.avg_deal_value * use.margin * use.attribution_factor

        month_value = labor_savings + error_savings + revenue_impact
        cumulative_value += month_value

        rows.append({
            "month": t,
            "adoption": adopt,
            "realized_fraction": realized_fraction,
            "automated_tasks": automated_tasks,
            "labor_savings": labor_savings,
            "error_savings": error_savings,
            "revenue_impact": revenue_impact,
            "month_value": month_value,
            "cumulative_value": cumulative_value
        })

    total_benefit = cumulative_value
    total_costs = costs.total()
    roi_pct = ((total_benefit - total_costs) / total_costs) * 100.0 if total_costs > 0 else float('inf')
    payback_month = next((r["month"] for r in rows if r["cumulative_value"] >= total_costs), None)

    return {
        "scenario": scenario.name,
        "K_max": scenario.K_max, "b": scenario.b, "t0": scenario.t0, "probability": scenario.probability,
        "rows": rows,
        "total_benefit": total_benefit,
        "total_costs": total_costs,
        "roi_pct": roi_pct,
        "payback_month": payback_month
    }

def probability_weighted_roi(results: List[Dict]) -> float:
    return sum(r["roi_pct"] * r["probability"] for r in results)

def run(config: Dict) -> Dict:
    # Load sections
    costs = Costs(**config.get("costs", {}))
    use = UseCaseParams(**config.get("use_case", {}))

    scenarios_cfg = config.get("scenarios", None)
    scenarios = DEFAULT_SCENARIOS if scenarios_cfg is None else [Scenario(**s) for s in scenarios_cfg]

    results = [monthly_projection(use, costs, s) for s in scenarios]
    pw_roi = probability_weighted_roi(results)

    out = {
        "inputs": {"costs": asdict(costs), "use_case": asdict(use)},
        "scenarios": results,
        "probability_weighted_roi_pct": pw_roi
    }
    return out

def main():
    p = argparse.ArgumentParser(description="Claude Credit ROI Predictor")
    p.add_argument("--config", required=True, help="Path to JSON config with costs/use_case/scenarios")
    p.add_argument("--out_json", default="roi_output.json", help="Output JSON path")
    p.add_argument("--out_csv", default="roi_monthly.csv", help="Output monthly CSV path (Expected scenario)")
    args = p.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    out = run(cfg)

    # Write JSON
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # Also dump monthly rows of Expected scenario to CSV for quick visualization
    expected = next((s for s in out["scenarios"] if s["scenario"].lower() == "expected"), out["scenarios"][0])
    import csv
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(expected["rows"][0].keys()))
        w.writeheader()
        for r in expected["rows"]:
            w.writerow(r)

    print(f"Wrote {args.out_json} and {args.out_csv}")

if __name__ == "__main__":
    main()
# --- thin wrapper for Streamlit app ---
try:
    from roi_engine import run_roi_analysis  # if this file defines it differently
except Exception:
    # when file name is this file, try local symbol
    try:
        run_roi_analysis  # noqa: F401
    except NameError:
        pass

def run(cfg: dict):
    try:
        return run_roi_analysis(cfg)
    except NameError:
        # If original module already exposes run
        from importlib import import_module
        m = import_module("src.roi_engine")
        return m.run(cfg)
