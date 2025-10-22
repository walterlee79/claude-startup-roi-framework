"""
Claude Credit ROI Prediction Engine
====================================

A comprehensive framework for predicting ROI of Claude API credits for startups.
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
    def calculate_adoption(month: int, k_max: float, b: float, t0: float) -> float:
        """
        Calculate adoption rate using logistic curve
        
        Formula: Adoption(t) = K_max / (1 + e^(-b*(t - t0)))
        
        Args:
            month: Current month (1-12)
            k_max: Maximum adoption rate (0.50-0.85)
            b: Slope parameter (0.25-0.50)
            t0: Inflection point in months (5-7)
        """
        return k_max / (1 + np.exp(-b * (month - t0)))
    
    @staticmethod
    def get_monthly_adoptions(months: int, scenario: ScenarioConfig) -> np.ndarray:
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
        
    def calculate_monthly_value(self, adoption_rate: float) -> Dict[str, float]:
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
        error_reduction = self.use_case.error_rate_before - self.use_case.error_rate_after
        error_savings = tasks_automated * error_reduction * self.use_case.cost_per_error
        
        # Revenue impact (for revenue-generating use cases)
        monthly_units = self.use_case.revenue_units_per_year / 12
        productivity_boost = adoption_rate * self.use_case.automation_potential
        
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
        adoption_rates = AdoptionCurve.get_monthly_adoptions(months, scenario)
        
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
        """Calculate probability-weighted ROI across multiple scenarios"""
        
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
                    use_case = UseCaseConfig(**self.base_config["use_case"])
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


def run_roi_analysis(config: Dict, scenarios: Optional[List[ScenarioConfig]] = None) -> Dict:
    """
    Main entry point for ROI analysis
    
    Args:
        config: Configuration dictionary with 'use_case' and 'costs' keys
        scenarios: Optional list of scenario configs (defaults to DEFAULT_SCENARIOS)
    
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
    print(f"Probability-Weighted ROI: {results['probability_weighted_roi_pct']:.1f}%")
