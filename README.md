# M&V Counterfactual Builder

Interactive Streamlit app for whole-building savings verification using the counterfactual method.

**Steve Kromer, P.E., CMVP #1 | Counterfactual Designs**

## What it does

Steps through the mechanics of Option C whole-building M&V:

1. **Baseline Model** — Fit a regression model (Mean / Simple Linear / Multiple Linear) to pre-retrofit utility data. View R², CV(RMSE), and actual vs. predicted chart.
2. **Reporting Period** — Reveal one month at a time. Each month shows how the frozen baseline model is applied to current weather data to construct the counterfactual, with arithmetic shown explicitly.
3. **Cumulative Summary** — Running savings total, monthly bar chart, cumulative line chart, and method reference.

## Sample data

50,000 sq ft commercial office building, Denver CO. 12-month baseline + 12-month reporting period.

Simple Linear model result: **kWh = 122,961 + 101.5 × HDD** | R² = 0.837 | CV(RMSE) = 9.9% | Annual savings = 15.1%

## Run locally

```bash
pip install -r requirements.txt
streamlit run mv_counterfactual_builder.py
```

## Model types

| Model | Equation | Best for |
|-------|----------|----------|
| Mean | kWh = constant | Non-weather-sensitive loads |
| Simple Linear | kWh = b₀ + b₁×HDD | Heating-dominated buildings |
| Multiple Linear | kWh = b₀ + b₁×HDD + b₂×CDD | Mixed heating/cooling climates |

## Key concept

The baseline model is **frozen** at the end of the baseline period. During the reporting period,
only the independent variables (HDD/CDD) are updated — the model coefficients never change.
This is what makes the counterfactual verifiable and auditable.
