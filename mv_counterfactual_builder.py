"""
M&V Counterfactual Builder
Whole-Building Option C — Interactive Month-by-Month Reporting Period Walkthrough

Steve Kromer, P.E., CMVP #1 | Counterfactual Designs

Run:  streamlit run mv_counterfactual_builder.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ── Sample Data ─────────────────────────────────────────────────────────────
# 50,000 sq ft commercial office building, Denver CO
# Baseline year: pre-retrofit (12 months)
# Reporting year: post-retrofit (same building, same weather drivers)

BASELINE_DATA = {
    "Month": ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
    "HDD":   [974,  812,  589,  294,   87,    5,    0,    0,   28,  236,  561,  862],
    "CDD":   [  0,    0,    6,   27,  127,  303,  418,  389,  213,   46,    3,    0],
    "kWh":   [235800, 207400, 170600, 130200, 112800, 133400,
               150200, 144800, 119600, 128400, 174200, 219600],
}

REPORTING_DATA = {
    "Month": ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
    "HDD":   [943,  798,  612,  278,   94,    8,    0,    0,   31,  251,  544,  891],
    "CDD":   [  0,    0,    4,   31,  119,  318,  401,  374,  228,   39,    5,    0],
    "kWh_actual": [199400, 175600, 145300, 110200,  95400, 113200,
                   127600, 123000, 101800, 109200, 148000, 187000],
}

MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

# ── Model Fitting ────────────────────────────────────────────────────────────

def fit_model(df, model_type):
    """Fit baseline model. Returns dict with coefficients, R², CV(RMSE)."""
    y = df["kWh"].values.astype(float)
    n = len(y)

    if model_type == "mean":
        y_pred = np.full(n, y.mean())
        eq = f"kWh = {y.mean():,.0f}  (annual mean)"
        coefs = {"intercept": y.mean()}

    elif model_type == "linear_hdd":
        x = df["HDD"].values.astype(float)
        xm, ym = x.mean(), y.mean()
        b1 = np.sum((x - xm) * (y - ym)) / np.sum((x - xm) ** 2)
        b0 = ym - b1 * xm
        y_pred = b0 + b1 * x
        sign = "+" if b1 >= 0 else "-"
        eq = f"kWh = {b0:,.0f} {sign} {abs(b1):,.1f} × HDD"
        coefs = {"intercept": b0, "hdd": b1}

    elif model_type == "linear_hdd_cdd":
        X = np.column_stack([np.ones(n), df["HDD"].values, df["CDD"].values]).astype(float)
        coef = np.linalg.lstsq(X, y, rcond=None)[0]
        b0, b1, b2 = coef
        y_pred = X @ coef
        s1 = "+" if b1 >= 0 else "-"
        s2 = "+" if b2 >= 0 else "-"
        eq = f"kWh = {b0:,.0f} {s1} {abs(b1):,.1f}×HDD {s2} {abs(b2):,.1f}×CDD"
        coefs = {"intercept": b0, "hdd": b1, "cdd": b2}

    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    rmse = np.sqrt(ss_res / n)
    cvrmse = rmse / y.mean() * 100

    return {
        "type": model_type,
        "coefs": coefs,
        "r2": r2,
        "cvrmse": cvrmse,
        "equation": eq,
        "y_pred": y_pred,
        "y_actual": y,
    }


def predict_counterfactual(model, hdd, cdd):
    """Apply fixed baseline model to reporting period independent variables."""
    c = model["coefs"]
    if model["type"] == "mean":
        return c["intercept"]
    elif model["type"] == "linear_hdd":
        return c["intercept"] + c["hdd"] * hdd
    elif model["type"] == "linear_hdd_cdd":
        return c["intercept"] + c["hdd"] * hdd + c["cdd"] * cdd


# ── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="M&V Counterfactual Builder",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ M&V Counterfactual Builder")
st.caption("Whole-Building Option C — Interactive Month-by-Month Reporting Period Walkthrough")

st.markdown("""
**How this works:** Select a baseline regression model, then step through the reporting
period one month at a time. Each month you *collect* the independent variable (HDD/CDD)
and *measure* actual consumption — and the tool shows you how the counterfactual is
constructed and how savings accumulate.
""")

# ── Session State Init ───────────────────────────────────────────────────────

if "model_fitted" not in st.session_state:
    st.session_state.model_fitted = False
if "model" not in st.session_state:
    st.session_state.model = None
if "model_type" not in st.session_state:
    st.session_state.model_type = "linear_hdd"
if "current_month" not in st.session_state:
    st.session_state.current_month = 0
if "revealed" not in st.session_state:
    st.session_state.revealed = []

df_base = pd.DataFrame(BASELINE_DATA)
df_rep  = pd.DataFrame(REPORTING_DATA)

tab1, tab2, tab3 = st.tabs(["📐 Baseline Model", "📅 Reporting Period", "📊 Cumulative Summary"])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — BASELINE MODEL
# ═══════════════════════════════════════════════════════════════════════════

with tab1:
    st.subheader("Step 1 — Select & Fit a Baseline Model")

    col_info, col_data = st.columns([1, 1])

    with col_info:
        st.markdown("""
**Building:** 50,000 sq ft commercial office, Denver CO
**Baseline period:** 12 months (pre-retrofit)
**Independent variables available:** HDD, CDD

The baseline model establishes the *expected* energy use as a function of
weather. Once fitted, the model is **frozen** — it does not change during
the reporting period. This is the fixed-model requirement for whole-building
savings verification.
        """)

        model_choice = st.radio(
            "Select model type:",
            options=["mean", "linear_hdd", "linear_hdd_cdd"],
            format_func=lambda x: {
                "mean":           "① Mean  —  no independent variable (simplest)",
                "linear_hdd":     "② Simple Linear  —  kWh = f(HDD)",
                "linear_hdd_cdd": "③ Multiple Linear  —  kWh = f(HDD, CDD)",
            }[x],
            index=1,
            key="model_radio",
        )
        st.session_state.model_type = model_choice

        if st.button("🔧 Fit Baseline Model", type="primary"):
            m = fit_model(df_base, model_choice)
            st.session_state.model = m
            st.session_state.model_fitted = True
            st.session_state.current_month = 0
            st.session_state.revealed = []

    with col_data:
        st.markdown("**Baseline data:**")
        disp = df_base.copy()
        disp["kWh"] = disp["kWh"].map("{:,}".format)
        st.dataframe(disp, use_container_width=True, hide_index=True)

    if st.session_state.model_fitted and st.session_state.model is not None:
        m = st.session_state.model
        st.divider()

        col_stats, col_chart = st.columns([1, 2])

        with col_stats:
            st.markdown("### Model Results")
            st.code(m["equation"], language=None)

            st.metric("R²", f"{m['r2']:.3f}",
                      delta="≥ 0.75 recommended" if m["r2"] < 0.75 else "✓ Good fit")
            st.metric("CV(RMSE)", f"{m['cvrmse']:.1f}%",
                      delta="≤ 20% recommended" if m["cvrmse"] > 20 else "✓ Good fit")

            if m["r2"] >= 0.75 and m["cvrmse"] <= 20:
                st.success("Model meets recommended goodness-of-fit criteria.")
            else:
                st.warning("Model may not provide reliable predictions. Consider a different form.")

            with st.expander("ℹ️ What do R² and CV(RMSE) mean?"):
                st.markdown("""
**R²** (coefficient of determination) measures how much of the variation
in energy use is explained by the model. R² = 1.0 is a perfect fit;
values ≥ 0.75 are generally considered reliable for whole-building work.

**CV(RMSE)** (coefficient of variation of root mean square error) measures
the typical prediction error as a percentage of mean energy use.
Lower is better; ≤ 20% is the standard threshold for reliable models.

Together they answer: *does this model reliably predict what the building
would have used without the retrofit?*
                """)

        with col_chart:
            fig, ax = plt.subplots(figsize=(8, 4))
            x_pos = np.arange(12)
            ax.bar(x_pos - 0.2, m["y_actual"] / 1000, 0.4,
                   label="Actual baseline kWh", color="#4C72B0", alpha=0.8)
            ax.bar(x_pos + 0.2, m["y_pred"] / 1000, 0.4,
                   label="Model predicted kWh", color="#DD8452", alpha=0.8)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(MONTHS)
            ax.set_ylabel("kWh (thousands)")
            ax.set_title(f"Baseline: Actual vs. Model Predicted  |  R²={m['r2']:.3f}  CV(RMSE)={m['cvrmse']:.1f}%")
            ax.legend()
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.info("✅ Model is fitted and locked. Proceed to the **Reporting Period** tab to build the counterfactual.")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — REPORTING PERIOD
# ═══════════════════════════════════════════════════════════════════════════

with tab2:
    st.subheader("Step 2 — Build the Counterfactual Month by Month")

    if not st.session_state.model_fitted:
        st.warning("⬅️ Go to the **Baseline Model** tab first and fit a model.")
    else:
        m = st.session_state.model

        st.markdown(f"""
**Fitted model:** `{m['equation']}`
**R²:** {m['r2']:.3f}   **CV(RMSE):** {m['cvrmse']:.1f}%

Each month, you collect the **independent variable** (HDD and/or CDD) from
your building's utility meter data, then apply the *frozen baseline model*
to calculate what the building *would have used* without the retrofit.
The difference is your **verified savings** for that month.
        """)

        st.divider()

        n_revealed = len(st.session_state.revealed)

        if n_revealed < 12:
            next_month = MONTHS[n_revealed]
            if st.button(f"▶ Reveal {next_month} reporting data", type="primary"):
                st.session_state.revealed.append(n_revealed)
                st.rerun()
        else:
            st.success("🎉 All 12 months revealed! See the **Cumulative Summary** tab for final results.")

        if st.button("🔄 Reset reporting period"):
            st.session_state.revealed = []
            st.rerun()

        st.divider()

        if len(st.session_state.revealed) == 0:
            st.info("Click **▶ Reveal Jan** to begin stepping through the reporting period.")
        else:
            rows = []
            for i in st.session_state.revealed:
                hdd    = REPORTING_DATA["HDD"][i]
                cdd    = REPORTING_DATA["CDD"][i]
                actual = REPORTING_DATA["kWh_actual"][i]
                cfact  = predict_counterfactual(m, hdd, cdd)
                saving = cfact - actual
                pct    = saving / cfact * 100 if cfact > 0 else 0
                rows.append({
                    "Month":          MONTHS[i],
                    "HDD":            hdd,
                    "CDD":            cdd,
                    "Actual kWh":     actual,
                    "Counterfactual": round(cfact),
                    "Savings kWh":    round(saving),
                    "Savings %":      round(pct, 1),
                })

            df_rows = pd.DataFrame(rows)
            latest  = rows[-1]

            col_a, col_b, col_c = st.columns(3)
            col_a.metric(f"{latest['Month']} — Actual kWh",    f"{latest['Actual kWh']:,}")
            col_b.metric(f"{latest['Month']} — Counterfactual", f"{latest['Counterfactual']:,}")
            col_c.metric(f"{latest['Month']} — Savings",
                         f"{latest['Savings kWh']:,} kWh",
                         delta=f"{latest['Savings %']}%")

            with st.expander("ℹ️ How is the counterfactual calculated for this month?"):
                i       = st.session_state.revealed[-1]
                hdd_v   = REPORTING_DATA["HDD"][i]
                cdd_v   = REPORTING_DATA["CDD"][i]
                cfact_v = predict_counterfactual(m, hdd_v, cdd_v)
                c       = m["coefs"]
                if m["type"] == "mean":
                    st.markdown(f"""
**Model:** Mean
**Formula:** kWh = {c['intercept']:,.0f}
**Result:** Counterfactual = **{cfact_v:,.0f} kWh** (same every month — no IV)

> The mean model ignores weather entirely. Simple, but weak for weather-sensitive buildings.
                    """)
                elif m["type"] == "linear_hdd":
                    st.markdown(f"""
**Model:** Simple Linear
**Formula:** kWh = {c['intercept']:,.0f} + {c['hdd']:,.1f} × HDD
**This month's HDD:** {hdd_v}
**Calculation:** {c['intercept']:,.0f} + {c['hdd']:,.1f} × {hdd_v} = **{cfact_v:,.0f} kWh**

> We plug this month's *actual* HDD into the *frozen* baseline equation.
> The model answers: "what would this building have used in this weather, without the retrofit?"
                    """)
                elif m["type"] == "linear_hdd_cdd":
                    st.markdown(f"""
**Model:** Multiple Linear
**Formula:** kWh = {c['intercept']:,.0f} + {c['hdd']:,.1f}×HDD + {c['cdd']:,.1f}×CDD
**This month's HDD:** {hdd_v}  **CDD:** {cdd_v}
**Calculation:** {c['intercept']:,.0f} + {c['hdd']:,.1f}×{hdd_v} + {c['cdd']:,.1f}×{cdd_v} = **{cfact_v:,.0f} kWh**

> Both heating AND cooling weather are captured. Better fit for buildings
> with significant cooling loads.
                    """)

            fig2, ax2 = plt.subplots(figsize=(10, 4))
            x = np.arange(len(rows))
            ax2.bar(x - 0.2, [r["Counterfactual"] / 1000 for r in rows], 0.4,
                    label="Counterfactual (what it would have used)", color="#DD8452", alpha=0.85)
            ax2.bar(x + 0.2, [r["Actual kWh"] / 1000 for r in rows], 0.4,
                    label="Actual (post-retrofit)", color="#4C72B0", alpha=0.85)
            ax2.set_xticks(x)
            ax2.set_xticklabels([r["Month"] for r in rows])
            ax2.set_ylabel("kWh (thousands)")
            ax2.set_title("Reporting Period: Counterfactual vs. Actual")
            ax2.legend()
            ax2.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

            st.markdown("**Month-by-month detail:**")
            df_display = df_rows.copy()
            df_display["Actual kWh"]     = df_display["Actual kWh"].map("{:,}".format)
            df_display["Counterfactual"] = df_display["Counterfactual"].map("{:,}".format)
            df_display["Savings kWh"]    = df_display["Savings kWh"].map("{:,}".format)
            df_display["Savings %"]      = df_display["Savings %"].map("{:.1f}%".format)
            st.dataframe(df_display, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — CUMULATIVE SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

with tab3:
    st.subheader("Step 3 — Cumulative Summary & Annual Savings")

    if not st.session_state.model_fitted:
        st.warning("⬅️ Fit a baseline model first (Tab 1), then step through the reporting period (Tab 2).")
    elif len(st.session_state.revealed) == 0:
        st.info("Reveal at least one month in the **Reporting Period** tab to see results here.")
    else:
        m = st.session_state.model
        rows = []
        for i in st.session_state.revealed:
            hdd    = REPORTING_DATA["HDD"][i]
            cdd    = REPORTING_DATA["CDD"][i]
            actual = REPORTING_DATA["kWh_actual"][i]
            cfact  = predict_counterfactual(m, hdd, cdd)
            saving = cfact - actual
            rows.append({
                "Month":              MONTHS[i],
                "Actual kWh":         actual,
                "Counterfactual":     round(cfact),
                "Savings kWh":        round(saving),
                "Cumulative Savings": 0,
            })

        cum = 0
        for r in rows:
            cum += r["Savings kWh"]
            r["Cumulative Savings"] = cum

        df_cum        = pd.DataFrame(rows)
        total_actual  = df_cum["Actual kWh"].sum()
        total_cfact   = df_cum["Counterfactual"].sum()
        total_savings = df_cum["Savings kWh"].sum()
        pct_savings   = total_savings / total_cfact * 100 if total_cfact > 0 else 0
        n_months      = len(rows)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Months reported",      f"{n_months} / 12")
        col2.metric("Total counterfactual", f"{total_cfact:,} kWh")
        col3.metric("Total actual",         f"{total_actual:,} kWh")
        col4.metric("Verified savings",
                    f"{total_savings:,} kWh",
                    delta=f"{pct_savings:.1f}% reduction")

        fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 4))

        months_shown = [r["Month"] for r in rows]
        savings_vals = [r["Savings kWh"] / 1000 for r in rows]
        colors = ["#2ca02c" if s >= 0 else "#d62728" for s in savings_vals]
        ax3a.bar(months_shown, savings_vals, color=colors, alpha=0.85)
        ax3a.axhline(0, color="black", linewidth=0.8)
        ax3a.set_ylabel("kWh saved (thousands)")
        ax3a.set_title("Monthly Verified Savings")
        ax3a.grid(axis="y", alpha=0.3)

        cum_vals = [r["Cumulative Savings"] / 1000 for r in rows]
        ax3b.plot(months_shown, cum_vals, "o-", color="#2ca02c", linewidth=2.5, markersize=7)
        ax3b.fill_between(range(len(months_shown)), cum_vals, alpha=0.15, color="#2ca02c")
        ax3b.set_ylabel("kWh saved (thousands, cumulative)")
        ax3b.set_title("Cumulative Verified Savings")
        ax3b.grid(alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

        st.markdown("**Running totals:**")
        df_disp2 = df_cum.copy()
        for col in ["Actual kWh", "Counterfactual", "Savings kWh", "Cumulative Savings"]:
            df_disp2[col] = df_disp2[col].map("{:,}".format)
        st.dataframe(df_disp2, use_container_width=True, hide_index=True)

        with st.expander("📋 Method Reference"):
            st.markdown(f"""
### Whole-Building Savings Verification — Counterfactual Method

**Core principle:** The baseline model is fitted to *pre-retrofit* data, then
applied to *reporting period* independent variables to construct the
**counterfactual** — what the building *would have consumed* without the retrofit.

**Savings formula:**
> Savings = Counterfactual − Actual Consumption

> Counterfactual = Baseline Model(Reporting Period IV data)

**This project:**
| Item | Value |
|------|-------|
| Baseline model | `{m['equation']}` |
| Model R² | {m['r2']:.3f} |
| Model CV(RMSE) | {m['cvrmse']:.1f}% |
| Months reported | {n_months} |
| Total counterfactual | {total_cfact:,} kWh |
| Total actual | {total_actual:,} kWh |
| Verified savings | {total_savings:,} kWh ({pct_savings:.1f}%) |

**Why freeze the model?**
The baseline model must not change during the reporting period. Refitting the
model to include reporting period data would corrupt the counterfactual —
the model would "learn" the post-retrofit behavior and understate savings.
The frozen model is what makes savings verifiable and auditable.
            """)

        if n_months < 12:
            st.info(f"📅 {12 - n_months} months remaining. Return to the **Reporting Period** tab to reveal more.")
        else:
            st.success(f"✅ Full 12-month reporting period complete. Annual verified savings: **{total_savings:,} kWh ({pct_savings:.1f}%)**.")
