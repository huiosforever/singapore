"""
First‑Impression Personalization Engine (FIPE) — Streamlit Demo
================================================================
Luxury‑hotel arrival personalization prototype for Bay Street Hospitality.

• Demonstrates how public‑demand signals (Google Trends, IG geotags, flight
  manifests) + PMS/CRM data can be fused into a dynamic arrival playbook.
• Uses fully synthetic data so the app can be shared publicly without NDA risk.
• Designed for rapid deployment via `streamlit run fipe_app.py` or inside a
  container on Railway/Vercel.

Author: Bay Street Quantamental Terminal proto‑team — 2025‑07‑07
"""

from __future__ import annotations

import random
import string
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

################################################################################
# ----------------------------- CONFIGURATION ---------------------------------#
################################################################################

st.set_page_config(
    page_title="FIPE – Arrival Personalization Demo",
    layout="wide",
    initial_sidebar_state="expanded",
)

random.seed(42)
np.random.seed(42)

N_GUESTS = 300  # Number of synthetic reservations
ARRIVAL_WINDOW_MIN = 15  # minutes we care about for first‑impression bleed

LOYALTY_TIERS = ["Bronze", "Silver", "Gold", "Elite"]
IG_ACTIVITY_LEVEL = ["Low", "Medium", "High"]
ORIGIN_COUNTRIES = [
    "United States",
    "China",
    "Indonesia",
    "Australia",
    "Japan",
    "India",
    "United Kingdom",
]

MICRO_TOUCHES = {
    "Gold": ["In‑room Champagne", "Upgrade to Marina‑view suite", "Hand‑written note"],
    "Elite": ["BMW airport transfer", "Private check‑in lounge", "Curated Spotify playlist"],
    "Silver": ["Welcome mocktail", "Late checkout request", "City tips card"],
    "Bronze": ["Singapore sling voucher", "Early check‑in wait‑list", "HSR Wi‑Fi code"],
}

################################################################################
# --------------------------- DATA GENERATION ---------------------------------#
################################################################################


def _random_id(size: int = 8) -> str:
    """Generate a pseudo guest ID."""
    return "G" + "".join(random.choices(string.ascii_uppercase + string.digits, k=size))


def generate_synthetic_reservations(n: int = N_GUESTS) -> pd.DataFrame:
    base_date = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    arrivals = []
    for _ in range(n):
        guest_id = _random_id()
        loyalty = random.choices(LOYALTY_TIERS, weights=[0.4, 0.3, 0.2, 0.1])[0]
        ig_level = random.choices(IG_ACTIVITY_LEVEL, weights=[0.5, 0.35, 0.15])[0]
        country = random.choice(ORIGIN_COUNTRIES)
        # Arrival within first 15 min window (for demo simplicity)
        arrival_delta = timedelta(minutes=random.randint(0, ARRIVAL_WINDOW_MIN))
        arrival_ts = base_date + arrival_delta
        fx_risk = np.clip(np.random.normal(0.5, 0.2), 0.0, 1.0)  # 0 = low, 1 = high
        arrivals.append(
            {
                "guest_id": guest_id,
                "loyalty_tier": loyalty,
                "ig_activity": ig_level,
                "origin_country": country,
                "arrival_ts": arrival_ts,
                "fx_risk": fx_risk,
            }
        )
    return pd.DataFrame(arrivals)


def generate_trend_signals(countries: list[str]) -> pd.DataFrame:
    trend_score = np.random.randint(30, 100, size=len(countries))  # mock Google Trends
    return pd.DataFrame({"origin_country": countries, "trend_score": trend_score})

################################################################################
# ---------------------- PERSONALIZATION SCORING ------------------------------#
################################################################################


def calculate_personalization_score(row: pd.Series, weights: dict[str, float]) -> float:
    # Map categorical to numeric for simple linear score
    loyalty_val = {
        "Bronze": 0.2,
        "Silver": 0.4,
        "Gold": 0.7,
        "Elite": 1.0,
    }[row["loyalty_tier"]]
    ig_val = {"Low": 0.2, "Medium": 0.5, "High": 1.0}[row["ig_activity"]]
    fx_val = 1 - row["fx_risk"]  # lower FX risk preferred
    trend_val = row["trend_score"] / 100

    score = (
        weights["Loyalty"] * loyalty_val
        + weights["IG Activity"] * ig_val
        + weights["Trend Demand"] * trend_val
        + weights["FX Stability"] * fx_val
    ) / sum(weights.values())
    return round(score, 3)


def recommend_micro_touch(row: pd.Series) -> str:
    tier = row["loyalty_tier"]
    return random.choice(MICRO_TOUCHES[tier])

################################################################################
# ------------------------------ MAIN APP -------------------------------------#
################################################################################


def main() -> None:
    st.sidebar.title("FIPE – Segmentation Weights")
    weight_loyalty = st.sidebar.slider("Loyalty", 0.0, 5.0, 3.0, 0.1)
    weight_ig = st.sidebar.slider("IG Activity", 0.0, 5.0, 2.0, 0.1)
    weight_trend = st.sidebar.slider("Trend Demand", 0.0, 5.0, 2.5, 0.1)
    weight_fx = st.sidebar.slider("FX Stability", 0.0, 5.0, 2.0, 0.1)

    weights = {
        "Loyalty": weight_loyalty,
        "IG Activity": weight_ig,
        "Trend Demand": weight_trend,
        "FX Stability": weight_fx,
    }

    # Data creation
    reservations = generate_synthetic_reservations(N_GUESTS)
    trends = generate_trend_signals(ORIGIN_COUNTRIES)
    df = reservations.merge(trends, on="origin_country", how="left")

    # Score and recommend
    df["personalization_score"] = df.apply(calculate_personalization_score, axis=1, weights=weights)
    df["recommendation"] = df.apply(recommend_micro_touch, axis=1)

    # KPI Section
    st.title("Arrival Personalization Control Tower")
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Score", f"{df['personalization_score'].mean():.2f}")
    col2.metric(">0.8 VIPs", f"{(df['personalization_score'] > 0.8).sum()}")
    col3.metric("Arrivals Tracked", len(df))

    # Score Distribution
    with st.expander("Score Distribution", expanded=True):
        fig_hist = px.histogram(
            df,
            x="personalization_score",
            nbins=20,
            title="Personalization Score Histogram",
            color_discrete_sequence=[None],  # adhere to default color palette
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # Arrival minute vs score scatter
    df["arrival_min"] = df["arrival_ts"].dt.minute
    scatter_tab, table_tab = st.tabs(["Scatter", "Guest Intel Table"])

    with scatter_tab:
        fig_scatter = px.scatter(
            df,
            x="arrival_min",
            y="personalization_score",
            color="loyalty_tier",
            size="trend_score",
            hover_data=["guest_id", "recommendation"],
            title="Score vs. Arrival Minute (Bubble size = Demand Trend)",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with table_tab:
        st.dataframe(
            df.sort_values("personalization_score", ascending=False)[
                [
                    "guest_id",
                    "loyalty_tier",
                    "ig_activity",
                    "origin_country",
                    "personalization_score",
                    "recommendation",
                ]
            ].reset_index(drop=True),
            use_container_width=True,
        )

    # Recommendation playbook download
    csv = df[[
        "guest_id",
        "loyalty_tier",
        "arrival_ts",
        "origin_country",
        "personalization_score",
        "recommendation",
    ]].to_csv(index=False)
    st.download_button(
        label="Download Arrival Playbook (CSV)",
        data=csv,
        file_name="fipe_arrival_playbook.csv",
        mime="text/csv",
    )

    st.caption(
        "\u24D8  All data is randomly generated.  Score formula is illustrative only."
    )


if __name__ == "__main__":
    main()
