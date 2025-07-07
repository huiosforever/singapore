"""
First-Impression Personalization Engine (FIPE) - Streamlit Demo
================================================================
Luxury-hotel arrival-personalization prototype for Bay Street Hospitality.

- Demonstrates how public-demand signals (Google Trends, IG geotags, flight
  manifests) plus PMS/CRM data can be fused into a dynamic arrival playbook.
- Uses fully synthetic data so the app can be shared publicly without NDA risk.
- Designed for rapid deployment via `streamlit run app.py` or inside a
  container on Railway / Vercel.

Author : Bay Street Quantamental Terminal proto-team
Date   : 2025-07-07
Tested : Python 3.13
"""

from __future__ import annotations

import random
import string
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# --------------------------------------------------------------------------- #
# CONFIGURATION                                                               #
# --------------------------------------------------------------------------- #

st.set_page_config(
    page_title="FIPE – Arrival Personalization Demo",
    layout="wide",
    initial_sidebar_state="expanded",
)

random.seed(42)
np.random.seed(42)

N_GUESTS = 300           # synthetic reservations to generate
ARRIVAL_WINDOW_MIN = 15  # minutes that matter for the first-impression bleed

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

MICRO_TOUCHES: dict[str, list[str]] = {
    "Gold": [
        "In-room Champagne",
        "Upgrade to Marina-view suite",
        "Hand-written note",
    ],
    "Elite": [
        "BMW airport transfer",
        "Private check-in lounge",
        "Curated Spotify playlist",
    ],
    "Silver": [
        "Welcome mocktail",
        "Late-checkout request",
        "City tips card",
    ],
    "Bronze": [
        "Singapore Sling voucher",
        "Early check-in wait-list",
        "HSR Wi-Fi code",
    ],
}

# --------------------------------------------------------------------------- #
# DATA GENERATION                                                             #
# --------------------------------------------------------------------------- #


def _random_id(size: int = 8) -> str:
    """Return a pseudo-guest ID."""
    return "G" + "".join(random.choices(string.ascii_uppercase + string.digits, k=size))


def generate_synthetic_reservations(n: int = N_GUESTS) -> pd.DataFrame:
    """Fabricate reservation records."""
    base_date = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    arrivals = []
    for _ in range(n):
        guest_id = _random_id()
        loyalty = random.choices(LOYALTY_TIERS, weights=[0.4, 0.3, 0.2, 0.1])[0]
        ig_level = random.choices(IG_ACTIVITY_LEVEL, weights=[0.5, 0.35, 0.15])[0]
        country = random.choice(ORIGIN_COUNTRIES)
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
    """Mock Google-Trends demand scores for each origin country."""
    trend_score = np.random.randint(30, 100, size=len(countries))
    return pd.DataFrame({"origin_country": countries, "trend_score": trend_score})


# --------------------------------------------------------------------------- #
# PERSONALIZATION SCORING                                                     #
# --------------------------------------------------------------------------- #


def calculate_personalization_score(row: pd.Series, weights: dict[str, float]) -> float:
    """Compute weighted personalization score (0-1)."""
    loyalty_val = {"Bronze": 0.2, "Silver": 0.4, "Gold": 0.7, "Elite": 1.0}[row["loyalty_tier"]]
    ig_val = {"Low": 0.2, "Medium": 0.5, "High": 1.0}[row["ig_activity"]]
    fx_val = 1.0 - row["fx_risk"]  # lower FX risk preferred
    trend_val = row["trend_score"] / 100.0

    score = (
        weights["Loyalty"] * loyalty_val
        + weights["IG Activity"] * ig_val
        + weights["Trend Demand"] * trend_val
        + weights["FX Stability"] * fx_val
    ) / sum(weights.values())
    return round(score, 3)


def recommend_micro_touch(row: pd.Series) -> str:
    """Return a micro-touch recommendation based on loyalty tier."""
    return random.choice(MICRO_TOUCHES[row["loyalty_tier"]])


# --------------------------------------------------------------------------- #
# STREAMLIT APP                                                               #
# --------------------------------------------------------------------------- #


def main() -> None:
    # --- sidebar controls --------------------------------------------------- #
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

    # --- generate & score data --------------------------------------------- #
    reservations = generate_synthetic_reservations(N_GUESTS)
    trends = generate_trend_signals(ORIGIN_COUNTRIES)
    df = reservations.merge(trends, on="origin_country", how="left")

    df["personalization_score"] = df.apply(calculate_personalization_score, axis=1, weights=weights)
    df["recommendation"] = df.apply(recommend_micro_touch, axis=1)
    df["arrival_min"] = df["arrival_ts"].dt.minute

    # --- KPI tiles ---------------------------------------------------------- #
    st.title("Arrival Personalization Control Tower")
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Score", f"{df['personalization_score'].mean():.2f}")
    c2.metric("> 0.8 VIPs", f"{(df['personalization_score'] > 0.8).sum()}")
    c3.metric("Arrivals Tracked", len(df))

    # --- histogram ---------------------------------------------------------- #
    with st.expander("Score Distribution", expanded=True):
        fig_hist = px.histogram(
            df,
            x="personalization_score",
            nbins=20,
            title="Personalization Score Histogram",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # --- scatter + table tabs ---------------------------------------------- #
    scatter_tab, table_tab = st.tabs(["Scatter", "Guest Intel Table"])

    with scatter_tab:
        fig_scatter = px.scatter(
            df,
            x="arrival_min",
            y="personalization_score",
            color="loyalty_tier",
            size="trend_score",
            hover_data=["guest_id", "recommendation"],
            title="Score vs Arrival Minute (bubble size = demand trend)",
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

    # --- download playbook -------------------------------------------------- #
    csv_bytes = df[
        [
            "guest_id",
            "loyalty_tier",
            "arrival_ts",
            "origin_country",
            "personalization_score",
            "recommendation",
        ]
    ].to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Arrival Playbook (CSV)",
        data=csv_bytes,
        file_name="fipe_arrival_playbook.csv",
        mime="text/csv",
    )

    st.caption("ℹ️ All data is randomly generated; scoring is illustrative only.")


if __name__ == "__main__":
    main()
