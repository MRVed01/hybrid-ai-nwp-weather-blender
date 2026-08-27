from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.blending import apply_quantile_mapping, blend_forecasts, normalize_weights
from src.config import DATA_FILE, EVAL_FILE, VARIABLES, WEIGHTS_FILE, QM_FILE
from src.data_generation import generate_synthetic_weather, save_dataset
from src.extremes import alert_summary
from src.features import make_features


st.set_page_config(
    page_title="Hybrid AI–NWP Weather Blender",
    page_icon="🌦️",
    layout="wide",
)


@st.cache_data
def load_data():
    if not DATA_FILE.exists():
        save_dataset(generate_synthetic_weather(), DATA_FILE)
    return pd.read_csv(DATA_FILE)


@st.cache_resource
def load_models():
    if not WEIGHTS_FILE.exists() or not QM_FILE.exists():
        import train_blender
        train_blender.main()
    return joblib.load(WEIGHTS_FILE), joblib.load(QM_FILE)


df = load_data()
bundle, qms = load_models()

st.title("🌦️ Hybrid AI–NWP Weather Blending Framework")
st.caption(
    "Context-aware meta-learning + tail-aware quantile mapping "
    "for adaptive multi-model forecasting."
)

with st.sidebar:
    st.header("Controls")
    variable = st.selectbox("Forecast variable", VARIABLES, index=1)
    lead = st.select_slider(
        "Forecast lead time", options=[6, 24, 72, 120], value=24
    )
    day = st.slider(
        "Synthetic forecast day",
        int(df["day"].min()),
        int(df["day"].max()),
        int(df["day"].max()),
    )
    season = st.selectbox(
        "Season", ["All", "winter", "spring", "summer", "monsoon"]
    )

part = df[
    (df["variable"] == variable)
    & (df["lead_hours"] == lead)
    & (df["day"] == day)
].copy()

if season != "All":
    part = part[part["season"] == season].copy()

if part.empty:
    st.warning("No grid cells match these filters.")
    st.stop()

models = bundle["models"][variable]
raw = np.column_stack([
    models[n].predict(make_features(part))
    for n in ["nwp", "ai", "ensemble"]
])
weights = normalize_weights(raw)

part["w_nwp"] = weights[:, 0]
part["w_ai"] = weights[:, 1]
part["w_ensemble"] = weights[:, 2]
part["blended_raw"] = blend_forecasts(part, weights)
part["blended"] = apply_quantile_mapping(
    part["blended_raw"].to_numpy(), qms[variable]
)
part["dominant_model"] = np.array(["NWP", "AI", "Ensemble"])[np.argmax(weights, axis=1)]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Mean NWP weight", f"{part.w_nwp.mean():.1%}")
c2.metric("Mean AI weight", f"{part.w_ai.mean():.1%}")
c3.metric("Mean Ensemble weight", f"{part.w_ensemble.mean():.1%}")
c4.metric("Dominant model", part["dominant_model"].mode().iat[0])

st.subheader("Dynamic Model Weight Map")
selected_model = st.radio(
    "Show weight", ["NWP", "AI", "Ensemble"], horizontal=True
)
weight_col = {"NWP": "w_nwp", "AI": "w_ai", "Ensemble": "w_ensemble"}[selected_model]

fig = px.density_heatmap(
    part,
    x="longitude",
    y="latitude",
    z=weight_col,
    nbinsx=20,
    nbinsy=15,
    range_color=[0, 1],
    title=f"{selected_model} adaptive weight at +{lead}h",
)
fig.update_layout(height=470)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Side-by-Side Spatial Forecasts")
map_df = part.melt(
    id_vars=["latitude", "longitude"],
    value_vars=["nwp", "ai", "ensemble", "blended", "truth"],
    var_name="model",
    value_name="value",
)
fig2 = px.scatter(
    map_df,
    x="longitude",
    y="latitude",
    color="value",
    facet_col="model",
    facet_col_wrap=3,
    hover_data=["value"],
    title=(
        f"{variable.title()} at +{lead}h — "
        "NWP / AI / Ensemble / Blended / Ground Truth"
    ),
)
fig2.update_traces(marker={"size": 8})
fig2.update_layout(height=650)
st.plotly_chart(fig2, use_container_width=True)

st.subheader("🚨 Extreme Weather Alert Panel")
alerts = alert_summary(
    df[(df["day"] == day) & (df["lead_hours"] == lead)]
)
a1, a2, a3, a4 = st.columns(4)
for col, (_, row) in zip([a1, a2, a3, a4], alerts.iterrows()):
    col.metric(row["event"], f'{int(row["count"]):,}')

event_df = part.copy()
event_df["alert"] = (
    ((variable == "temperature") & (event_df["truth"] >= 35))
    | ((variable == "rainfall") & (event_df["truth"] >= 50))
    | ((variable == "wind") & (event_df["truth"] >= 17))
    | ((variable == "pressure") & (event_df["truth"] <= 995))
    | (event_df["cyclone_regime"] == 1)
)
alerts_here = event_df[event_df["alert"]]

if not alerts_here.empty:
    st.dataframe(
        alerts_here[
            ["latitude", "longitude", "truth", "blended",
             "dominant_model", "cyclone_regime"]
        ].sort_values("truth", ascending=False).head(20),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.success("No extreme cells detected for the selected variable/lead/day.")

st.subheader("📈 Lead-Time Weight Shift")
lead_df = df[
    (df["variable"] == variable) & (df["day"] == day)
].copy()
all_raw = np.column_stack([
    models[n].predict(make_features(lead_df))
    for n in ["nwp", "ai", "ensemble"]
])
all_w = normalize_weights(all_raw)
lead_df["NWP"] = all_w[:, 0]
lead_df["AI"] = all_w[:, 1]
lead_df["Ensemble"] = all_w[:, 2]

curve = lead_df.groupby("lead_hours")[["NWP", "AI", "Ensemble"]].mean().reset_index()
curve_long = curve.melt(
    "lead_hours", var_name="model", value_name="weight"
)
fig3 = px.line(
    curve_long,
    x="lead_hours",
    y="weight",
    color="model",
    markers=True,
)
fig3.update_yaxes(range=[0, 1], tickformat=".0%")
fig3.update_layout(
    height=400,
    xaxis_title="Forecast lead (hours)",
    yaxis_title="Mean adaptive weight",
)
st.plotly_chart(fig3, use_container_width=True)

if EVAL_FILE.exists():
    st.subheader("Verification Evidence")
    eval_df = pd.read_csv(EVAL_FILE)
    st.dataframe(eval_df.round(3), use_container_width=True, hide_index=True)
else:
    st.info("Run `python evaluate.py` to populate the verification table.")
