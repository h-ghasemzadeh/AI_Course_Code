import plotly.graph_objects as go
import pandas as pd

# Data
years = [2000, 2005, 2010, 2015, 2020, 2023, 2025, 2030, 2035]
share = [0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.28, 0.32]

# Create figure
fig = go.Figure()

# Add main line
fig.add_trace(go.Scatter(
    x=years,
    y=share,
    mode='lines+markers',
    line=dict(color='green', width=4),
    marker=dict(size=8),
    name='Geothermal Share'
))

# Update layout
fig.update_layout(
    title="Geothermal Energy's Share of Global Total Energy Production (2000-2035p)",
    xaxis_title='Year',
    yaxis_title='Share of Global Total Energy (%)',
    yaxis_range=[0.15, 0.35],
    showlegend=True,
    template='plotly_white'
)

# Add vertical line for forecast
fig.add_vline(x=2023.5, line_dash="dash", line_color="red", annotation_text="Forecast Start")

fig.show()