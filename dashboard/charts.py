"""Graphiques Plotly du dashboard (palette adaptée au daltonisme)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

BLUE = "#0072B2"  # accord / facteur protecteur
ORANGE = "#D55E00"  # refus / facteur de risque
GREY = "#8C8C8C"
FONT = {"size": 16}


def gauge(probability: float, threshold: float) -> go.Figure:
    """Jauge de la probabilité de défaut, avec le seuil de refus marqué."""
    upper = 0.5 if max(probability, threshold) < 0.4 else 1.0
    color = ORANGE if probability >= threshold else BLUE
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": " %", "valueformat": ".1f", "font": {"size": 44}},
            gauge={
                "axis": {"range": [0, upper * 100], "ticksuffix": " %"},
                "bar": {"color": color, "thickness": 0.35},
                "steps": [
                    {"range": [0, threshold * 100], "color": "#CFE5F3"},
                    {"range": [threshold * 100, upper * 100], "color": "#F7D9C4"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 4},
                    "thickness": 0.9,
                    "value": threshold * 100,
                },
            },
            title={"text": "Probabilité de défaut"},
        )
    )
    fig.update_layout(height=300, margin={"t": 70, "b": 10, "l": 60, "r": 60}, font=FONT)
    return fig


def factors_bar(factors: pd.DataFrame) -> go.Figure:
    """Barres des facteurs : orange (▲ augmente le risque) et bleu (▼ le réduit)."""
    data = factors.sort_values("impact")
    colors = [ORANGE if v > 0 else BLUE for v in data["impact"]]
    texts = [("▲ " if v > 0 else "▼ ") + f"{abs(v):.2f}" for v in data["impact"]]
    fig = go.Figure(
        go.Bar(
            x=data["impact"],
            y=data["libelle"],
            orientation="h",
            marker_color=colors,
            text=texts,
            textposition="outside",
            hovertext=data["valeur"],
        )
    )
    reach = float(data["impact"].abs().max()) * 1.4
    fig.update_layout(
        height=120 + 48 * len(data),
        margin={"l": 10, "r": 40, "t": 10, "b": 40},
        xaxis={"range": [-reach, reach], "title": "◄ réduit le risque | augmente le risque ►"},
        yaxis={"automargin": True},
        font=FONT,
        showlegend=False,
    )
    return fig


def distribution(
    everyone: pd.Series, similar: pd.Series, client_value: float | None, title: str
) -> go.Figure:
    """Histogrammes normalisés (tous / similaires) et position du client."""
    low, high = np.nanpercentile(everyone, [1, 99])
    bins = {"start": low, "end": high, "size": (high - low) / 30 if high > low else 1}
    fig = go.Figure()
    fig.add_histogram(
        x=everyone.clip(low, high), xbins=bins, histnorm="percent", name="Tous les clients",
        marker_color=GREY, opacity=0.6,
    )
    if len(similar) >= 10:
        fig.add_histogram(
            x=similar.clip(low, high), xbins=bins, histnorm="percent",
            name="Clients similaires", marker_color=BLUE, opacity=0.6,
        )
    if client_value is not None and not np.isnan(client_value):
        fig.add_vline(
            x=float(np.clip(client_value, low, high)), line_width=4, line_color=ORANGE,
            annotation_text="Ce client", annotation_font_size=16, annotation_position="top",
        )
    fig.update_layout(
        barmode="overlay", height=380, font=FONT, xaxis_title=title,
        yaxis_title="% de clients", legend={"orientation": "h", "y": 1.12},
        margin={"t": 70, "b": 50},
    )
    return fig
