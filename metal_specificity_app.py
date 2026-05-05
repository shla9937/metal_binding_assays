#!/usr/bin/env python

import base64
import io
import numpy as np
import pandas as pd

import dash
from dash import Dash, dcc, html, Input, Output, State, dash_table
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ------------------------------
# Helpers
# ------------------------------

def _decode_csv(contents: str) -> pd.DataFrame:
    """Base64-decode a Dash upload payload and return a DataFrame."""
    _, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    df.columns = [c.strip() for c in df.columns]
    return df


def parse_kd_results_csv(df_raw: pd.DataFrame, protein_name: str) -> pd.Series:
    """Convert a long-form *_kd_results.csv from dsf_analysis.py into a
    Metal-indexed Series of Kd values in M.

    Selects the Apo-Tm temperature (identified from the 'Apo' metal row).
    Drops Apo and EDTA control rows.  Kd column is in µM; converts to M.
    Returns None if the DataFrame does not look like a kd_results file.
    """
    if not {'Metal', 'Temperature', 'Kd'}.issubset(df_raw.columns):
        return None

    # Use the temperature recorded for the Apo metal row as the analysis temperature
    apo_rows = df_raw[df_raw['Metal'] == 'Apo']
    target_temp = apo_rows['Temperature'].iloc[0] if not apo_rows.empty else df_raw['Temperature'].iloc[0]

    df_filt = df_raw[df_raw['Temperature'] == target_temp].copy()
    df_filt = df_filt[~df_filt['Metal'].isin(['Apo', 'EDTA'])]
    df_filt['Kd_M'] = pd.to_numeric(df_filt['Kd'], errors='coerce') * 1e-6  # µM → M
    df_filt = df_filt[df_filt['Kd_M'].notna()]

    series = df_filt.set_index('Metal')['Kd_M']
    series.name = protein_name
    return series


def parse_contents(contents: str, filename: str):
    """Parse an uploaded CSV.

    Accepts two formats:
    1. Long-form *_kd_results.csv from dsf_analysis.py — returns a pd.Series
       (Metal index, Kd in M) named after the protein.
    2. Wide-form CSV (rows = metals, columns = proteins, values = Kd in µM) —
       returns a pd.DataFrame (Metal index, protein columns, Kd in M).
    """
    df_raw = _decode_csv(contents)

    # --- Long-form kd_results format ---
    if {'Metal', 'Temperature', 'Kd'}.issubset(df_raw.columns):
        protein_name = filename
        for suffix in ('_kd_results.csv', '.csv'):
            if protein_name.endswith(suffix):
                protein_name = protein_name[:-len(suffix)]
                break
        result = parse_kd_results_csv(df_raw, protein_name)
        if result is not None:
            return result

    # --- Wide-form fallback ---
    if df_raw.shape[1] < 2:
        raise ValueError("CSV must have at least two columns.")
    df_raw.rename(columns={df_raw.columns[0]: 'Metal'}, inplace=True)
    df_raw.set_index('Metal', inplace=True)
    df_raw = df_raw.apply(pd.to_numeric, errors='coerce') * 1e-6  # µM → M
    df_raw = df_raw.dropna(axis=0, how='all').dropna(axis=1, how='all')
    if df_raw.empty:
        raise ValueError("No numeric Kd values found after cleaning.")
    return df_raw

def compute_specificity(df: pd.DataFrame):
    """Compute per-protein metrics.
    Returns:
      best_metal: pd.Series of the metal with the smallest Kd per protein
      best_kd: pd.Series of the smallest Kd per protein
      specificity_fold: pd.Series of (2nd-best Kd / best Kd). Higher = more specific. NaN if not computable.
    """
    best_metal = df.idxmin(axis=0, skipna=True)
    best_kd = df.min(axis=0, skipna=True)

    # Compute 2nd-best per column
    specificity_fold = pd.Series(index=df.columns, dtype=float)
    for col in df.columns:
        vals = df[col].dropna().sort_values()
        if len(vals) >= 2:
            best = vals.iloc[0]
            second = vals.iloc[1]
            specificity_fold[col] = (second) / (best)
        else:
            specificity_fold[col] = np.nan

    return best_metal, best_kd, specificity_fold


def build_figure(df: pd.DataFrame):
    """Build a heatmap figure with a bottom row indicating the best-binding metal and specificity for each protein.
    Main heatmap is grayscale, scaled per protein on log10(Kd).
    Hover shows raw Kd with appropriate units.
    """
    # 1. Grayscale: log10(Kd), column-normalized
    z_log = df.copy()
    for col in z_log.columns:
        col_vals = np.log10(z_log[col].values)
        min_val = np.nanmin(col_vals)
        max_val = np.nanmax(col_vals)
        normed = (col_vals - min_val) / (max_val - min_val)
        normed[np.isnan(col_vals)] = np.nan
        z_log[col] = normed
    x_labels = df.columns.astype(str)
    y_labels = df.index.astype(str)
    z_main = z_log.loc[y_labels, x_labels].values

    # 2. Prepare custom hover text with formatted Kd and units
    def format_kd(val):
        if pd.isna(val):
            return "—"
        elif val < 1e-6:
            return f"{val*1e9:.2f} nM"
        elif val < 1e-3:
            return f"{val*1e6:.2f} µM"
        elif val < 1:
            return f"{val*1e3:.2f} mM"
        else:
            return f"{val:.2f} M"

    kd_text = np.empty(df.shape, dtype=object)
    for i, y in enumerate(df.index):
        for j, x in enumerate(df.columns):
            kd_text[i, j] = format_kd(df.loc[y, x])

    # 3. Compute winners and specificity
    best_metal, best_kd, specificity_fold = compute_specificity(df)

    # 4. Bottom categorical strip (as before)
    metals = df.index.astype(str).tolist()
    base_colors = (
        ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
         '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    )
    while len(base_colors) < len(metals):
        base_colors = base_colors + base_colors
    metal_to_color = {m: base_colors[i] for i, m in enumerate(metals)}
    winners = best_metal.astype(str).reindex(df.columns)
    winner_colors = [metal_to_color.get(m, '#CCCCCC') for m in winners]
    metal_to_id = {m: i for i, m in enumerate(metals)}
    z_strip_ids = np.array([[metal_to_id.get(m, -1) for m in winners]])
    max_id = max(metal_to_id.values()) if metal_to_id else 0
    if max_id == 0:
        colorscale = [[0.0, '#1f77b4'], [1.0, '#1f77b4']]
    else:
        colorscale = []
        for m, mid in metal_to_id.items():
            t = mid / max(max_id, 1)
            c = metal_to_color[m]
            colorscale.append([max(0.0, t - 1e-9), c])
            colorscale.append([min(1.0, t + 1e-9), c])

    # 5. Build figure
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02,
        row_heights=[0.88, 0.12]
    )

    heat = go.Heatmap(
        z=z_log.values,
        x=df.columns.astype(str),
        y=df.index.astype(str),
        coloraxis='coloraxis',
        text=kd_text,
        hovertemplate='Protein: %{x}<br>Metal: %{y}<br>Kd: %{text}<extra></extra>',
    )
    fig.add_trace(heat, row=1, col=1)

    # Bottom strip
    strip = go.Heatmap(
        z=z_strip_ids,
        x=df.columns.astype(str),
        y=["Best metal (fold specificity)"]*1,
        colorscale=colorscale,
        showscale=False,
        hoverinfo='skip'
    )
    fig.add_trace(strip, row=2, col=1)

    # Add text labels on the strip: e.g., "Cu\n12.3×"
    strip_text = [
        f"{winners.iloc[i]}\n" + (f"{specificity_fold.iloc[i]:.1f}×" if pd.notna(specificity_fold.iloc[i]) else "—")
        for i in range(len(winners))
    ]
    fig.add_trace(
        go.Scatter(
            x=df.columns.astype(str),
            y=["Best metal (fold specificity)"]*len(df.columns),
            mode='text',
            text=strip_text,
            textposition='middle center',
            hoverinfo='skip',
        ),
        row=2, col=1
    )

    # Layout and color axis for main heatmap
    fig.update_layout(
        coloraxis=dict(
            colorscale='Greys_r',
            colorbar=dict(title='Relative log10(Kd) per protein')
        ),
        margin=dict(l=80, r=20, t=40, b=60),
        height=700,
        xaxis=dict(
            title='Protein',
            side='top',
            tickangle=-90,
            showticklabels=True,
            tickmode='array',
            ticktext=list(df.columns),
            showgrid=False,
        ),
        xaxis2=dict(title=None, showticklabels=False),
        yaxis=dict(
            title='Metal',
            tickmode='array',
            ticktext=list(df.index),
            showgrid=False,
        ),
        yaxis2=dict(title=None),
        template='plotly_white',
    )

    return fig, best_metal, best_kd, specificity_fold


# ------------------------------
# Dash App
# ------------------------------
app = Dash(__name__)
app.title = "Protein–Metal Kd Heatmap"

app.layout = html.Div(
    className='container',
    children=[
        html.H1("Protein–Metal Kd Analyzer"),
        html.P(
            "Upload one or more *_kd_results.csv files from dsf_analysis.py — one file per protein. "
            "Each file becomes one protein column. Apo and EDTA controls are excluded automatically. "
            "Wide-form CSVs (rows = metals, columns = proteins, Kd in µM) are also accepted."
        ),
        dcc.Upload(
            id='upload-data',
            children=html.Div(["Drag and Drop or ", html.A("Select CSV files")]),
            style={
                'width': '100%', 'height': '60px', 'lineHeight': '60px',
                'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
                'textAlign': 'center', 'margin': '10px 0'
            },
            multiple=True
        ),
        html.Div(id='file-info', style={'marginBottom': '10px', 'fontStyle': 'italic'}),
        dcc.Graph(id='heatmap', style={'height': '750px'}),
        html.H3("Per-Protein Summary"),
        dash_table.DataTable(
            id='summary-table',
            columns=[
                {"name": "Protein", "id": "Protein"},
                {"name": "Best metal", "id": "Best metal"},
                {"name": "Best Kd (M)", "id": "Best Kd (M)", 'type': 'numeric', 'format': {'specifier': '.3g'}},
                {"name": "Specificity (2nd/Best, ×)", "id": "Specificity (×)", 'type': 'numeric', 'format': {'specifier': '.2f'}},
            ],
            data=[],
            sort_action='native',
            filter_action='native',
            page_size=20,
            style_table={'overflowX': 'auto'},
        ),
        html.Hr(),
        html.Details([
            html.Summary("Notes on calculations"),
            html.Ul([
                html.Li("The heatmap shows log10(Kd) values (lower is tighter)."),
                html.Li("The bottom strip indicates the tightest-binding metal per protein and the fold specificity, defined as (2nd-best Kd / best Kd). Higher means more selective."),
                html.Li("If a protein has only one non-missing Kd, specificity is left blank (—)."),
            ])
        ])
    ]
)


def _empty_figure(message: str, color: str = 'black') -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        annotations=[dict(text=message, x=0.5, y=0.5, xref='paper', yref='paper',
                          showarrow=False, font=dict(size=14, color=color))],
        xaxis={'visible': False}, yaxis={'visible': False},
        template='plotly_white', height=500
    )
    return fig


@app.callback(
    Output('file-info', 'children'),
    Output('heatmap', 'figure'),
    Output('summary-table', 'data'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_output(contents_list, filenames):
    if not contents_list:
        return ("No files uploaded yet.", _empty_figure("Upload CSV files to see the heatmap"), [])

    # Normalise to lists (Dash passes a scalar when multiple=True but only one file dropped)
    if isinstance(contents_list, str):
        contents_list = [contents_list]
        filenames = [filenames]

    series_list = []
    errors = []

    for contents, filename in zip(contents_list, filenames or []):
        try:
            result = parse_contents(contents, filename or 'uploaded.csv')
            if isinstance(result, pd.Series):
                series_list.append(result)
            elif isinstance(result, pd.DataFrame):
                for col in result.columns:
                    series_list.append(result[col])
        except Exception as e:
            errors.append(f"{filename}: {e}")

    if not series_list:
        msg = "Could not parse any files. " + ("Errors: " + "; ".join(errors) if errors else "")
        return (msg, _empty_figure(msg, color='crimson'), [])

    df = pd.concat(series_list, axis=1)
    fig, best_metal, best_kd, specificity = build_figure(df)

    summary = pd.DataFrame({
        'Protein': df.columns,
        'Best metal': best_metal.values,
        'Best Kd (M)': best_kd.values,
        'Specificity (×)': specificity.values,
    })

    loaded_names = list(df.columns)
    info = f"Loaded {len(loaded_names)} protein(s): {', '.join(loaded_names)}. {df.shape[0]} metals."
    if errors:
        info += "  Errors: " + "; ".join(errors)

    return (info, fig, summary.to_dict('records'))


if __name__ == '__main__':
    app.run(debug=True)
