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

def parse_contents(contents: str, filename: str) -> pd.DataFrame:
    """Parse uploaded CSV into a tidy DataFrame with Metals as index and Proteins as columns.
    The first column is interpreted as the metal name/index.
    """
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

    # Use the first column as index (metals)
    if df.shape[1] < 2:
        raise ValueError("CSV must have at least two columns: a 'Metal' column and one or more protein columns.")

    df.rename(columns={df.columns[0]: 'Metal'}, inplace=True)
    df.set_index('Metal', inplace=True)

    # Coerce to numeric (Kd values); non-numeric become NaN
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df * 1e-6 # delete if we change the Kd csv to M
    df = df.dropna(axis=0, how='all').dropna(axis=1, how='all')
    if df.empty:
        raise ValueError("No numeric Kd values found after cleaning.")
    return df

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
        normed[np.isnan(col_vals)] = np.nan  # <-- preserve NaNs
        z_log[col] = normed
    n_cols = len(df.columns)
    n_rows = len(df.index)
    x_vals = np.arange(n_cols)
    y_vals = np.arange(n_rows)
    x_labels = df.columns.astype(str)
    y_labels = df.index.astype(str)
    z_main = z_log.loc[y_labels, x_labels].values  # ensure order

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

    # Heatmap with integer positions
    heat = go.Heatmap(
        z=z_main,
        x=x_labels,
        y=y_labels,
        coloraxis='coloraxis',
        text=kd_text,
        hovertemplate='Protein: %{x}<br>Metal: %{y}<br>Kd: %{text}<extra></extra>',
    )
    fig.add_trace(heat, row=1, col=1)
    strip = go.Heatmap(
        z=z_strip_ids,
        x=x_vals,
        y=[0],  # place below the last row
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
    # Add text labels on the strip (update x to integer positions)
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=[0]*n_cols,
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
            tickvals=(x_vals + 0.5).tolist(),
            ticktext=x_labels.tolist(),
            showgrid=True,
            gridwidth=0,
            dtick=1,
            zeroline=False,
        ),
        xaxis2=dict(title=None, showticklabels=False),
        yaxis=dict(
            title='Metal',
            tickmode='array',
            tickvals=(y_vals + 0.5).tolist(),
            ticktext=y_labels.tolist(),
            showgrid=True,
            gridwidth=0,
            dtick=1,
            zeroline=False,
        ),
        yaxis2=dict(title=None, ticks=""),
        template='plotly_white',
    )

    fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(0,0,0,0.05)')
    fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(0,0,0,0.05)')

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
            "Upload a CSV with Metals as rows and Proteins as columns (first column is metal names). "
            "Cells should be Kd values in molar units (smaller = tighter)."
        ),
        dcc.Upload(
            id='upload-data',
            children=html.Div(["Drag and Drop or ", html.A("Select CSV")]),
            style={
                'width': '100%', 'height': '60px', 'lineHeight': '60px',
                'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
                'textAlign': 'center', 'margin': '10px 0'
            },
            multiple=False
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


@app.callback(
    Output('file-info', 'children'),
    Output('heatmap', 'figure'),
    Output('summary-table', 'data'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)

def update_output(contents, filename):
    if contents is None:
        # Show an empty placeholder figure
        placeholder = go.Figure()
        placeholder.update_layout(
            annotations=[dict(
                text="Upload a CSV to see the heatmap",
                x=0.5, y=0.5, xref='paper', yref='paper', showarrow=False,
                font=dict(size=16)
            )],
            xaxis={'visible': False}, yaxis={'visible': False},
            template='plotly_white', height=500
        )
        return ("No file uploaded yet.", placeholder, [])

    try:
        df = parse_contents(contents, filename or 'uploaded.csv')
        fig, best_metal, best_kd, specificity = build_figure(df)

        summary = pd.DataFrame({
            'Protein': df.columns,
            'Best metal': best_metal.values,
            'Best Kd (M)': best_kd.values,
            'Specificity (×)': specificity.values,
        })

        return (f"Loaded: {filename} — shape {df.shape[0]} metals × {df.shape[1]} proteins.",
                fig,
                summary.to_dict('records')) 

    except Exception as e:
        # Error figure
        err_fig = go.Figure()
        err_fig.update_layout(
            annotations=[dict(
                text=f"Error: {str(e)}",
                x=0.5, y=0.5, xref='paper', yref='paper', showarrow=False,
                font=dict(size=14, color='crimson')
            )],
            xaxis={'visible': False}, yaxis={'visible': False},
            template='plotly_white', height=400
        )
        return (f"Failed to load: {filename}", err_fig, [])


if __name__ == '__main__':
    app.run(debug=True)
