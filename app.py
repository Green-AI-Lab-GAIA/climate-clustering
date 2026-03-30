import sys
import streamlit as st
import pandas as pd
import numpy as np
import torch
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import os

if '..' not in sys.path:
    sys.path.insert(0, '.')

import src.inference as inf
from src.el_nino import read_enso_data

st.set_page_config(page_title="Climate Teleconnections", layout="wide")
st.title("Climate Regimes & ENSO Teleconnections")

# ── Data loading (cached) ───────────────────────────────────────────────────
CONFIG_DIR = "checkpoint/temperature-run2"
RESULTS_DIR = "results"

config_files = [f for f in os.listdir(CONFIG_DIR) if f.endswith(".yaml")]
config_file = os.path.join(CONFIG_DIR, config_files[0])

result_dirs = [d for d in os.listdir(RESULTS_DIR) if os.path.isdir(os.path.join(RESULTS_DIR, d))]
save_path = os.path.join(RESULTS_DIR, result_dirs[0])

VALIDATION = True


@st.cache_resource(show_spinner="Loading dataset and model...")
def load_all(config_file, save_path, validation):
    import yaml
    with open(config_file, 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    params_data, dataset = inf.read_data(config_file, validation=validation)
    target_encoder, prot = inf.load_model(params)

    E, F = inf.get_model_results(
        read_path=save_path, params=params, dataset=dataset,
        encoder=target_encoder, prototypes=prot,
    )
    E_val, F_val = None, None
    if validation:
        E_val, F_val = inf.get_model_results(
            read_path=save_path, params=params, dataset=dataset.validation_imgs,
            encoder=target_encoder, prototypes=prot, validation=True,
        )

    tsne_E, tsne_prot, tsne_Eval = inf.get_TSNE(
        save_path, E=E.cpu(), prot=prot.cpu(),
        E_val=E_val.cpu() if validation and E_val is not None else None,
        validation=validation,
    )

    cluster_prob, cluster_id = torch.max(F, dim=1)
    df = pd.DataFrame({
        "cluster_id": cluster_id.cpu().numpy(),
        "cluster_prob": cluster_prob.cpu().numpy(),
        "sample_type": "train",
        "date": dataset.time,
    })

    if validation:
        val_cluster_prob, val_cluster_id = torch.max(F_val, dim=1)
        df_val = pd.DataFrame({
            "cluster_id": val_cluster_id.cpu().numpy(),
            "cluster_prob": val_cluster_prob.cpu().numpy(),
            "sample_type": "val",
            "date": dataset.val_time,
        })
        df = pd.concat([df, df_val], ignore_index=True)

    df = df.sort_values('date').reset_index(drop=True)

    # Combined dataset tensor (for sample visualisation)
    train_imgs = dataset[:][0] if hasattr(dataset, '__getitem__') else dataset.imgs
    if validation and dataset.validation_imgs is not None:
        combined = torch.cat((train_imgs, dataset.validation_imgs))
    else:
        combined = train_imgs
    combined = combined[:len(df)]

    return params, df, combined, tsne_E, tsne_prot, tsne_Eval, F, F_val


@st.cache_data(show_spinner="Loading ONI index...")
def load_oni(path=None):
    import os
    from src.el_nino import read_enso_data

    if path is None:
        candidates = [
            '../data/oni_index.xlsx',
            'data/oni_index.xlsx',
            os.path.join(os.path.dirname(__file__), '..', 'data', 'oni_index.xlsx')
        ]

        for candidate in candidates:
            if os.path.exists(candidate):
                path = candidate
                break

    if path is None or not os.path.exists(path):
        raise FileNotFoundError("ONI index file not found in expected locations.")

    return read_enso_data(path)


# ── Load ─────────────────────────────────────────────────────────────────────
try:
    params, df, combined_dataset, tsne_E, tsne_prot, tsne_Eval, F, F_val = load_all(
        config_file, save_path, VALIDATION
    )
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

oni_index = load_oni()

vars_names = params['data']['surf_vars']
norm_means = params['data']['norm_means']
norm_stds = params['data']['norm_stds']
nvars = len(vars_names)

df['date'] = pd.to_datetime(df['date'])
df['date_period'] = df['date'].dt.to_period('M')
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year

# Compute per-sample average temperatures (denormalized)
for vi, vname in enumerate(vars_names):
    df[f'Average {vname}'] = (combined_dataset[:, vi, :, :].mean(dim=(1, 2)) * norm_stds[vi] + norm_means[vi]).cpu().numpy()[:len(df)]


def get_season(date):
    m, d = date.month, date.day
    if (m == 12 and d >= 21) or m in (1, 2) or (m == 3 and d <= 20):
        return 'Summer'
    elif (m == 3 and d >= 21) or m in (4, 5) or (m == 6 and d <= 20):
        return 'Autumn'
    elif (m == 6 and d >= 21) or m in (7, 8) or (m == 9 and d <= 22):
        return 'Winter'
    else:
        return 'Spring'


df['season'] = df['date'].apply(get_season)

# Seasonal grouping of clusters
seasonal_dist = pd.crosstab(df['cluster_id'], df['season'], normalize='index')
seasonal_dist['dominant_season'] = seasonal_dist.idxmax(axis=1)
grupos_map_season = (
    seasonal_dist.reset_index()
    .groupby('dominant_season')
    .apply(lambda x: x['cluster_id'].tolist(), include_groups=False)
    .to_dict()
)

SEASON_OPTIONS = {
    "DJF (Summer)": [12, 1, 2],
    "MAM (Autumn)": [3, 4, 5],
    "JJA (Winter)": [6, 7, 8],
    "SON (Spring)": [9, 10, 11],
}

# Merge ENSO labels onto full df (used by both pages)
df_el_nino = df.merge(oni_index, left_on='date_period', right_index=True, how='left')
df_el_nino['month'] = df_el_nino['date'].dt.month
df_el_nino['year'] = df_el_nino['date'].dt.year
df_el_nino['season'] = df['season']
# for vi, vname in enumerate(vars_names):
#     df_el_nino[f'Average {vname}'] = df[f'Average {vname}']

# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════

st.sidebar.header("Navigation")

page = st.sidebar.radio(
    "Page", ["Climate Regimes Exploration", "ENSO Teleconnections",  "Time Series"], key="page_nav"
)

st.sidebar.markdown("---")
st.sidebar.header("Analysis Configuration")

comparison_mode = st.sidebar.radio(
    "Comparison mode",
    ["ENSO Regimes (Nino vs Neutral)", "Nino vs Climatology"],
    help=(
        "**ENSO Regimes**: P(k | Nino) - P(k | Neutral)\n\n"
        "**Nino vs Climatology**: P(k | Nino, month) - P(k | month)"
    ),
)

st.sidebar.markdown("---")
st.sidebar.subheader("Date Filter")

min_year = int(df['year'].min())
max_year = int(df['year'].max())
default_start = max(min_year, max_year - 30)

use_filter = st.sidebar.checkbox("Filter date range", value=False)
if use_filter:
    start_year, end_year = st.sidebar.slider(
        "Year range", min_year, max_year, (default_start, max_year)
    )
else:
    start_year, end_year = min_year, max_year

# Apply date filter
df_filtered = df[(df['year'] >= start_year) & (df['year'] <= end_year)].copy()

st.sidebar.caption(f"Using {len(df_filtered)} samples ({start_year}–{end_year})")

# Merge ENSO labels onto filtered data
df_enso = df_filtered.merge(oni_index, left_on='date_period', right_index=True, how='left')


# ═════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

def compute_anomaly_bar(df_enso, mode):
    """Compute anomaly per cluster: bar chart data."""
    if mode == "ENSO Regimes (Nino vs Neutral)":
        cond_prob = pd.crosstab(df_enso['Label'], df_enso['cluster_id'], normalize='index') * 100
        if "El Niño" not in cond_prob.index or "Neutro" not in cond_prob.index:
            return None
        anomaly = cond_prob.loc["El Niño"] - cond_prob.loc["Neutro"]
        title = "P(k | El Nino) - P(k | Neutral)"
    else:
        p_clim = df_enso['cluster_id'].value_counts(normalize=True) * 100
        df_nino = df_enso[df_enso['Label'] == 'El Niño']
        if df_nino.empty:
            return None
        p_nino = df_nino['cluster_id'].value_counts(normalize=True) * 100
        all_clusters = sorted(df_enso['cluster_id'].unique())
        anomaly = pd.Series(
            [p_nino.get(c, 0) - p_clim.get(c, 0) for c in all_clusters],
            index=all_clusters,
        )
        title = "P(k | El Nino) - P(k | Climatology)"
    return anomaly, title

def compute_lagged_anomalies(df_base, oni_index, mode):
    """Compute lagged anomalies across lags -12 to 12."""
    anomalias_list = {}
    df_base = df_base.copy()
    df_base['date_period'] = df_base['date'].dt.to_period('M')

    for lag in range(-12, 13):
        df_lag = df_base.merge(
            oni_index.shift(lag).dropna(),
            left_on='date_period', right_index=True, how='left',
            suffixes=('_orig', ''),
        )
        if mode == "ENSO Regimes (Nino vs Neutral)":
            cp = pd.crosstab(df_lag['Label'], df_lag['cluster_id'], normalize='index') * 100
            if "El Niño" in cp.index and "Neutro" in cp.index:
                anomalias_list[lag] = cp.loc["El Niño"] - cp.loc["Neutro"]
        else:
            p_clim = df_lag['cluster_id'].value_counts(normalize=True) * 100
            df_nino = df_lag[df_lag['Label'] == 'El Niño']
            if not df_nino.empty:
                p_nino = df_nino['cluster_id'].value_counts(normalize=True) * 100
                all_clusters = sorted(df_lag['cluster_id'].unique())
                anomalias_list[lag] = pd.Series(
                    [p_nino.get(c, 0) - p_clim.get(c, 0) for c in all_clusters],
                    index=all_clusters,
                )

    if not anomalias_list:
        return None
    anom = pd.DataFrame(anomalias_list).sort_index(axis=1)
    anom.columns.name = 'Lag'
    return anom


def compute_heatmap_per_cluster(df_base, oni_index, cluster_id, mode):
    """Compute month x lag heatmap for a single cluster."""
    df_base = df_base.copy()
    df_base['date_period'] = df_base['date'].dt.to_period('M')
    climate_anon = {}

    for month in range(1, 13):
        for lag in range(-12, 13):
            df_lag = df_base.merge(
                oni_index.shift(lag).dropna(),
                left_on='date_period', right_index=True, how='left',
                suffixes=('_orig', ''),
            )

            if mode == "ENSO Regimes (Nino vs Neutral)":
                cur_nino = df_lag[(df_lag['Label'] == "El Niño") & (df_lag['month'] == month)]
                cur_neutral = df_lag[(df_lag['Label'] == "Neutro") & (df_lag['month'] == month)]
                p_nino = (cur_nino['cluster_id'] == cluster_id).sum() / len(cur_nino) if len(cur_nino) > 0 else 0
                p_neutral = (cur_neutral['cluster_id'] == cluster_id).sum() / len(cur_neutral) if len(cur_neutral) > 0 else 0
                climate_anon[(month, lag)] = (p_nino - p_neutral) * 100
            else:
                cur_nino = df_lag[(df_lag['Label'] == "El Niño") & (df_lag['month'] == month)]
                cur_all = df_lag[df_lag['month'] == month]
                p_nino = (cur_nino['cluster_id'] == cluster_id).sum() / len(cur_nino) if len(cur_nino) > 0 else 0
                p_clim = (cur_all['cluster_id'] == cluster_id).sum() / len(cur_all) if len(cur_all) > 0 else 0
                climate_anon[(month, lag)] = (p_nino - p_clim) * 100

    climate_df = pd.DataFrame.from_dict(climate_anon, orient='index', columns=['Anomaly'])
    climate_df.index = pd.MultiIndex.from_tuples(climate_df.index, names=['Month', 'Lag'])
    climate_df = climate_df.reset_index()
    return climate_df.pivot(index='Month', columns='Lag', values='Anomaly')


def get_top_80_clusters(df_cond):
    """Get clusters that sum to 80% of the probability mass."""
    dist = df_cond['cluster_id'].value_counts(normalize=True)
    cumsum = dist.cumsum()
    top_clusters = []
    for cluster, cum_val in cumsum.items():
        top_clusters.append(cluster)
        if cum_val >= 0.8:
            break
    return top_clusters, dist


@st.cache_data(show_spinner="Computing quantile deviations...")
def compute_quantile_deviations(_combined_dataset, _df_cluster_ids, _grupos_map_season, _nvars, n_quantiles=200):
    """Compute quantile deviations for all season groups. Returns dict of season -> list of traces data."""
    q = np.linspace(0.01, 0.99, n_quantiles)
    result = {}
    for season_name, group in _grupos_map_season.items():
        season_data = []
        for var in range(_nvars):
            global_data = _combined_dataset[:, var, :, :].ravel().numpy()
            global_q = np.quantile(global_data, q)
            for g in group:
                idx = _df_cluster_ids[_df_cluster_ids == g].index.values
                data = _combined_dataset[:, var, :, :][idx].ravel().numpy()
                qg = np.quantile(data, q)
                season_data.append({
                    'var': var, 'cluster': g,
                    'q': q.tolist(), 'delta': (qg - global_q).tolist(),
                })
        result[season_name] = season_data
    return result


@st.cache_data(show_spinner="Computing monthly frequencies...")
def compute_monthly_freq(_df_month, _df_cluster_id):
    """Compute monthly frequency crosstab."""
    return pd.crosstab(_df_month, _df_cluster_id, normalize='index') * 100


@st.cache_data(show_spinner="Computing windowed anomalies...")
def compute_windowed_anomalies(_df_el_nino_years, _df_el_nino_labels, _df_el_nino_clusters, min_yr, max_yr, window_size, mode):
    """Compute anomalies per time window."""
    all_vals = []
    window_labels = []
    for start_yr in range(min_yr, max_yr, window_size):
        mask = (_df_el_nino_years >= start_yr) & (_df_el_nino_years < start_yr + window_size)
        cur_labels = _df_el_nino_labels[mask]
        cur_clusters = _df_el_nino_clusters[mask]
        if len(cur_labels) == 0:
            continue

        if mode == "ENSO Regimes (Nino vs Neutral)":
            cp = pd.crosstab(cur_labels, cur_clusters, normalize='index') * 100
            if not {"El Niño", "Neutro"}.issubset(cp.index):
                continue
            anomaly = cp.loc["El Niño"] - cp.loc["Neutro"]
        else:
            p_clim = cur_clusters.value_counts(normalize=True) * 100
            nino_mask = cur_labels == 'El Niño'
            if nino_mask.sum() == 0:
                continue
            p_nino = cur_clusters[nino_mask].value_counts(normalize=True) * 100
            all_c = sorted(cur_clusters.unique())
            anomaly = pd.Series([p_nino.get(c, 0) - p_clim.get(c, 0) for c in all_c], index=all_c)

        all_vals.append(anomaly)
        cur_years = _df_el_nino_years[mask]
        end_yr = min(start_yr + window_size - 1, int(cur_years.max()))
        window_labels.append(f"{start_yr}-{end_yr}")

    if not all_vals:
        return None
    return pd.DataFrame(all_vals, index=window_labels)


@st.cache_data(show_spinner="Computing epoch comparison...")
def compute_epoch_comparison(_df_years, _df_labels, _df_clusters, cutoff_year, mode):
    """Compute epoch-based anomaly comparison."""
    all_clusters_sorted = sorted(_df_clusters.unique())
    epochs = {
        f"Past (<={cutoff_year})": _df_years <= cutoff_year,
        f"Present (>{cutoff_year})": _df_years > cutoff_year,
    }
    results = []
    for epoch_name, mask in epochs.items():
        epoch_clusters = _df_clusters[mask]
        epoch_labels = _df_labels[mask]

        if mode == "ENSO Regimes (Nino vs Neutral)":
            p_base = epoch_clusters[epoch_labels == 'Neutro'].value_counts(normalize=True)
        else:
            p_base = epoch_clusters.value_counts(normalize=True)

        p_nino = epoch_clusters[epoch_labels == 'El Niño'].value_counts(normalize=True)

        for cluster in all_clusters_sorted:
            results.append({
                'Epoch': epoch_name, 'Cluster': cluster,
                'Delta_P': p_nino.get(cluster, 0) - p_base.get(cluster, 0),
            })
    return pd.DataFrame(results)


def compute_all_clusters_heatmap(df_base, oni_index, mode, lag):
    """Compute cluster x month heatmap for a given lag."""
    df_base = df_base.copy()
    df_base['date_period'] = df_base['date'].dt.to_period('M')
    all_clusters = sorted(df_base['cluster_id'].unique())

    df_lag = df_base.merge(
        oni_index.shift(lag).dropna(),
        left_on='date_period', right_index=True, how='left',
        suffixes=('_orig', ''),
    )

    results = {}
    for month in range(1, 13):
        for cluster_id in all_clusters:
            if mode == "ENSO Regimes (Nino vs Neutral)":
                cur_nino = df_lag[(df_lag['Label'] == "El Niño") & (df_lag['month'] == month)]
                cur_neutral = df_lag[(df_lag['Label'] == "Neutro") & (df_lag['month'] == month)]
                p_nino = (cur_nino['cluster_id'] == cluster_id).sum() / len(cur_nino) if len(cur_nino) > 0 else 0
                p_neutral = (cur_neutral['cluster_id'] == cluster_id).sum() / len(cur_neutral) if len(cur_neutral) > 0 else 0
                results[(cluster_id, month)] = (p_nino - p_neutral) * 100
            else:
                cur_nino = df_lag[(df_lag['Label'] == "El Niño") & (df_lag['month'] == month)]
                cur_all = df_lag[df_lag['month'] == month]
                p_nino = (cur_nino['cluster_id'] == cluster_id).sum() / len(cur_nino) if len(cur_nino) > 0 else 0
                p_clim = (cur_all['cluster_id'] == cluster_id).sum() / len(cur_all) if len(cur_all) > 0 else 0
                results[(cluster_id, month)] = (p_nino - p_clim) * 100

    result_df = pd.DataFrame.from_dict(results, orient='index', columns=['Anomaly'])
    result_df.index = pd.MultiIndex.from_tuples(result_df.index, names=['Cluster', 'Month'])
    result_df = result_df.reset_index()
    return result_df.pivot(index='Cluster', columns='Month', values='Anomaly')


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: EXPLORATION
# ═════════════════════════════════════════════════════════════════════════════

if page == "Exploration":
    st.header("Latent Space Visualisation (t-SNE)")

    color_options = ["Cluster ID", "Season", "ENSO Label", "ONI Value"] + [
        f"Average {v}" for v in vars_names
    ]
    color_option = st.radio(
        "Color by", color_options, horizontal=True, key="tsne_color"
    )

    season_colors = {
        "Summer": "#d6092b", "Autumn": "#d66f09",
        "Winter": "#14a1cc", "Spring": "#16b548",
    }

    # Build t-SNE dataframe (train samples)
    cluster_labels = F.argmax(axis=1).cpu().numpy()
    n_train = len(tsne_E)

    df_train = df[df['sample_type'] == 'train'].reset_index(drop=True)
    df_train_oni = df_el_nino[df_el_nino['sample_type'] == 'train'].reset_index(drop=True)

    df_tsne = pd.DataFrame({
        "x": np.array(tsne_E)[:, 0],
        "y": np.array(tsne_E)[:, 1],
        "cluster_id": cluster_labels.astype(str),
        "season": df_train['season'].values[:n_train],
        "Label": df_train_oni['Label'].values[:n_train],
        "ONI": df_train_oni['ONI'].values[:n_train],
    })
    for vi, vname in enumerate(vars_names):
        df_tsne[f"Average {vname}"] = df_train[f"Average {vname}"].values[:n_train]

    if color_option == "Cluster ID":
        fig = px.scatter(df_tsne, x="x", y="y", color="cluster_id",
                         opacity=0.3, hover_data=["cluster_id"],
                         title="t-SNE — Clusters")
    elif color_option == "Season":
        fig = px.scatter(df_tsne, x="x", y="y", color="season",
                         opacity=0.3, title="t-SNE — Seasons",
                         color_discrete_map=season_colors)
    elif color_option == "ENSO Label":
        fig = px.scatter(df_tsne, x="x", y="y", color="Label",
                         opacity=0.3, title="t-SNE — ENSO Phase",
                         color_discrete_map={"El Niño": "red", "La Niña": "blue",
                                             "Neutro": "gray"})
    elif color_option == "ONI Value":
        fig = px.scatter(df_tsne, x="x", y="y", color="ONI",
                         opacity=0.3, title="t-SNE — ONI Value",
                         color_continuous_scale="RdBu_r")
    else:
        fig = px.scatter(df_tsne, x="x", y="y", color=color_option,
                         opacity=0.5, title=f"t-SNE — {color_option}",
                         color_continuous_scale="coolwarm")

    # Add prototypes
    prot_df = pd.DataFrame({
        "x": np.array(tsne_prot)[:, 0],
        "y": np.array(tsne_prot)[:, 1],
        "proto": [str(i) for i in range(len(tsne_prot))],
    })
    fig.add_trace(go.Scatter(
        x=prot_df["x"], y=prot_df["y"], mode="markers+text",
        marker=dict(size=12, color="black", symbol="diamond"),
        text=prot_df["proto"], textposition="top center",
        name="Prototypes", showlegend=True,
    ))
    fig.update_layout(height=800, xaxis_title="t-SNE 1", yaxis_title="t-SNE 2")
    st.plotly_chart(fig, use_container_width=True)

    # ── Quantile Deviation ───────────────────────────────────────────────────
    st.header("Quantile Deviation per Season Group")

    quantile_data = compute_quantile_deviations(
        combined_dataset, df['cluster_id'], grupos_map_season, nvars
    )

    linestyles = ['solid', 'dash', 'dot', 'dashdot']
    cols_q = st.columns(2)
    for ax_id, (season_name, group) in enumerate(grupos_map_season.items()):
        with cols_q[ax_id % 2]:
            fig_q = go.Figure()
            palette = sns.color_palette("bright", n_colors=len(group))

            for trace in quantile_data[season_name]:
                gi = group.index(trace['cluster'])
                color_rgb = palette[gi]
                color_str = f"rgb({int(color_rgb[0]*255)},{int(color_rgb[1]*255)},{int(color_rgb[2]*255)})"
                fig_q.add_trace(go.Scatter(
                    x=trace['q'], y=trace['delta'],
                    mode='lines', name=f"C{trace['cluster']}" if trace['var'] == 0 else None,
                    line=dict(color=color_str, dash=linestyles[trace['var'] % len(linestyles)]),
                    opacity=0.6, showlegend=(trace['var'] == 0),
                    legendgroup=str(trace['cluster']),
                ))

            for var in range(nvars):
                fig_q.add_trace(go.Scatter(
                    x=[None], y=[None], mode='lines',
                    line=dict(color='black', dash=linestyles[var % len(linestyles)], width=2),
                    name=vars_names[var], showlegend=True,
                ))

            fig_q.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
            fig_q.update_layout(
                title=f"{season_name} Clusters",
                xaxis_title="Quantile",
                yaxis_title="Delta Quantile (Cluster - Global)",
                height=400,
            )
            st.plotly_chart(fig_q, use_container_width=True)

    # ── Monthly Frequency Distribution ───────────────────────────────────────
    st.header("Monthly Frequency Distribution")

    month_freq = compute_monthly_freq(df['month'], df['cluster_id'])

    cols_mf = st.columns(2)
    for ax_id, (season_name, group) in enumerate(grupos_map_season.items()):
        with cols_mf[ax_id % 2]:
            fig_mf = go.Figure()
            palette = sns.color_palette("bright", n_colors=len(group))

            for gi, g in enumerate(group):
                if g in month_freq.columns:
                    color_rgb = palette[gi]
                    color_str = f"rgb({int(color_rgb[0]*255)},{int(color_rgb[1]*255)},{int(color_rgb[2]*255)})"
                    fig_mf.add_trace(go.Scatter(
                        x=list(range(1, 13)), y=month_freq[g].values,
                        mode='lines+markers', name=f"C{g}",
                        line=dict(color=color_str), opacity=0.6,
                    ))

            fig_mf.update_layout(
                title=f"{season_name} Clusters",
                xaxis_title="Month", yaxis_title="Frequency (%)",
                xaxis=dict(tickmode='linear', tick0=1, dtick=1),
                height=400,
            )
            st.plotly_chart(fig_mf, use_container_width=True)

    st.stop()

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: TIME SERIES
# ═════════════════════════════════════════════════════════════════════════════

if page == "Time Series":

    # ── Rolling Averages ─────────────────────────────────────────────────────
    st.header("Cluster Occurrence Trend")

    all_clusters = sorted(df['cluster_id'].unique())
    selected_cluster = st.selectbox("Select Cluster", all_clusters, key="trend_cluster")

    df_trend = df_el_nino.copy().sort_values('date').set_index('date')
    df_trend['is_target'] = (df_trend['cluster_id'] == selected_cluster).astype(int)

    df_monthly = df_trend.resample('MS').agg({
        'is_target': 'mean',
        'Label': 'first',
    })
    df_monthly['MA_6m'] = df_monthly['is_target'].rolling(window=6).mean()
    df_monthly['MA_12m'] = df_monthly['is_target'].rolling(window=12).mean()
    df_monthly['MA_24m'] = df_monthly['is_target'].rolling(window=24).mean()

    fig_trend = go.Figure()

    # El Nino shading
    is_nino = df_monthly['Label'] == 'El Niño'
    nino_periods = []
    in_nino = False
    start_dt = None
    for dt, val in is_nino.items():
        if val and not in_nino:
            start_dt = dt
            in_nino = True
        elif not val and in_nino:
            nino_periods.append((start_dt, dt))
            in_nino = False
    if in_nino:
        nino_periods.append((start_dt, df_monthly.index[-1]))

    for s, e in nino_periods:
        fig_trend.add_vrect(
            x0=s, x1=e, fillcolor="red", opacity=0.1, line_width=0,
            annotation_text="El Nino" if s == nino_periods[0][0] else None,
        )

    fig_trend.add_trace(go.Scatter(
        x=df_monthly.index, y=df_monthly['MA_6m'],
        name='Short-term (6mo)', line=dict(color='blue', width=1), opacity=0.5))
    fig_trend.add_trace(go.Scatter(
        x=df_monthly.index, y=df_monthly['MA_12m'],
        name='Annual (12mo)', line=dict(color='darkblue', width=2)))
    fig_trend.add_trace(go.Scatter(
        x=df_monthly.index, y=df_monthly['MA_24m'],
        name='Long-term (24mo)', line=dict(color='darkred', width=2.5)))

    fig_trend.update_layout(
        title=f"Cluster {selected_cluster} Occurrence vs El Nino Events",
        xaxis_title="Year", yaxis_title="Relative Frequency", height=500,
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    # ── Time Windows Anomalies ───────────────────────────────────────────────
    st.header("Anomalies Across Time Windows")

    window_size = st.slider("Window size (years)", 5, 20, 11, key="window_size")

    mode_label_ts = comparison_mode
    min_yr = int(df_el_nino['year'].min())
    max_yr = int(df_el_nino['year'].max())

    df_anomaly_time = compute_windowed_anomalies(
        df_el_nino['year'], df_el_nino['Label'], df_el_nino['cluster_id'],
        min_yr, max_yr, window_size, mode_label_ts,
    )

    if df_anomaly_time is None:
        st.warning("Not enough data for windowed analysis.")
    else:

        n_windows = len(df_anomaly_time)
        n_grid_cols = min(3, n_windows)
        n_grid_rows = int(np.ceil(n_windows / n_grid_cols))

        fig_win, axes_win = plt.subplots(
            n_grid_rows, n_grid_cols,
            figsize=(5 * n_grid_cols, 4 * n_grid_rows),
            sharey=True, sharex=True, squeeze=False,
        )

        for i in range(n_windows):
            ax = axes_win[i // n_grid_cols][i % n_grid_cols]
            vals = df_anomaly_time.iloc[i]
            colors = ['darkblue' if v >= 0 else 'darkred' for v in vals.values]
            ax.bar(vals.index.astype(str), vals.values, color=colors)
            ax.axhline(0, color='black', linestyle='--', alpha=0.7)
            ax.set_title(df_anomaly_time.index[i], fontsize=10)
            ax.tick_params(axis='x', labelsize=7, rotation=90)

        # Remove empty subplots
        for j in range(n_windows, n_grid_rows * n_grid_cols):
            fig_win.delaxes(axes_win[j // n_grid_cols][j % n_grid_cols])

        if mode_label_ts == "ENSO Regimes (Nino vs Neutral)":
            suptitle = "Anomalies: P(k | El Nino) - P(k | Neutral)"
        else:
            suptitle = "Anomalies: P(k | El Nino) - P(k | Climatology)"

        fig_win.suptitle(suptitle, fontsize=14)
        fig_win.supxlabel("Cluster ID")
        fig_win.supylabel("Anomaly (%)")
        plt.tight_layout()
        st.pyplot(fig_win)
        plt.close(fig_win)

    # ── Epoch Comparison ─────────────────────────────────────────────────────
    st.header("El Nino Signal: Past vs Present")

    cutoff_year = st.slider("Epoch cutoff year", min_yr + 5, max_yr - 5, 2000, key="epoch_cutoff")

    df_epochs = compute_epoch_comparison(
        df_el_nino['year'], df_el_nino['Label'], df_el_nino['cluster_id'],
        cutoff_year, mode_label_ts,
    )

    fig_epoch = px.bar(
        df_epochs, x='Cluster', y='Delta_P', color='Epoch',
        barmode='group',
        title=f"El Nino Signal by Epoch (cutoff: {cutoff_year})",
        color_discrete_sequence=['#4575b4', '#d73027'],
    )
    fig_epoch.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
    fig_epoch.update_layout(
        xaxis_title="Cluster ID", yaxis_title="Anomaly (Delta P)", height=500,
    )
    st.plotly_chart(fig_epoch, use_container_width=True)

    st.stop()

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: ENSO TELECONNECTIONS
# ═════════════════════════════════════════════════════════════════════════════

mode_label = comparison_mode
if mode_label == "ENSO Regimes (Nino vs Neutral)":
    subtitle = "Comparing P(k | El Nino) vs P(k | Neutral)"
else:
    subtitle = "Comparing P(k | El Nino) vs P(k | Climatology)"

st.caption(f"{subtitle} | Period: {start_year}–{end_year}")

tab_anom, tab_lag, tab_heatmap, tab_season = st.tabs([
    "1. Anomalies",
    "2. Lagged Anomalies",
    "3. Lagged Heatmaps (by month)",
    "4. Season Distributions",
])

# ── Tab 1: Anomalies ────────────────────────────────────────────────────────
with tab_anom:
    st.header("Cluster Probability Anomalies")

    result = compute_anomaly_bar(df_enso, mode_label)
    if result is None:
        st.warning("Not enough data for both ENSO phases in the selected period.")
    else:
        anomaly, title = result
        colors = ["#1f77b4" if v >= 0 else "#d62728" for v in anomaly.values]

        fig_anom = go.Figure(go.Bar(
            x=[str(c) for c in anomaly.index],
            y=anomaly.values,
            marker_color=colors,
        ))
        fig_anom.add_hline(y=0, line_dash="dash", line_color="black")
        fig_anom.update_layout(
            title=title,
            xaxis_title="Cluster ID",
            yaxis_title="Delta P (%)",
            height=450,
        )
        st.plotly_chart(fig_anom, use_container_width=True)

# ── Tab 2: Lagged Anomalies ─────────────────────────────────────────────────
with tab_lag:
    st.header("Lagged Anomalies by Season Group")

    anom = compute_lagged_anomalies(df_filtered, oni_index, mode_label)

    if anom is None:
        st.warning("Not enough data for lagged analysis in the selected period.")
    else:
        cols = st.columns(2)
        for idx, (season, cur_clusters) in enumerate(grupos_map_season.items()):
            with cols[idx % 2]:
                fig_lag = go.Figure()
                for c in cur_clusters:
                    if c in anom.index:
                        fig_lag.add_trace(go.Scatter(
                            x=list(anom.columns), y=anom.loc[c].values,
                            mode='lines+markers', name=f"Cluster {c}",
                        ))
                fig_lag.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.3)
                fig_lag.add_vline(x=0, line_dash="dash", line_color="black", opacity=0.3)
                fig_lag.update_layout(
                    title=f"{season} Clusters",
                    xaxis_title="ENSO Lag (months)",
                    yaxis_title="Anomaly (%)",
                    height=400,
                )
                st.plotly_chart(fig_lag, use_container_width=True)

# ── Tab 3: Lagged Heatmaps (month x lag) ────────────────────────────────────
with tab_heatmap:
    st.header("Lagged Anomaly Heatmap per Cluster")
    st.caption("Month x Lag heatmap for selected clusters")

    all_clusters = sorted(df_filtered['cluster_id'].unique())
    selected_clusters = st.multiselect(
        "Select clusters to display",
        all_clusters,
        default=all_clusters[:4] if len(all_clusters) >= 4 else all_clusters,
        key="heatmap_clusters",
    )

    if not selected_clusters:
        st.info("Select at least one cluster above.")
    else:
        n_cols = min(2, len(selected_clusters))
        hm_cols = st.columns(n_cols)

        for i, cur_cluster in enumerate(selected_clusters):
            with hm_cols[i % n_cols]:
                pivot = compute_heatmap_per_cluster(
                    df_filtered, oni_index, cur_cluster, mode_label
                )
                if mode_label == "ENSO Regimes (Nino vs Neutral)":
                    delta_label = "P(k|Nino) - P(k|Neutral)"
                else:
                    delta_label = "P(k|Nino,m) - P(k|m)"

                fig_hm, ax_hm = plt.subplots(figsize=(10, 5))
                sns.heatmap(
                    pivot, cmap='RdBu_r', center=0, annot=True, fmt=".1f",
                    linewidths=0.5, cbar_kws={'label': 'Delta P (%)'},
                    ax=ax_hm,
                )
                ax_hm.set_title(f"Cluster {cur_cluster}\n{delta_label}")
                ax_hm.set_ylabel("Month")
                ax_hm.set_xlabel("ENSO Lag (months)")
                plt.tight_layout()
                st.pyplot(fig_hm)
                plt.close(fig_hm)

# ── Tab 4: Season Distributions ─────────────────────────────────────────────
with tab_season:
    st.header("Season Distributions")

    # ── 4.1: All-clusters x months heatmap at a selected lag ─────────────
    st.subheader("All Clusters x Month Heatmap")

    selected_lag = st.slider(
        "ENSO Lag (months)", -12, 12, 0, key="season_lag"
    )

    if mode_label == "ENSO Regimes (Nino vs Neutral)":
        delta_label = "P(k|Nino) - P(k|Neutral)"
    else:
        delta_label = "P(k|Nino,m) - P(k|m)"

    heatmap_all = compute_all_clusters_heatmap(
        df_filtered, oni_index, mode_label, selected_lag
    )

    fig_hm_all, ax_hm_all = plt.subplots(figsize=(14, 10))
    sns.heatmap(
        heatmap_all, cmap='RdBu_r', center=0, annot=True, fmt=".1f",
        linewidths=0.5, cbar_kws={'label': 'Delta P (%)'},
        ax=ax_hm_all,
    )
    ax_hm_all.set_title(f"{delta_label} | Lag = {selected_lag} months")
    ax_hm_all.set_ylabel("Cluster ID")
    ax_hm_all.set_xlabel("Month")
    plt.tight_layout()
    st.pyplot(fig_hm_all)
    plt.close(fig_hm_all)

    # ── 4.2: Top 80% cluster samples — Nino vs Neutral/Climatology ──────
    st.subheader("Top 80% Cluster Samples")

    sel_season_key = st.selectbox(
        "Season", list(SEASON_OPTIONS.keys()), key="season_dist"
    )
    season_months = SEASON_OPTIONS[sel_season_key]

    df_season_filt = df_enso[df_enso['month'].isin(season_months)]

    if mode_label == "ENSO Regimes (Nino vs Neutral)":
        df_scenario_a = df_season_filt[df_season_filt['Label'] == 'El Niño']
        df_scenario_b = df_season_filt[df_season_filt['Label'] == 'Neutro']
        label_a, label_b = "El Nino", "Neutral"
    else:
        df_scenario_a = df_season_filt[df_season_filt['Label'] == 'El Niño']
        df_scenario_b = df_season_filt
        label_a, label_b = "El Nino", "Climatology"

    if df_scenario_a.empty:
        st.warning(f"No El Nino samples in {sel_season_key} for the selected period.")
    else:
        top_a, dist_a = get_top_80_clusters(df_scenario_a)
        top_b, dist_b = get_top_80_clusters(df_scenario_b)

        # Distribution bar chart
        all_top = sorted(set(top_a + top_b))
        comparison = pd.DataFrame({
            label_a: dist_a.reindex(all_top).fillna(0),
            label_b: dist_b.reindex(all_top).fillna(0),
        }).sort_index()

        fig_dist = go.Figure()
        fig_dist.add_trace(go.Bar(
            x=[str(c) for c in comparison.index],
            y=comparison[label_b], name=label_b,
            marker_color='#B0BEC5',
        ))
        fig_dist.add_trace(go.Bar(
            x=[str(c) for c in comparison.index],
            y=comparison[label_a], name=label_a,
            marker_color='#E53935',
        ))
        fig_dist.update_layout(
            barmode='group',
            title=f"Top 80% Clusters in {sel_season_key}: {label_a} vs {label_b}",
            xaxis_title="Cluster ID", yaxis_title="Probability", height=400,
        )
        st.plotly_chart(fig_dist, use_container_width=True)

        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.markdown(f"**{label_a}** top 80%: {top_a}")
        with col_info2:
            st.markdown(f"**{label_b}** top 80%: {top_b}")

        # Sample images grid
        st.markdown("---")
        st.markdown("**Representative Samples (highest cluster probability)**")

        # Compute var stats for color scale
        var_stats = {}
        for var in range(nvars):
            vmin = (combined_dataset[:, var, :, :] * norm_stds[var] + norm_means[var]).min().item()
            vmax = (combined_dataset[:, var, :, :] * norm_stds[var] + norm_means[var]).max().item()
            var_stats[var] = (vmin, vmax)

        max_cols = max(len(top_a), len(top_b))
        total_rows = 2 * nvars

        fig_samples, axes = plt.subplots(
            total_rows, max_cols,
            figsize=(3 * max_cols, 2.5 * total_rows),
            squeeze=False,
        )

        # -- Row block 1: Scenario B (Neutral / Climatology) --
        for idx in range(max_cols):
            if idx < len(top_b):
                cid = top_b[idx]
                prob = dist_b[cid] * 100
                sample = df_season_filt[df_season_filt['cluster_id'] == cid].nlargest(1, 'cluster_prob')

                if not sample.empty:
                    im_idx = sample.index[0]
                    image_data = combined_dataset[im_idx]

                    for var in range(nvars):
                        ax = axes[var, idx]
                        vmin, vmax = var_stats[var]
                        img = image_data[var].cpu().numpy() * norm_stds[var] + norm_means[var]
                        ax.imshow(img, cmap='coolwarm', vmin=vmin, vmax=vmax)
                        ax.axis('off')
                        if var == 0:
                            ax.set_title(f"C{cid} ({prob:.1f}%)", fontweight='bold')
                        if idx == 0:
                            ax.text(-0.1, 0.5, f"{label_b}\n{vars_names[var]}",
                                    transform=ax.transAxes, ha='right', va='center',
                                    fontweight='bold', fontsize=10)
            else:
                for var in range(nvars):
                    axes[var, idx].axis('off')

        # -- Row block 2: Scenario A (El Nino) --
        for idx in range(max_cols):
            if idx < len(top_a):
                cid = top_a[idx]
                prob = dist_a[cid] * 100
                sample = df_season_filt[df_season_filt['cluster_id'] == cid].nlargest(1, 'cluster_prob')

                if not sample.empty:
                    im_idx = sample.index[0]
                    image_data = combined_dataset[im_idx]

                    for var in range(nvars):
                        row_idx = nvars + var
                        ax = axes[row_idx, idx]
                        vmin, vmax = var_stats[var]
                        img = image_data[var].cpu().numpy() * norm_stds[var] + norm_means[var]
                        ax.imshow(img, cmap='coolwarm', vmin=vmin, vmax=vmax)
                        ax.axis('off')
                        if var == 0:
                            ax.set_title(f"C{cid} ({prob:.1f}%)", fontweight='bold')
                        if idx == 0:
                            ax.text(-0.1, 0.5, f"{label_a}\n{vars_names[var]}",
                                    transform=ax.transAxes, ha='right', va='center',
                                    fontweight='bold', fontsize=10)
            else:
                for var in range(nvars):
                    axes[nvars + var, idx].axis('off')

        plt.suptitle(
            f"{sel_season_key} Climate Regimes: Top 80% Clusters — {label_b} vs {label_a}",
            fontsize=14, y=1.02,
        )
        plt.tight_layout()
        st.pyplot(fig_samples)
        plt.close(fig_samples)
