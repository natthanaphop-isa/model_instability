"""
Combine cross-sample Optimism vs MAPE / CII BOX PLOTS for
Logistic Regression and ANN into single figures.

Loads full distributions from saved files:
  - Optimism:  from optimism_values.npy (if exists) else recomputes
  - MAPE:      from bootstrap_probs.npy + full_predictions.csv
  - CII:       from full_predictions.csv

Produces two figures saved to results/combined/:
  1. combined_optimism_vs_mape.png
  2. combined_optimism_vs_cii.png
"""

import json
import os
os.environ['MPLCONFIGDIR'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.matplotlib_cache')
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm
fm.findSystemFonts = lambda *args, **kwargs: []
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr

matplotlib.rcdefaults()
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
warnings.filterwarnings('ignore')

# ── Configuration ────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, 'results')
OUTPUT_DIR = os.path.join(RESULTS, 'combined_with_epv')
os.makedirs(OUTPUT_DIR, exist_ok=True)

SAMPLE_SIZES = [511, 2042, 5103, 10208, 20416, 40830]
FRACTIONS    = [0.0125, 0.05, 0.125, 0.25, 0.50, 1]
RANDOM_STATE = 931
N_BOOTSTRAP  = 200

# Mapping of sample size to its Events Per Variable (EPV) value
EPV_MAPPING = {
    511: 4.50,
    2042: 17.88,
    5103: 44.50,
    10208: 89.12,
    20416: 178.25,
    40830: 356.38
}

# Dataset
DF_PATH  = os.path.join(BASE, 'dataset', 'gusto_dataset(Sheet1).csv')
FEATURES = ['age', 'sex', 'hyp', 'htn', 'hrt', 'ste', 'pmi', 'sysbp']
TARGET   = 'day30'


# ── Data loading helpers ─────────────────────────────────────────
def load_gusto():
    df = pd.read_csv(DF_PATH)
    df['sex'] = df['sex'].apply(lambda x: 1 if x == 'male' else 0)
    df['pmi'] = df['pmi'].apply(lambda x: 1 if x == 'yes' else 0)
    return df


def subsample(df, frac):
    if frac == 1:
        return df
    return pd.concat([
        group.sample(frac=frac, replace=False, random_state=RANDOM_STATE)
        for _, group in df.groupby(TARGET)
    ]).reset_index(drop=True)


# ── Load distributions from saved files ──────────────────────────
def load_optimism(model_dir, n):
    """Load optimism_values.npy if it exists, else return None."""
    path = os.path.join(model_dir, f'df{n}', 'optimism_values.npy')
    if os.path.exists(path):
        return np.load(path)
    return None


def reconstruct_mape(model_dir, n):
    """Reconstruct per-bootstrap MAPE from bootstrap_probs.npy."""
    bp = np.load(os.path.join(model_dir, f'df{n}', 'bootstrap_probs.npy'))
    preds = pd.read_csv(os.path.join(model_dir, f'df{n}', 'full_predictions.csv'))
    origin = preds['origin_predict'].values
    return np.mean(np.abs(bp - origin[np.newaxis, :]), axis=1)


def reconstruct_cii(model_dir, n, threshold=0.1):
    """Reconstruct per-bootstrap CII from full_predictions.csv."""
    preds = pd.read_csv(os.path.join(model_dir, f'df{n}', 'full_predictions.csv'))
    origin = preds['origin_predict'].values
    cii_vals = []
    for i in range(N_BOOTSTRAP):
        col = f'{i}_bootstrap_probs'
        if col not in preds.columns:
            break
        bp = preds[col].values
        changes = ((bp >= threshold) & (origin < threshold)) | \
                  ((bp < threshold) & (origin >= threshold))
        cii_vals.append(changes.mean())
    return np.array(cii_vals)


# ── Recompute optimism only if missing ───────────────────────────
def compute_optimism(model_name, df, n, frac):
    """Recompute optimism via GridSearchCV + bootstrap C-statistic."""
    print(f"    [RECOMPUTE] optimism for {model_name} n={n} ...")
    df_sam = subsample(df, frac)
    X = df_sam[FEATURES]
    y = df_sam[TARGET]

    if model_name == 'Logistic Regression':
        base = LogisticRegression(random_state=RANDOM_STATE)
        pg = {'penalty': ['l2'],
              'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
              'solver': ["newton-cholesky", "sag", "saga", "lbfgs"]}
    else:
        base = MLPClassifier(random_state=RANDOM_STATE, max_iter=500)
        pg = {'hidden_layer_sizes': [(8,), (16,), (32,), (64,), (32,16), (64,32), (64,32,16)],
              'activation': ['logistic', 'tanh', 'relu'],
              'solver': ['adam'],
              'learning_rate': ['constant', 'adaptive'],
              'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
              'early_stopping': [True]}

    grid = GridSearchCV(estimator=base, param_grid=pg, cv=10, n_jobs=1, verbose=0)
    grid.fit(X, y)
    model = grid.best_estimator_
    model.fit(X, y)
    apparent_auc = roc_auc_score(y, model.predict_proba(X)[:, 1])

    data = pd.concat([X, y], axis=1)
    c_bs = []
    for _ in range(N_BOOTSTRAP):
        samp = data.sample(n=len(data), replace=True).reset_index(drop=True)
        xs, ys = samp.iloc[:, :-1], samp.iloc[:, -1]
        model.fit(xs, ys)
        c_bs.append(roc_auc_score(ys, model.predict_proba(xs)[:, 1]))

    return np.abs(np.array(c_bs) - apparent_auc)


# ── Build full summary ───────────────────────────────────────────
def build_summary(model_name, model_dir, df=None):
    """Load all distributions from saved files."""
    summary = []
    for n, frac in zip(SAMPLE_SIZES, FRACTIONS):
        result_dir = os.path.join(model_dir, f'df{n}')
        if not os.path.isdir(result_dir):
            print(f"  ⚠ Missing: {result_dir}")
            continue

        print(f"  {model_name} n={n}:")

        # Optimism: load from .npy or recompute
        optimism = load_optimism(model_dir, n)
        if optimism is not None:
            print(f"    [OK] optimism loaded ({len(optimism)} values)")
        else:
            if df is None:
                df = load_gusto()
            optimism = compute_optimism(model_name, df, n, frac)
            # Cache for next time
            np.save(os.path.join(result_dir, 'optimism_values.npy'), optimism)
            print(f"    [CACHED] optimism saved ({len(optimism)} values)")

        # MAPE: reconstruct from saved files
        mape = reconstruct_mape(model_dir, n)
        print(f"    [OK] mape reconstructed ({len(mape)} values)")

        # CII: reconstruct from saved files
        cii = reconstruct_cii(model_dir, n)
        print(f"    [OK] cii reconstructed ({len(cii)} values)")

        summary.append({
            'n': n,
            'optimism_values': optimism,
            'mape_values': mape,
            'cii_values': cii,
        })
    return sorted(summary, key=lambda d: d['n'])


# ── Plotting (box plots, matching original style) ────────────────
def plot_combined_boxplots(
    lr_summary, ann_summary, key1, key2, label1, label2,
    color_lr, color_ann, title, filename, edge_style2='-', color_ann_right=None
):
    """
    Side-by-side box plots: for each sample size, 4 boxes:
      LR-metric1 | ANN-metric1 | LR-metric2 | ANN-metric2
    Left y-axis = metric1, Right y-axis = metric2.
    Visual encoding: color = model, border style = metric.
      - metric1 (Optimism): solid fill, black solid border
      - metric2 (MAPE/CII): light fill, colored border (style via edge_style2)
    """
    labels = [f"n = {d['n']}\n(EPV = {EPV_MAPPING[d['n']]:.2f})" for d in lr_summary]
    n_groups = len(labels)

    fig, ax1 = plt.subplots(figsize=(max(14, n_groups * 3.2), 7))
    fig.patch.set_facecolor('white')
    ax1.set_facecolor('white')

    group_centers = np.arange(n_groups) * 4.5
    w = 0.55

    pos_lr1  = group_centers - 1.5 * w - 0.1
    pos_ann1 = group_centers - 0.5 * w - 0.03
    pos_lr2  = group_centers + 0.5 * w + 0.03
    pos_ann2 = group_centers + 1.5 * w + 0.1

    color_r_ann = color_ann_right if color_ann_right is not None else color_ann

    flier_lr = dict(marker='o', markersize=2, markeredgecolor=color_lr, alpha=0.3)
    flier_ann = dict(marker='o', markersize=2, markeredgecolor=color_ann, alpha=0.3)
    flier_ann_r = dict(marker='o', markersize=2, markeredgecolor=color_r_ann, alpha=0.3)

    # ── Left y-axis: metric1 (Optimism - Solid Fill, Black Border) ──
    bp1_lr = ax1.boxplot(
        [d[key1] for d in lr_summary], positions=pos_lr1, widths=w,
        patch_artist=True, showfliers=True, flierprops=flier_lr)
    for p in bp1_lr['boxes']:
        p.set_facecolor(mcolors.to_rgba(color_lr, alpha=0.75))
        p.set_edgecolor('black')
        p.set_linewidth(1.0)
    for m in bp1_lr['medians']:
        m.set_color('black')
        m.set_linewidth(1.5)
    for wl in bp1_lr['whiskers']:
        wl.set_color('black')
        wl.set_linewidth(1.0)
    for c in bp1_lr['caps']:
        c.set_color('black')
        c.set_linewidth(1.0)

    bp1_ann = ax1.boxplot(
        [d[key1] for d in ann_summary], positions=pos_ann1, widths=w,
        patch_artist=True, showfliers=True, flierprops=flier_ann)
    for p in bp1_ann['boxes']:
        p.set_facecolor(mcolors.to_rgba(color_ann, alpha=0.75))
        p.set_edgecolor('black')
        p.set_linewidth(1.0)
    for m in bp1_ann['medians']:
        m.set_color('black')
        m.set_linewidth(1.5)
    for wl in bp1_ann['whiskers']:
        wl.set_color('black')
        wl.set_linewidth(1.0)
    for c in bp1_ann['caps']:
        c.set_color('black')
        c.set_linewidth(1.0)

    ax1.set_ylabel(label1, fontsize=12)
    ax1.tick_params(axis='y')

    # ── Right y-axis: metric2 (MAPE/CII - Light Fill, Colored Border) ──
    ax2 = ax1.twinx()

    bp2_lr = ax2.boxplot(
        [d[key2] for d in lr_summary], positions=pos_lr2, widths=w,
        patch_artist=True, showfliers=True, flierprops=flier_lr)
    for p in bp2_lr['boxes']:
        p.set_facecolor(mcolors.to_rgba(color_lr, alpha=0.12))
        p.set_edgecolor(mcolors.to_rgba(color_lr, alpha=1.0))
        p.set_linewidth(1.8)
        p.set_linestyle(edge_style2)
    for m in bp2_lr['medians']:
        m.set_color(mcolors.to_rgba(color_lr, alpha=1.0))
        m.set_linewidth(2.0)
    for wl in bp2_lr['whiskers']:
        wl.set_color(mcolors.to_rgba(color_lr, alpha=0.8))
        wl.set_linewidth(1.2)
        wl.set_linestyle(edge_style2)
    for c in bp2_lr['caps']:
        c.set_color(mcolors.to_rgba(color_lr, alpha=0.8))
        c.set_linewidth(1.2)

    bp2_ann = ax2.boxplot(
        [d[key2] for d in ann_summary], positions=pos_ann2, widths=w,
        patch_artist=True, showfliers=True, flierprops=flier_ann_r)
    for p in bp2_ann['boxes']:
        p.set_facecolor(mcolors.to_rgba(color_r_ann, alpha=0.12))
        p.set_edgecolor(mcolors.to_rgba(color_r_ann, alpha=1.0))
        p.set_linewidth(1.8)
        p.set_linestyle(edge_style2)
    for m in bp2_ann['medians']:
        m.set_color(mcolors.to_rgba(color_r_ann, alpha=1.0))
        m.set_linewidth(2.0)
    for wl in bp2_ann['whiskers']:
        wl.set_color(mcolors.to_rgba(color_r_ann, alpha=0.8))
        wl.set_linewidth(1.2)
        wl.set_linestyle(edge_style2)
    for c in bp2_ann['caps']:
        c.set_color(mcolors.to_rgba(color_r_ann, alpha=0.8))
        c.set_linewidth(1.2)

    ax2.set_ylabel(label2, fontsize=12)
    ax2.tick_params(axis='y')

    # ── X-axis ──
    ax1.set_xticks(group_centers)
    ax1.set_xticklabels(labels)
    ax1.set_xlabel('Sample Size and Events Per Variable (EPV)', fontsize=12)
    ax1.set_title(title, fontsize=14)
    ax1.set_xlim(group_centers[0] - 2.0, group_centers[-1] + 2.0)

    # ── Alternating background bands ──
    band_half = 2.0
    for i, center in enumerate(group_centers):
        if i % 2 == 0:
            ax1.axvspan(center - band_half, center + band_half,
                        color='#f0f0f0', zorder=0)

    # ── Legend ──
    legend_items = [
        Patch(facecolor=mcolors.to_rgba(color_lr, alpha=0.75),  edgecolor='black', linewidth=1.0, label=f'LR — {label1}'),
        Patch(facecolor=mcolors.to_rgba(color_ann, alpha=0.75), edgecolor='black', linewidth=1.0, label=f'ANN — {label1}'),
        Patch(facecolor=mcolors.to_rgba(color_lr, alpha=0.12),  edgecolor=color_lr, linewidth=1.8, linestyle=edge_style2, label=f'LR — {label2}'),
        Patch(facecolor=mcolors.to_rgba(color_r_ann, alpha=0.12), edgecolor=color_r_ann, linewidth=1.8, linestyle=edge_style2, label=f'ANN — {label2}'),
    ]
    ax1.legend(handles=legend_items, loc='upper right', frameon=True, fontsize=12).set_zorder(10)

    ax1.grid(False)
    ax2.grid(False)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"  ✓ Saved: {save_path}")


# ── All-three-metrics plot ───────────────────────────────────────
def plot_all_three_boxplots(
    lr_summary, ann_summary, color_lr, color_ann, filename, color_ann_cii=None
):
    """
    Side-by-side box plots with ALL three metrics per sample size:
      LR-Opt | ANN-Opt | LR-MAPE | ANN-MAPE | LR-CII | ANN-CII
    Left y-axis  = Optimism  (solid fill, black edge)
    Right y-axis = MAPE & CII (light/medium fill, colored edge)
    """
    labels = [f"n = {d['n']}\n(EPV = {EPV_MAPPING[d['n']]:.2f})" for d in lr_summary]
    n_groups = len(labels)

    fig, ax1 = plt.subplots(figsize=(max(14, n_groups * 3.2), 7))
    fig.patch.set_facecolor('white')
    ax1.set_facecolor('white')

    group_centers = np.arange(n_groups) * 4.8
    w = 0.62

    # 6 positions per group: Opt(LR,ANN)  MAPE(LR,ANN)  CII(LR,ANN)
    pos_opt_lr   = group_centers - 2.5 * w - 0.06
    pos_opt_ann  = group_centers - 1.5 * w - 0.03
    pos_mape_lr  = group_centers - 0.5 * w + 0.0
    pos_mape_ann = group_centers + 0.5 * w + 0.0
    pos_cii_lr   = group_centers + 1.5 * w + 0.03
    pos_cii_ann  = group_centers + 2.5 * w + 0.06

    color_cii_ann_val = color_ann_cii if color_ann_cii is not None else color_ann

    flier_lr  = dict(marker='o', markersize=2, markeredgecolor=color_lr, alpha=0.3)
    flier_ann = dict(marker='o', markersize=2, markeredgecolor=color_ann, alpha=0.3)
    flier_ann_cii = dict(marker='o', markersize=2, markeredgecolor=color_cii_ann_val, alpha=0.3)

    def style_boxes(bp, facecolor, edgecolor, edge_lw=1.0, linestyle='-'):
        for p in bp['boxes']:
            p.set_facecolor(facecolor)
            p.set_edgecolor(edgecolor)
            p.set_linewidth(edge_lw)
            p.set_linestyle(linestyle)
        for m in bp['medians']:
            m.set_color('black')
            m.set_linewidth(1.5)
        for wl in bp['whiskers']:
            wl.set_color(edgecolor)
            wl.set_linewidth(1.0)
        for c in bp['caps']:
            c.set_color(edgecolor)
            c.set_linewidth(1.0)

    # ── Left y-axis: Optimism (solid fill, black edge) ──
    bp_opt_lr = ax1.boxplot(
        [d['optimism_values'] for d in lr_summary], positions=pos_opt_lr,
        widths=w, patch_artist=True, showfliers=True, flierprops=flier_lr)
    style_boxes(bp_opt_lr, mcolors.to_rgba(color_lr, 0.75), 'black')

    bp_opt_ann = ax1.boxplot(
        [d['optimism_values'] for d in ann_summary], positions=pos_opt_ann,
        widths=w, patch_artist=True, showfliers=True, flierprops=flier_ann)
    style_boxes(bp_opt_ann, mcolors.to_rgba(color_ann, 0.75), 'black')

    ax1.set_ylabel('Optimism', fontsize=12)

    # ── Right y-axis: MAPE & CII ──
    ax2 = ax1.twinx()

    # MAPE — light fill, solid colored edge
    bp_mape_lr = ax2.boxplot(
        [d['mape_values'] for d in lr_summary], positions=pos_mape_lr,
        widths=w, patch_artist=True, showfliers=True, flierprops=flier_lr)
    style_boxes(bp_mape_lr, mcolors.to_rgba(color_lr, 0.12), color_lr, 1.8)

    bp_mape_ann = ax2.boxplot(
        [d['mape_values'] for d in ann_summary], positions=pos_mape_ann,
        widths=w, patch_artist=True, showfliers=True, flierprops=flier_ann)
    style_boxes(bp_mape_ann, mcolors.to_rgba(color_ann, 0.12), color_ann, 1.8)

    # CII — medium fill, dashed colored edge
    bp_cii_lr = ax2.boxplot(
        [d['cii_values'] for d in lr_summary], positions=pos_cii_lr,
        widths=w, patch_artist=True, showfliers=True, flierprops=flier_lr)
    style_boxes(bp_cii_lr, mcolors.to_rgba(color_lr, 0.30), color_lr, 1.8, '--')

    bp_cii_ann = ax2.boxplot(
        [d['cii_values'] for d in ann_summary], positions=pos_cii_ann,
        widths=w, patch_artist=True, showfliers=True, flierprops=flier_ann_cii)
    style_boxes(bp_cii_ann, mcolors.to_rgba(color_cii_ann_val, 0.30), color_cii_ann_val, 1.8, '--')

    ax2.set_ylabel('MAPE / CII', fontsize=12)

    # ── X-axis ──
    ax1.set_xticks(group_centers)
    ax1.set_xticklabels(labels)
    ax1.set_xlabel('Sample Size and Events Per Variable (EPV)', fontsize=12)
    ax1.set_title('Optimism, MAPE and CII across Sample Sizes', fontsize=14)
    ax1.set_xlim(group_centers[0] - 2.3, group_centers[-1] + 2.3)

    # ── Alternating background bands ──
    band_half = 2.2
    for i, center in enumerate(group_centers):
        if i % 2 == 0:
            ax1.axvspan(center - band_half, center + band_half,
                        color='#f0f0f0', zorder=0)

    # ── Legend ──
    legend_items = [
        Patch(facecolor=mcolors.to_rgba(color_lr, 0.75),  edgecolor='black',   linewidth=1.0, label='LR — Optimism'),
        Patch(facecolor=mcolors.to_rgba(color_ann, 0.75), edgecolor='black',   linewidth=1.0, label='ANN — Optimism'),
        Patch(facecolor=mcolors.to_rgba(color_lr, 0.12),  edgecolor=color_lr,  linewidth=1.8, label='LR — MAPE'),
        Patch(facecolor=mcolors.to_rgba(color_ann, 0.12), edgecolor=color_ann, linewidth=1.8, label='ANN — MAPE'),
        Patch(facecolor=mcolors.to_rgba(color_lr, 0.30),  edgecolor=color_lr,  linewidth=1.8, linestyle='--', label='LR — CII'),
        Patch(facecolor=mcolors.to_rgba(color_cii_ann_val, 0.30), edgecolor=color_cii_ann_val, linewidth=1.8, linestyle='--', label='ANN — CII'),
    ]
    ax1.legend(handles=legend_items, loc='upper right', frameon=True,
               fontsize=11, ncol=2).set_zorder(10)

    ax1.grid(False)
    ax2.grid(False)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"  ✓ Saved: {save_path}")


# ── Median + Scatter (2-metric) ──────────────────────────────────
def plot_combined_median_scatter(
    lr_summary, ann_summary, key1, key2, label1, label2,
    color_lr, color_ann, title, filename, marker2='o', color_ann_right=None
):
    """
    Median markers + jittered scatter for 2 metrics across sample sizes.
    Same layout as plot_combined_boxplots but with scatter instead of boxes.
    """
    labels = [f"n = {d['n']}\n(EPV = {EPV_MAPPING[d['n']]:.2f})" for d in lr_summary]
    n_groups = len(labels)

    fig, ax1 = plt.subplots(figsize=(max(14, n_groups * 3.2), 7))
    fig.patch.set_facecolor('white')
    ax1.set_facecolor('white')

    group_centers = np.arange(n_groups) * 4.5
    w = 0.55
    jitter_w = w * 0.35

    pos_lr1  = group_centers - 1.5 * w - 0.1
    pos_ann1 = group_centers - 0.5 * w - 0.03
    pos_lr2  = group_centers + 0.5 * w + 0.03
    pos_ann2 = group_centers + 1.5 * w + 0.1

    color_r_ann = color_ann_right if color_ann_right is not None else color_ann

    rng = np.random.default_rng(42)

    def draw_scatter(ax, data_list, positions, color, alpha_dot=0.25, size=24):
        for pos, vals in zip(positions, data_list):
            jitter = rng.uniform(-jitter_w, jitter_w, size=len(vals))
            ax.scatter(pos + jitter, vals, s=size, alpha=alpha_dot,
                       color=color, edgecolors='none', zorder=2)

    def draw_median(ax, data_list, positions, color, marker='_', ms=18, mew=3.5):
        medians = [np.median(v) for v in data_list]
        if marker == '_':
            ax.scatter(positions, medians, marker=marker, s=ms**2,
                       color='black', linewidths=mew, zorder=4)
        else:
            ax.scatter(positions, medians, marker=marker, s=ms**2,
                       color=color, edgecolors='black', linewidths=1.0,
                       zorder=4)

    # ── Left y-axis: metric1 (Optimism) ──
    data_lr1  = [d[key1] for d in lr_summary]
    data_ann1 = [d[key1] for d in ann_summary]

    draw_scatter(ax1, data_lr1,  pos_lr1,  color_lr)
    draw_scatter(ax1, data_ann1, pos_ann1, color_ann)
    draw_median(ax1, data_lr1,  pos_lr1,  color_lr,  marker='_')
    draw_median(ax1, data_ann1, pos_ann1, color_ann, marker='_')

    ax1.set_ylabel(label1, fontsize=12)
    ax1.tick_params(axis='y')

    # ── Right y-axis: metric2 (MAPE / CII) ──
    ax2 = ax1.twinx()
    data_lr2  = [d[key2] for d in lr_summary]
    data_ann2 = [d[key2] for d in ann_summary]

    draw_scatter(ax2, data_lr2,  pos_lr2,  color_lr,  alpha_dot=0.20, size=18)
    draw_scatter(ax2, data_ann2, pos_ann2, color_r_ann, alpha_dot=0.20, size=18)
    draw_median(ax2, data_lr2,  pos_lr2,  color_lr,  marker=marker2, ms=10)
    draw_median(ax2, data_ann2, pos_ann2, color_r_ann, marker=marker2, ms=10)

    ax2.set_ylabel(label2, fontsize=12)
    ax2.tick_params(axis='y')

    # ── X-axis ──
    ax1.set_xticks(group_centers)
    ax1.set_xticklabels(labels)
    ax1.set_xlabel('Sample Size and Events Per Variable (EPV)', fontsize=12)
    ax1.set_title(title, fontsize=14)
    ax1.set_xlim(group_centers[0] - 2.0, group_centers[-1] + 2.0)

    # ── Alternating background bands ──
    band_half = 2.0
    for i, center in enumerate(group_centers):
        if i % 2 == 0:
            ax1.axvspan(center - band_half, center + band_half,
                        color='#f0f0f0', zorder=0)

    # ── Legend ──
    from matplotlib.lines import Line2D
    legend_items = [
        Line2D([0], [0], marker='o', color=color_lr, alpha=0.4, markersize=8, linestyle='None', label='LR (dots)'),
        Line2D([0], [0], marker='o', color=color_r_ann, alpha=0.4, markersize=8, linestyle='None', label='ANN (dots)'),
        Line2D([0], [0], marker='_', color='black',  markersize=12, linestyle='None', markeredgewidth=3.5, label=f'LR — {label1} (median)'),
        Line2D([0], [0], marker='_', color='black',  markersize=12, linestyle='None', markeredgewidth=3.5, label=f'ANN — {label1} (median)'),
        Line2D([0], [0], marker=marker2, color=color_lr,  markersize=8, linestyle='None', label=f'LR — {label2} (median)'),
        Line2D([0], [0], marker=marker2, color=color_r_ann, markersize=8, linestyle='None', label=f'ANN — {label2} (median)'),
    ]
    ax1.legend(handles=legend_items, loc='upper right', frameon=True, fontsize=12).set_zorder(10)

    ax1.grid(False)
    ax2.grid(False)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"  ✓ Saved: {save_path}")


# ── Mean + Scatter (stacked: MAPE + CII in one PNG for a single model) ──
def plot_single_model_mean_scatter_stacked(
    summary, color_opt, color_mape, color_cii, model_name, filename
):
    """
    Two-row figure combining Optimism vs MAPE (top) and Optimism vs CII (bottom)
    for a single model in a single PNG, sharing the same x-axis layout.
    Uses metric-specific colors and coordinates label/tick colors with the axes.
    Plots the mean of each metric as a color-coded horizontal line with a white outline.
    """
    from matplotlib.lines import Line2D

    labels = [f"n = {d['n']}\n(EPV = {EPV_MAPPING[d['n']]:.2f})" for d in summary]
    n_groups = len(labels)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(max(14, n_groups * 3.2), 13), sharex=False
    )
    fig.patch.set_facecolor('white')

    group_centers = np.arange(n_groups) * 4.5
    w = 0.55
    jitter_w = w * 0.35
    rng = np.random.default_rng(42)

    pos_left  = group_centers - 0.45
    pos_right = group_centers + 0.45

    def draw_scatter(ax, data_list, positions, color, alpha_dot=0.25, size=24):
        for pos, vals in zip(positions, data_list):
            jitter = rng.uniform(-jitter_w, jitter_w, size=len(vals))
            ax.scatter(pos + jitter, vals, s=size, alpha=alpha_dot,
                       color=color, edgecolors='none', zorder=2)

    def draw_mean(ax, data_list, positions, color, marker='_', ms=18, mew=3.5):
        means = [np.mean(v) for v in data_list]
        # Draw a white outline behind the mean line to make it pop out of the scatter cloud
        ax.scatter(positions, means, marker='_', s=ms**2,
                   color='white', linewidths=mew + 2.0, zorder=3)
        # Draw the actual mean line using the metric's own color
        ax.scatter(positions, means, marker='_', s=ms**2,
                   color=color, linewidths=mew, zorder=4)

    def draw_panel(ax_left, key2, label2, marker2, panel_title, color_r):
        ax_left.set_facecolor('white')

        # Left y-axis: Optimism
        data_opt = [d['optimism_values'] for d in summary]
        draw_scatter(ax_left, data_opt, pos_left, color_opt, alpha_dot=0.25, size=24)
        draw_mean(ax_left, data_opt, pos_left, color_opt, marker='_', ms=18, mew=3.5)
        ax_left.set_ylabel('Optimism', fontsize=12, color=color_opt)
        ax_left.tick_params(axis='y', labelcolor=color_opt)

        # Right y-axis: MAPE or CII
        ax_right = ax_left.twinx()
        data_right = [d[key2] for d in summary]
        draw_scatter(ax_right, data_right, pos_right, color_r, alpha_dot=0.20, size=18)
        draw_mean(ax_right, data_right, pos_right, color_r, marker='_', ms=18, mew=3.5)
        ax_right.set_ylabel(label2, fontsize=12, color=color_r)
        ax_right.tick_params(axis='y', labelcolor=color_r)

        ax_left.set_xticks(group_centers)
        ax_left.set_xticklabels(labels)
        ax_left.set_xlabel('Sample Size and Events Per Variable (EPV)', fontsize=12)
        ax_left.set_title(panel_title, fontsize=13)
        ax_left.set_xlim(group_centers[0] - 2.0, group_centers[-1] + 2.0)

        # Alternating background bands
        for i, center in enumerate(group_centers):
            if i % 2 == 0:
                ax_left.axvspan(center - 2.0, center + 2.0,
                                color='#f0f0f0', zorder=0)

        legend_items = [
            Line2D([0], [0], marker='o', color=color_opt, alpha=0.4, markersize=8,
                   linestyle='None', label='Optimism (dots)'),
            Line2D([0], [0], marker='o', color=color_r, alpha=0.4, markersize=8,
                   linestyle='None', label=f'{label2} (dots)'),
            Line2D([0], [0], marker='_', color=color_opt, markersize=12,
                   linestyle='None', markeredgewidth=3.5, label='Optimism (mean)'),
            Line2D([0], [0], marker='_', color=color_r, markersize=12,
                   linestyle='None', markeredgewidth=3.5, label=f'{label2} (mean)'),
        ]
        ax_left.legend(handles=legend_items, loc='upper right', frameon=True, fontsize=11).set_zorder(10)
        ax_left.grid(False)
        ax_right.grid(False)

    # Top panel: Optimism vs MAPE
    draw_panel(ax_top, 'mape_values', 'MAPE', 'o', f'{model_name} — Optimism and MAPE across Sample Sizes', color_mape)
    # Bottom panel: Optimism vs CII
    draw_panel(ax_bot, 'cii_values',  'CII',  'D', f'{model_name} — Optimism and CII across Sample Sizes', color_cii)

    plt.tight_layout()
    fig.subplots_adjust(hspace=0.20)  # Reduced vertical space for a more compact, elegant gap
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"  ✓ Saved: {save_path}")



# ── Median + Scatter (all-three-metrics) ─────────────────────────
def plot_all_three_median_scatter(
    lr_summary, ann_summary, color_lr, color_ann, filename, color_ann_cii=None
):
    """
    Median markers + jittered scatter for all 3 metrics across sample sizes.
    Same layout as plot_all_three_boxplots but with scatter instead of boxes.
    """
    labels = [f"n = {d['n']}\n(EPV = {EPV_MAPPING[d['n']]:.2f})" for d in lr_summary]
    n_groups = len(labels)

    fig, ax1 = plt.subplots(figsize=(max(14, n_groups * 3.2), 7))
    fig.patch.set_facecolor('white')
    ax1.set_facecolor('white')

    group_centers = np.arange(n_groups) * 4.8
    w = 0.62
    jitter_w = w * 0.35

    pos_opt_lr   = group_centers - 2.5 * w - 0.06
    pos_opt_ann  = group_centers - 1.5 * w - 0.03
    pos_mape_lr  = group_centers - 0.5 * w + 0.0
    pos_mape_ann = group_centers + 0.5 * w + 0.0
    pos_cii_lr   = group_centers + 1.5 * w + 0.03
    pos_cii_ann  = group_centers + 2.5 * w + 0.06

    color_cii_ann_val = color_ann_cii if color_ann_cii is not None else color_ann

    rng = np.random.default_rng(42)

    def draw_scatter(ax, data_list, positions, color, alpha_dot=0.25, size=24):
        for pos, vals in zip(positions, data_list):
            jitter = rng.uniform(-jitter_w, jitter_w, size=len(vals))
            ax.scatter(pos + jitter, vals, s=size, alpha=alpha_dot,
                       color=color, edgecolors='none', zorder=2)

    def draw_median(ax, data_list, positions, color, marker='_', ms=14, mew=3.5):
        medians = [np.median(v) for v in data_list]
        if marker == '_':
            ax.scatter(positions, medians, marker=marker, s=ms**2,
                       color='black', linewidths=mew, zorder=4)
        else:
            ax.scatter(positions, medians, marker=marker, s=ms**2,
                       color=color, edgecolors='black', linewidths=1.0,
                       zorder=4)

    # ── Left y-axis: Optimism ──
    opt_lr  = [d['optimism_values'] for d in lr_summary]
    opt_ann = [d['optimism_values'] for d in ann_summary]

    draw_scatter(ax1, opt_lr,  pos_opt_lr,  color_lr)
    draw_scatter(ax1, opt_ann, pos_opt_ann, color_ann)
    draw_median(ax1, opt_lr,  pos_opt_lr,  color_lr,  marker='_')
    draw_median(ax1, opt_ann, pos_opt_ann, color_ann, marker='_')

    ax1.set_ylabel('Optimism', fontsize=12)

    # ── Right y-axis: MAPE & CII ──
    ax2 = ax1.twinx()

    mape_lr  = [d['mape_values'] for d in lr_summary]
    mape_ann = [d['mape_values'] for d in ann_summary]
    cii_lr   = [d['cii_values'] for d in lr_summary]
    cii_ann  = [d['cii_values'] for d in ann_summary]

    # MAPE — circle marker
    draw_scatter(ax2, mape_lr,  pos_mape_lr,  color_lr,  alpha_dot=0.20, size=18)
    draw_scatter(ax2, mape_ann, pos_mape_ann, color_ann, alpha_dot=0.20, size=18)
    draw_median(ax2, mape_lr,  pos_mape_lr,  color_lr,  marker='o', ms=10)
    draw_median(ax2, mape_ann, pos_mape_ann, color_ann, marker='o', ms=10)

    # CII — diamond marker
    draw_scatter(ax2, cii_lr,  pos_cii_lr,  color_lr,  alpha_dot=0.20, size=18)
    draw_scatter(ax2, cii_ann, pos_cii_ann, color_cii_ann_val, alpha_dot=0.20, size=18)
    draw_median(ax2, cii_lr,  pos_cii_lr,  color_lr,  marker='D', ms=8)
    draw_median(ax2, cii_ann, pos_cii_ann, color_cii_ann_val, marker='D', ms=8)

    ax2.set_ylabel('MAPE / CII', fontsize=12)

    # ── X-axis ──
    ax1.set_xticks(group_centers)
    ax1.set_xticklabels(labels)
    ax1.set_xlabel('Sample Size and Events Per Variable (EPV)', fontsize=12)
    ax1.set_title('Optimism, MAPE and CII across Sample Sizes (Median + Scatter)', fontsize=14)
    ax1.set_xlim(group_centers[0] - 2.3, group_centers[-1] + 2.3)

    # ── Alternating background bands ──
    band_half = 2.2
    for i, center in enumerate(group_centers):
        if i % 2 == 0:
            ax1.axvspan(center - band_half, center + band_half,
                        color='#f0f0f0', zorder=0)

    # ── Legend ──
    from matplotlib.lines import Line2D
    legend_items = [
        Line2D([0], [0], marker='o', color=color_lr, alpha=0.4, markersize=8, linestyle='None', label='LR (dots)'),
        Line2D([0], [0], marker='o', color=color_cii_ann_val, alpha=0.4, markersize=8, linestyle='None', label='ANN (dots)'),
        Line2D([0], [0], marker='_', color='black',  markersize=12, linestyle='None', markeredgewidth=3.5, label='LR — Optimism (median)'),
        Line2D([0], [0], marker='_', color='black',  markersize=12, linestyle='None', markeredgewidth=3.5, label='ANN — Optimism (median)'),
        Line2D([0], [0], marker='o', color=color_lr,  markersize=8, linestyle='None', label='LR — MAPE (median)'),
        Line2D([0], [0], marker='o', color=color_ann, markersize=8, linestyle='None', label='ANN — MAPE (median)'),
        Line2D([0], [0], marker='D', color=color_lr,  markersize=7, linestyle='None', label='LR — CII (median)'),
        Line2D([0], [0], marker='D', color=color_cii_ann_val, markersize=7, linestyle='None', label='ANN — CII (median)'),
    ]
    ax1.legend(handles=legend_items, loc='upper right', frameon=True,
               fontsize=11, ncol=2).set_zorder(10)

    ax1.grid(False)
    ax2.grid(False)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"  ✓ Saved: {save_path}")


def plot_median_trends_with_bands(lr_summary, ann_summary, color_lr, color_ann, filename, color_ann_cii=None):
    """
    Plots a 3-panel horizontal figure (Optimism, MAPE, CII) showing:
      - Median trends as solid lines with markers.
      - Shaded IQR bands (25th to 75th percentile) for spread.
      - Lighter shaded 5th-95th percentile bands for outer bounds.
    """
    metrics = [
        ('optimism_values', 'Optimism', 'Optimism Trend'),
        ('mape_values', 'MAPE', 'MAPE Trend'),
        ('cii_values', 'CII', 'Classification Instability Index (CII)')
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharex=False)
    fig.patch.set_facecolor('white')
    
    x = [d['n'] for d in lr_summary] # sample sizes
    color_cii_ann_val = color_ann_cii if color_ann_cii is not None else color_ann
    
    for idx, (key, label, title) in enumerate(metrics):
        ax = axes[idx]
        ax.set_facecolor('white')
        
        c_ann = color_cii_ann_val if key == 'cii_values' else color_ann
        
        # Calculate stats for LR
        lr_medians = [np.median(d[key]) for d in lr_summary]
        lr_q25 = [np.percentile(d[key], 25) for d in lr_summary]
        lr_q75 = [np.percentile(d[key], 75) for d in lr_summary]
        lr_p5 = [np.percentile(d[key], 5) for d in lr_summary]
        lr_p95 = [np.percentile(d[key], 95) for d in lr_summary]
        
        # Calculate stats for ANN
        ann_medians = [np.median(d[key]) for d in ann_summary]
        ann_q25 = [np.percentile(d[key], 25) for d in ann_summary]
        ann_q75 = [np.percentile(d[key], 75) for d in ann_summary]
        ann_p5 = [np.percentile(d[key], 5) for d in ann_summary]
        ann_p95 = [np.percentile(d[key], 95) for d in ann_summary]
        
        # Plot ANN outer band (5th-95th) - very light
        ax.fill_between(x, ann_p5, ann_p95, color=c_ann, alpha=0.06, label='ANN 5th–95th %ile')
        # Plot LR outer band (5th-95th) - very light
        ax.fill_between(x, lr_p5, lr_p95, color=color_lr, alpha=0.06, label='LR 5th–95th %ile')
        
        # Plot ANN IQR band (25th-75th) - medium
        ax.fill_between(x, ann_q25, ann_q75, color=c_ann, alpha=0.18, label='ANN IQR (25th–75th)')
        # Plot LR IQR band (25th-75th) - medium
        ax.fill_between(x, lr_q25, lr_q75, color=color_lr, alpha=0.18, label='LR IQR (25th–75th)')
        
        # Plot solid lines for Medians
        ax.plot(x, ann_medians, 'o-', color=c_ann, linewidth=2.5, markersize=7, label='ANN Median')
        ax.plot(x, lr_medians, 's-', color=color_lr, linewidth=2.5, markersize=6, label='LR Median')
        
        # Formatting
        ax.set_xscale('log')  # exponential growth of sample size
        ax.set_xticks(x)
        ax.set_xticklabels([f"n = {val}\n(EPV = {EPV_MAPPING[val]:.2f})" for val in x], rotation=30, ha='right', fontsize=8.5)
        ax.set_xlabel('Sample Size and Events Per Variable (EPV)', fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold', pad=12)
        ax.grid(True, linestyle=':', alpha=0.6)
        
        if idx == 0:
            ax.legend(frameon=True, fontsize=9, loc='upper right')
            
    fig.suptitle('Stability Metric Trends across Sample Sizes (Median, IQR, and 5th–95th Percentiles)',
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"  ✓ Saved: {save_path}")


# ── Per-bootstrap correlation analysis ───────────────────────────

def bootstrap_correlations(x, y, n_resample=200, random_state=42):
    """
    Bootstrap-resample paired vectors (x, y) and compute Pearson r
    for each resample, producing a distribution of correlation coefficients.

    Args:
        x, y: arrays of length N (the 200 per-bootstrap values).
        n_resample: how many bootstrap resamples to draw.
        random_state: seed for reproducibility.

    Returns:
        np.array of shape (n_resample,) with Pearson r values.
    """
    rng = np.random.default_rng(random_state)
    n = len(x)
    corrs = []
    for _ in range(n_resample):
        idx = rng.integers(0, n, size=n)
        try:
            r, _ = pearsonr(x[idx], y[idx])
            corrs.append(r)
        except Exception:
            corrs.append(np.nan)
    return np.array(corrs)


def plot_corr_bootstrap_boxplots(
    lr_summary, ann_summary, metric_key, metric_label,
    color_lr, color_ann, filename, n_resample=200, marker='o'
):
    """
    Version 1 — Boxplots of bootstrapped Pearson r(Optimism, metric)
    for LR and ANN side-by-side at each sample size.
    """
    labels = [f"n = {d['n']}\n(EPV = {EPV_MAPPING[d['n']]:.2f})" for d in lr_summary]
    n_groups = len(labels)

    # Compute bootstrapped correlations at each sample size
    lr_corrs = []
    ann_corrs = []
    for d_lr, d_ann in zip(lr_summary, ann_summary):
        lr_corrs.append(bootstrap_correlations(
            np.asarray(d_lr['optimism_values'], dtype=float),
            np.asarray(d_lr[metric_key], dtype=float),
            n_resample=n_resample
        ))
        ann_corrs.append(bootstrap_correlations(
            np.asarray(d_ann['optimism_values'], dtype=float),
            np.asarray(d_ann[metric_key], dtype=float),
            n_resample=n_resample
        ))

    fig, ax = plt.subplots(figsize=(max(12, n_groups * 2.5), 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    group_centers = np.arange(n_groups) * 2.5
    offset = 0.45

    # LR boxplots
    bp_lr = ax.boxplot(
        lr_corrs, positions=group_centers - offset, widths=0.65,
        patch_artist=True, showfliers=True,
        flierprops=dict(marker=marker, markersize=2, alpha=0.3, markeredgecolor=color_lr)
    )
    for p in bp_lr['boxes']:
        p.set_facecolor(mcolors.to_rgba(color_lr, 0.55))
        p.set_edgecolor(color_lr)
        p.set_linewidth(1.2)
    for m in bp_lr['medians']:
        m.set_color('black')
        m.set_linewidth(2.0)
    for w in bp_lr['whiskers'] + bp_lr['caps']:
        w.set_color(color_lr)
        w.set_linewidth(1.0)

    # ANN boxplots
    bp_ann = ax.boxplot(
        ann_corrs, positions=group_centers + offset, widths=0.65,
        patch_artist=True, showfliers=True,
        flierprops=dict(marker=marker, markersize=2, alpha=0.3, markeredgecolor=color_ann)
    )
    for p in bp_ann['boxes']:
        p.set_facecolor(mcolors.to_rgba(color_ann, 0.55))
        p.set_edgecolor(color_ann)
        p.set_linewidth(1.2)
    for m in bp_ann['medians']:
        m.set_color('black')
        m.set_linewidth(2.0)
    for w in bp_ann['whiskers'] + bp_ann['caps']:
        w.set_color(color_ann)
        w.set_linewidth(1.0)

    ax.axhline(0, color='grey', linestyle=':', linewidth=0.8, alpha=0.6)

    ax.set_xticks(group_centers)
    ax.set_xticklabels(labels)
    ax.set_xlabel('Sample Size and Events Per Variable (EPV)', fontsize=12)
    ax.set_ylabel(f'Pearson r (Optimism compared to {metric_label})', fontsize=12)
    ax.set_title(
        f'Bootstrapped Correlation: Optimism compared to {metric_label} across Sample Sizes',
        fontsize=14
    )

    # Alternating background bands
    band_half = 1.15
    for i, center in enumerate(group_centers):
        if i % 2 == 0:
            ax.axvspan(center - band_half, center + band_half,
                       color='#f0f0f0', zorder=0)

    legend_items = [
        Patch(facecolor=mcolors.to_rgba(color_lr, 0.55), edgecolor=color_lr,
              linewidth=1.2, label='Logistic Regression'),
        Patch(facecolor=mcolors.to_rgba(color_ann, 0.55), edgecolor=color_ann,
              linewidth=1.2, label='ANN'),
    ]
    ax.legend(handles=legend_items, loc='upper right', frameon=True, fontsize=11)
    ax.grid(False)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"  ✓ Saved: {save_path}")


def plot_corr_bootstrap_spaghetti(
    lr_summary, ann_summary, color_lr, color_ann, filename, n_resample=200, color_ann_cii=None
):
    """
    Version 2 — Spaghetti line chart of bootstrapped correlations.
    200 faint lines per panel; median line is bold with annotations.
    2×2 layout: [LR opt-mape, ANN opt-mape] / [LR opt-cii, ANN opt-cii]
    """
    x_vals = [d['n'] for d in lr_summary]
    x_pos = np.arange(len(x_vals))  # evenly-spaced positions

    color_cii_ann_val = color_ann_cii if color_ann_cii is not None else color_ann

    configs = [
        (lr_summary,  'mape_values', 'MAPE', color_lr,  'Logistic Regression', 'o'),
        (ann_summary, 'mape_values', 'MAPE', color_ann, 'ANN',                 'o'),
        (lr_summary,  'cii_values',  'CII',  color_lr,  'Logistic Regression', 'D'),
        (ann_summary, 'cii_values',  'CII',  color_cii_ann_val, 'ANN',         'D'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.patch.set_facecolor('white')

    for idx, (summary, metric_key, metric_label, color, model_name, marker) in enumerate(configs):
        row, col = divmod(idx, 2)
        ax = axes[row, col]
        ax.set_facecolor('white')

        # Compute bootstrapped correlations at each sample size
        all_corrs = []  # list of arrays, each (n_resample,)
        for d in summary:
            corrs = bootstrap_correlations(
                np.asarray(d['optimism_values'], dtype=float),
                np.asarray(d[metric_key], dtype=float),
                n_resample=n_resample
            )
            all_corrs.append(corrs)
        all_corrs = np.array(all_corrs)  # shape: (n_sample_sizes, n_resample)

        # Compute original p-values at each sample size
        p_values = []
        for d in summary:
            x_vals_orig = np.asarray(d['optimism_values'], dtype=float)
            y_vals_orig = np.asarray(d[metric_key], dtype=float)
            if np.std(x_vals_orig) == 0 or np.std(y_vals_orig) == 0:
                p_values.append(np.nan)
            else:
                _, p_val = pearsonr(x_vals_orig, y_vals_orig)
                p_values.append(p_val)

        # Plot individual bootstrap lines (very faint)
        for j in range(n_resample):
            ax.plot(x_pos, all_corrs[:, j],
                    color=color, alpha=0.04, linewidth=0.6, zorder=1)

        # Plot median line (bold) — skip NaN points
        medians = np.nanmedian(all_corrs, axis=1)
        valid = ~np.isnan(medians)
        ax.plot(x_pos[valid], medians[valid], color=color, linewidth=3.0,
                marker=marker, markersize=7 if marker == 'o' else 6, markeredgecolor='white',
                markeredgewidth=1.5, label='Median r', zorder=10)
        # Mark NaN points with an 'x'
        if (~valid).any():
            ax.scatter(x_pos[~valid], np.zeros((~valid).sum()), marker='x',
                       s=80, color=color, linewidths=2.5, zorder=10)

        # Annotate median values and p-values (alternate above/below to avoid overlap)
        offsets = [(0, 16), (0, -22), (0, 16), (0, -22), (0, 16), (0, -22)]
        for i, (xp, med) in enumerate(zip(x_pos, medians)):
            ofs = offsets[i % len(offsets)]
            p_val = p_values[i]
            
            if np.isnan(med):
                label = 'N/A'
            else:
                if np.isnan(p_val):
                    p_str = 'p = N/A'
                elif p_val < 0.001:
                    p_str = 'p < 0.001'
                else:
                    p_str = f'p = {p_val:.3f}'
                label = f'r = {med:.3f}\n({p_str})'
                
            ax.annotate(
                label, (xp, med if not np.isnan(med) else 0),
                textcoords='offset points', xytext=ofs,
                ha='center', fontsize=8.0, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.7),
                clip_on=False
            )

        ax.axhline(0, color='grey', linestyle=':', linewidth=0.8, alpha=0.6)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"n = {n}\n(EPV = {EPV_MAPPING[n]:.2f})" for n in x_vals])
        ax.set_xlabel('Sample Size and Events Per Variable (EPV)', fontsize=12)
        ax.set_ylabel(f'Pearson r (Optimism compared to {metric_label})', fontsize=12)
        ax.set_title(f'{model_name} — Correlation: Optimism compared to {metric_label}',
                     fontsize=13)
        ax.legend(loc='upper right', frameon=True, fontsize=11)
        ax.grid(True, linestyle=':', alpha=0.4)

    fig.suptitle(
        'Per-Bootstrap Correlation: Optimism compared to Instability Metrics\n'
        '(200 bootstrap resamples per sample size; median line bolded)',
        fontsize=15, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"  ✓ Saved: {save_path}")


# ── Main ─────────────────────────────────────────────────────────
if __name__ == '__main__':
    df = None  # lazy-loaded only if optimism needs recomputation

    print("── Building Logistic Regression summary ──")
    lr_dir = os.path.join(RESULTS, 'Logistic Regression')
    lr_summary = build_summary('Logistic Regression', lr_dir, df)

    print("\n── Building ANN summary ──")
    ann_dir = os.path.join(RESULTS, 'ANN')
    ann_summary = build_summary('ANN', ann_dir, df)

    # Curated scientific palette
    CLR_LR      = '#2b7b9f'  # Slate-teal blue for LR
    CLR_ANN     = '#ff7f0e'  # Vibrant orange for ANN
    CLR_ANN_CII = CLR_ANN    # Set ANN CII color to orange to enforce the one-model-one-color scheme

    # ── Box Plots ──
    # 1. Combined Optimism vs MAPE
    print("\nPlotting combined Optimism vs MAPE (box plots)...")
    plot_combined_boxplots(
        lr_summary, ann_summary,
        key1='optimism_values', key2='mape_values',
        label1='Optimism', label2='MAPE',
        color_lr=CLR_LR,  color_ann=CLR_ANN,
        title='Optimism and MAPE across Sample Sizes',
        filename='combined_optimism_vs_mape.png'
    )

    # 2. Combined Optimism vs CII
    print("Plotting combined Optimism vs CII (box plots)...")
    plot_combined_boxplots(
        lr_summary, ann_summary,
        key1='optimism_values', key2='cii_values',
        label1='Optimism', label2='CII',
        color_lr=CLR_LR,  color_ann=CLR_ANN,
        title='Optimism and CII across Sample Sizes',
        filename='combined_optimism_vs_cii.png',
        edge_style2='--',
        color_ann_right=CLR_ANN_CII
    )

    # 3. All three metrics in one chart
    print("Plotting combined Optimism + MAPE + CII (box plots)...")
    plot_all_three_boxplots(
        lr_summary, ann_summary,
        color_lr=CLR_LR, color_ann=CLR_ANN,
        filename='combined_all_three.png',
        color_ann_cii=CLR_ANN_CII
    )

    # ── Median + Scatter Plots ──
    # 4. Optimism vs MAPE (median + scatter)
    print("\nPlotting Optimism vs MAPE (median + scatter)...")
    plot_combined_median_scatter(
        lr_summary, ann_summary,
        key1='optimism_values', key2='mape_values',
        label1='Optimism', label2='MAPE',
        color_lr=CLR_LR,  color_ann=CLR_ANN,
        title='Optimism and MAPE across Sample Sizes (Median + Scatter)',
        filename='scatter_optimism_vs_mape.png',
        marker2='o'
    )

    # 5. Optimism vs CII (median + scatter)
    print("Plotting Optimism vs CII (median + scatter)...")
    plot_combined_median_scatter(
        lr_summary, ann_summary,
        key1='optimism_values', key2='cii_values',
        label1='Optimism', label2='CII',
        color_lr=CLR_LR,  color_ann=CLR_ANN,
        title='Optimism and CII across Sample Sizes (Median + Scatter)',
        filename='scatter_optimism_vs_cii.png',
        marker2='D',
        color_ann_right=CLR_ANN_CII
    )

    # 6. All three metrics (median + scatter)
    print("Plotting Optimism + MAPE + CII (median + scatter)...")
    plot_all_three_median_scatter(
        lr_summary, ann_summary,
        color_lr=CLR_LR, color_ann=CLR_ANN,
        filename='scatter_all_three.png',
        color_ann_cii=CLR_ANN_CII
    )

    # 6b. Model-specific stacked scatter charts (Optimism vs MAPE & CII)
    # Using a distinct metric-based color scheme
    CLR_METRIC_OPT  = '#2c3e50'  # Slate/Navy for Optimism
    CLR_METRIC_MAPE = '#27ae60'  # Green for MAPE
    CLR_METRIC_CII  = '#8e44ad'  # Purple for CII

    print("Plotting Logistic Regression Optimism vs MAPE + CII (stacked scatter)...")
    plot_single_model_mean_scatter_stacked(
        lr_summary,
        color_opt=CLR_METRIC_OPT,
        color_mape=CLR_METRIC_MAPE,
        color_cii=CLR_METRIC_CII,
        model_name='Logistic Regression',
        filename='scatter_optimism_mape_cii_lr.png'
    )

    print("Plotting ANN Optimism vs MAPE + CII (stacked scatter)...")
    plot_single_model_mean_scatter_stacked(
        ann_summary,
        color_opt=CLR_METRIC_OPT,
        color_mape=CLR_METRIC_MAPE,
        color_cii=CLR_METRIC_CII,
        model_name='ANN',
        filename='scatter_optimism_mape_cii_ann.png'
    )

    # 7. Combined clean Median Trends with Shaded IQR and 5th-95th Percentile Bands (No Scatter points)
    print("\nPlotting clean stability metric trends with shaded percentile bands...")
    plot_median_trends_with_bands(
        lr_summary, ann_summary,
        color_lr=CLR_LR, color_ann=CLR_ANN,
        color_ann_cii=CLR_ANN_CII,
        filename='median_trends_with_bands.png'
    )

    # ── Per-bootstrap correlation analysis ──
    # Version 1: Boxplots of bootstrapped correlations
    print("\nPlotting bootstrapped correlation boxplots: Optimism vs MAPE...")
    plot_corr_bootstrap_boxplots(
        lr_summary, ann_summary,
        metric_key='mape_values', metric_label='MAPE',
        color_lr=CLR_LR, color_ann=CLR_ANN,
        filename='corr_bootstrap_boxplot_opt_mape.png',
        marker='o'
    )

    print("Plotting bootstrapped correlation boxplots: Optimism vs CII...")
    plot_corr_bootstrap_boxplots(
        lr_summary, ann_summary,
        metric_key='cii_values', metric_label='CII',
        color_lr=CLR_LR, color_ann=CLR_ANN_CII,
        filename='corr_bootstrap_boxplot_opt_cii.png',
        marker='D'
    )

    # Version 2: Spaghetti line charts
    print("Plotting bootstrapped correlation spaghetti line charts...")
    plot_corr_bootstrap_spaghetti(
        lr_summary, ann_summary,
        color_lr=CLR_LR, color_ann=CLR_ANN,
        color_ann_cii=CLR_ANN_CII,
        filename='corr_bootstrap_spaghetti.png'
    )

    print("\nDone! Combined figures saved to:", OUTPUT_DIR)
