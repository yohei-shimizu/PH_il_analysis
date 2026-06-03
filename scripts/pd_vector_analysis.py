"""
pd_vector解析:
1. 各ビンとD_cumulのPearson相関マップ (2D birth-death空間で可視化)
2. 残差解析: RDF+z密度の残差をpd_vectorで説明できるか
3. H2ダイアグラム条件別可視化: 温度・IL種・電場・D値の高低
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold, train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = str(REPO_ROOT / 'cache')
OUT_DIR   = REPO_ROOT / 'results'
DATA_DIR  = REPO_ROOT / 'data'
GRID_SIZE = 64
GRID_MAX  = 20.0
BIN_WIDTH = GRID_MAX / GRID_SIZE  # 0.3125 Å
TEMPS = [250, 300, 350, 400, 450]
SNAP_NS  = [100, 60, 70, 80, 90]
SNAP_SEG = [10,   6,  7,  8,  9]

# birth/death 軸の中心値
bin_centers = (np.arange(GRID_SIZE) + 0.5) * BIN_WIDTH  # 0..20Å

# ============================================================
# 補助関数
# ============================================================
def vec_to_2d(vec):
    """2080次元pd_vectorを64×64の2Dマップに変換 (上三角部分)"""
    mat = np.zeros((GRID_SIZE, GRID_SIZE))
    rows, cols = np.triu_indices(GRID_SIZE, k=0)
    mat[rows, cols] = vec
    return mat

def build_sim_list():
    rows = []
    sim_id = 0
    for il_type in ['BF4', 'TFSI']:
        p_base = 0 if il_type == 'BF4' else 250
        for comp_idx in range(5):
            for temp_idx, temp in enumerate(TEMPS):
                for ef_idx in range(2):
                    for snap_idx in range(5):
                        p_idx = p_base + comp_idx*50 + temp_idx*10 + ef_idx*5 + snap_idx
                        rows.append({
                            'sim_id': sim_id, 'il_type': il_type,
                            'comp_idx': comp_idx, 'temp': temp,
                            'efield': ef_idx, 'snap_idx': snap_idx,
                            'snap_ns': SNAP_NS[snap_idx], 'p_idx': p_idx,
                        })
                    sim_id += 1
    return pd.DataFrame(rows)

def make_pca_features(X_raw, n_components=20):
    vt = VarianceThreshold(threshold=1e-10)
    X_vt = vt.fit_transform(X_raw)
    sc = StandardScaler()
    X_sc = sc.fit_transform(X_vt)
    n_comp = min(n_components, X_sc.shape[1], X_sc.shape[0]-1)
    pca = PCA(n_components=n_comp, random_state=42)
    return pca.fit_transform(X_sc), pca, sc, vt

def eval_model(X_tr, y_tr, X_te, y_te, g_tr, n_estimators=300):
    model = ExtraTreesRegressor(
        n_estimators=n_estimators, max_depth=5, max_features=0.5,
        min_samples_leaf=2, random_state=42, n_jobs=-1)
    gkf = GroupKFold(n_splits=5)
    cv = cross_val_score(model, X_tr, y_tr, groups=g_tr, cv=gkf, scoring='r2')
    model.fit(X_tr, y_tr)
    r2_te = r2_score(y_te, model.predict(X_te))
    return model, r2_te, cv.mean(), cv.std()

# ============================================================
# メイン
# ============================================================
def main():
    print("=" * 70)
    print("pd_vector 詳細解析: 相関マップ + 残差解析 + H2可視化")
    print("=" * 70)

    # ---- データ読み込み ----
    df_sim = build_sim_list()
    dc = pd.read_csv(DATA_DIR / 'D_cumulative.csv').set_index('p_idx')
    pd_vecs = np.load(f'{CACHE_DIR}/pd_vectors.npy')         # (500, 2080)
    rdf_mat  = np.load(f'{CACHE_DIR}/rdf_matrix.npy')        # (500, 600)
    zd_mat   = np.load(f'{CACHE_DIR}/zdensity_matrix.npy')   # (500, 50)

    # D_cumul を p_idx 順に並べる
    pidxs  = df_sim['p_idx'].values
    groups = df_sim['sim_id'].values
    y_raw  = np.array([dc.loc[p, 'D_cumul'] if p in dc.index else np.nan for p in pidxs])
    y_log  = np.log1p(y_raw)

    valid = ~np.isnan(y_raw)
    N = valid.sum()
    df_v   = df_sim[valid].reset_index(drop=True)
    pd_v   = pd_vecs[pidxs[valid]]   # (500, 2080)
    rdf_v  = np.nan_to_num(rdf_mat[pidxs[valid]])
    zd_v   = np.nan_to_num(zd_mat[pidxs[valid]])
    y      = y_log[valid]
    y_D    = y_raw[valid]
    g      = groups[valid]

    print(f"サンプル数: {N}, D範囲: {y_D.min():.2f} ~ {y_D.max():.2f}")

    # ============================================================
    # [1] pd_vector 各ビンとD_cumulのPearson相関マップ
    # ============================================================
    print("\n[1] pd_vector各ビン vs D_cumul Pearson相関マップ計算中...")
    corr_vec  = np.zeros(2080)
    pval_vec  = np.ones(2080)
    for k in range(2080):
        col = pd_v[:, k]
        mask = col > 0  # 非ゼロのサンプルのみ
        if mask.sum() > 20:
            r, p = pearsonr(col[mask], y_log[valid][mask])
            corr_vec[k] = r
            pval_vec[k] = p

    corr_map = vec_to_2d(corr_vec)  # (64, 64)
    pval_map = vec_to_2d(pval_vec)

    # 有意なビン (p<0.05) の相関係数
    sig_mask = (pval_vec < 0.05) & (corr_vec != 0)
    print(f"  有意なビン(p<0.05): {sig_mask.sum()}/2080")
    print(f"  最大 r = {corr_vec[sig_mask].max():.3f}, 最小 r = {corr_vec[sig_mask].min():.3f}")

    # 高相関ビンのbirth/death位置を報告
    print("\n  [高相関ビン top10 (|r|>0.25)]")
    rows_t, cols_t = np.triu_indices(GRID_SIZE, k=0)
    abs_corr = np.abs(corr_vec)
    top_idx = np.argsort(abs_corr)[::-1]
    count = 0
    for idx in top_idx:
        if abs_corr[idx] > 0.25 and pval_vec[idx] < 0.05:
            bi = rows_t[idx]; di = cols_t[idx]
            birth = bin_centers[bi]; death = bin_centers[di]
            print(f"    birth={birth:.2f}Å, death={death:.2f}Å, "
                  f"persistence={death-birth:.2f}Å, r={corr_vec[idx]:+.3f}")
            count += 1
            if count >= 10: break

    # ============================================================
    # [2] 残差解析
    # ============================================================
    print("\n[2] 残差解析...")

    # train/test分割
    unique_sims = np.unique(g)
    train_sims, test_sims = train_test_split(unique_sims, test_size=0.3, random_state=42)
    tr = np.isin(g, train_sims)
    te = np.isin(g, test_sims)
    g_tr = g[tr]

    # 特徴量: RDF+z密度+物理パラメータ
    X_rdf_pca, *_ = make_pca_features(rdf_v, 20)
    X_zd_pca,  *_ = make_pca_features(zd_v, 20)
    X_pd_pca,  *_ = make_pca_features(pd_v, 20)

    sc_phys = StandardScaler()
    X_phys = np.column_stack([
        (df_v['il_type'] == 'TFSI').astype(float),
        df_v['temp'].values / 450.0,
        df_v['comp_idx'].values / 4.0,
        df_v['efield'].values.astype(float),
        df_v['snap_ns'].values / 100.0,
    ])
    X_phys_sc = sc_phys.fit_transform(X_phys)

    # モデルA: RDF+zdensity+phys → 残差を計算
    X_rz = np.hstack([X_rdf_pca, X_zd_pca, X_phys_sc])
    model_rz, r2_rz_te, cv_rz, _ = eval_model(X_rz[tr], y[tr], X_rz[te], y[te], g_tr)
    model_rz.fit(X_rz[tr], y[tr])

    # 全サンプル(train+test)の残差
    y_pred_all = model_rz.predict(X_rz)
    residuals  = y - y_pred_all  # log1p空間での残差

    print(f"  RDF+zdensity+phys: Test R²={r2_rz_te:.4f}, CV R²={cv_rz:.4f}")
    print(f"  残差 標準偏差={residuals.std():.4f}, 範囲: {residuals.min():.3f} ~ {residuals.max():.3f}")

    # モデルB: pd_vectorで残差を予測
    model_pd_res, r2_pd_res_te, cv_pd_res, _ = eval_model(
        X_pd_pca[tr], residuals[tr], X_pd_pca[te], residuals[te], g_tr)
    print(f"  pd_vector → 残差予測: Test R²={r2_pd_res_te:.4f}, CV R²={cv_pd_res:.4f}")

    # モデルC: pd_vectorでDを直接予測 (比較)
    model_pd_D, r2_pd_D_te, cv_pd_D, _ = eval_model(
        X_pd_pca[tr], y[tr], X_pd_pca[te], y[te], g_tr)
    print(f"  pd_vector → D直接予測: Test R²={r2_pd_D_te:.4f}, CV R²={cv_pd_D:.4f}")

    # モデルD: 全部合わせた場合
    X_all = np.hstack([X_rdf_pca, X_zd_pca, X_pd_pca, X_phys_sc])
    model_all, r2_all_te, cv_all, _ = eval_model(
        X_all[tr], y[tr], X_all[te], y[te], g_tr)
    print(f"  RDF+zdensity+pd+phys: Test R²={r2_all_te:.4f}, CV R²={cv_all:.4f}")

    # 偏相関: pd_vector PC1 vs D (RDF・z密度の影響を除いた後)
    from scipy.stats import pearsonr as pr
    # 残差とpd_vector PCの相関 = RDF・z密度の影響除去後のpd_vector情報
    print("\n  偏相関 (RDF+zdensity除去後のpd_vector PC vs D):")
    for i in range(min(5, X_pd_pca.shape[1])):
        r_direct, _ = pr(X_pd_pca[:, i], y)
        r_partial, _ = pr(X_pd_pca[:, i], residuals)
        print(f"    PC{i+1}: 直接相関 r={r_direct:+.3f}, 偏相関 r={r_partial:+.3f}")

    # ============================================================
    # [3] H2ダイアグラム条件別可視化
    # ============================================================
    print("\n[3] H2ダイアグラム条件別平均を計算中...")

    def avg_map(mask, label=''):
        if mask.sum() == 0:
            return np.zeros((GRID_SIZE, GRID_SIZE))
        mean_vec = pd_v[mask].mean(axis=0)
        return vec_to_2d(mean_vec)

    # 温度別
    maps_temp = {t: avg_map(df_v['temp'] == t, f'{t}K') for t in TEMPS}

    # IL種別
    map_BF4  = avg_map(df_v['il_type'] == 'BF4',  'BF4')
    map_TFSI = avg_map(df_v['il_type'] == 'TFSI', 'TFSI')

    # 電場別
    map_ef0 = avg_map(df_v['efield'] == 0, 'ef=0')
    map_ef1 = avg_map(df_v['efield'] == 1, 'ef=1')

    # D値の高低別 (中央値で分割)
    med = np.median(y_D)
    map_highD = avg_map(y_D >= med, 'HighD')
    map_lowD  = avg_map(y_D <  med, 'LowD')

    # ============================================================
    # [4] プロット
    # ============================================================
    print("\n[4] 図を作成中...")

    # 上三角マスク (birth > death の領域を白にする)
    lower_mask = np.tril(np.ones((GRID_SIZE, GRID_SIZE), dtype=bool), k=-1)

    def apply_mask(m):
        m2 = m.copy()
        m2[lower_mask] = np.nan
        return m2

    # -------- 図1: 相関マップ --------
    fig1, axes = plt.subplots(1, 2, figsize=(15, 6))

    ax = axes[0]
    cmap_corr = apply_mask(corr_map)
    im = ax.imshow(cmap_corr, origin='lower', cmap='RdBu_r', vmin=-0.5, vmax=0.5,
                   extent=[0, GRID_MAX, 0, GRID_MAX], aspect='equal')
    plt.colorbar(im, ax=ax, label='Pearson r')
    ax.set_xlabel('birth (Å)')
    ax.set_ylabel('death (Å)')
    ax.set_title('pd_vector 各ビン vs log1p(D_cumul)\nPearson 相関係数マップ')
    # 対角線
    ax.plot([0, GRID_MAX], [0, GRID_MAX], 'k--', lw=0.5, label='birth=death')
    # 物理的に意味のある領域にアノテーション
    ax.axvline(x=1.42, color='gray', lw=0.7, ls=':', label='グラフェンC-C (1.42Å)')
    ax.axvline(x=2.35, color='orange', lw=0.7, ls=':', label='Na-O 第1殻 (2.35Å)')
    ax.legend(fontsize=7, loc='upper left')

    # 有意ではないビンをマスク
    sig_map = pval_map.copy()
    sig_map[lower_mask] = np.nan
    insig_overlay = np.where(sig_map > 0.05, 0.5, np.nan)
    ax.imshow(insig_overlay, origin='lower', cmap='gray', alpha=0.4,
              extent=[0, GRID_MAX, 0, GRID_MAX], aspect='equal', vmin=0, vmax=1)

    # p値マップ
    ax2 = axes[1]
    pval_show = -np.log10(np.clip(pval_map, 1e-30, 1))
    pval_show = apply_mask(pval_show)
    im2 = ax2.imshow(pval_show, origin='lower', cmap='hot_r', vmin=0, vmax=10,
                     extent=[0, GRID_MAX, 0, GRID_MAX], aspect='equal')
    plt.colorbar(im2, ax=ax2, label='-log10(p値)')
    ax2.plot([0, GRID_MAX], [0, GRID_MAX], 'k--', lw=0.5)
    ax2.axvline(x=1.42, color='cyan', lw=0.7, ls=':')
    ax2.axvline(x=2.35, color='lime', lw=0.7, ls=':')
    ax2.set_xlabel('birth (Å)')
    ax2.set_ylabel('death (Å)')
    ax2.set_title('統計的有意性マップ (-log10 p値)\n明るいほど有意')

    plt.tight_layout()
    fig1.savefig(OUT_DIR / 'pd_corr_map.png', dpi=150, bbox_inches='tight')
    print("  pd_corr_map.png 保存")

    # -------- 図2: 残差解析 --------
    fig2, axes = plt.subplots(1, 3, figsize=(16, 5))

    # RDF+zdensity予測 vs 真値
    ax = axes[0]
    y_pred_te = model_rz.predict(X_rz[te])
    ax.scatter(y[te], y_pred_te, alpha=0.5, s=20, c=df_v['temp'].values[te],
               cmap='coolwarm', label=None)
    lim = [y.min()-0.2, y.max()+0.2]
    ax.plot(lim, lim, 'k--', lw=1)
    ax.set_xlabel('log1p(D_cumul) 真値')
    ax.set_ylabel('log1p(D_cumul) 予測')
    ax.set_title(f'RDF+zdensity+phys 予測\nTest R²={r2_rz_te:.3f}')

    # 残差 vs pd_vector予測
    ax = axes[1]
    res_pred_te = model_pd_res.predict(X_pd_pca[te])
    ax.scatter(residuals[te], res_pred_te, alpha=0.5, s=20, c=df_v['temp'].values[te],
               cmap='coolwarm')
    lim2 = [residuals.min()-0.1, residuals.max()+0.1]
    ax.plot(lim2, lim2, 'k--', lw=1)
    ax.set_xlabel('残差 (真値 - RDF予測) in log1p空間')
    ax.set_ylabel('pd_vectorによる残差予測')
    ax.set_title(f'pd_vectorが捉える追加情報\nTest R²={r2_pd_res_te:.3f}')
    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(0, color='gray', lw=0.5)

    # R²比較棒グラフ
    ax = axes[2]
    labels_r = ['RDF+zd+phys', 'pd_vector→残差', 'pd_vector→D直接', 'RDF+zd+pd+phys']
    cv_vals  = [cv_rz, cv_pd_res, cv_pd_D, cv_all]
    te_vals  = [r2_rz_te, r2_pd_res_te, r2_pd_D_te, r2_all_te]
    x = np.arange(len(labels_r))
    ax.bar(x-0.2, cv_vals, 0.4, label='CV R²', color='steelblue')
    ax.bar(x+0.2, te_vals, 0.4, label='Test R²', color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels(labels_r, rotation=20, ha='right', fontsize=8)
    ax.set_ylabel('R²')
    ax.set_title('残差解析: pd_vectorの付加価値')
    ax.legend()
    ax.axhline(0, color='k', lw=0.5)
    ax.set_ylim(-0.1, 0.85)

    plt.tight_layout()
    fig2.savefig(OUT_DIR / 'pd_residual_analysis.png', dpi=150, bbox_inches='tight')
    print("  pd_residual_analysis.png 保存")

    # -------- 図3: H2ダイアグラム条件別 --------
    fig3 = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(3, 5, figure=fig3, hspace=0.4, wspace=0.3)

    # 行1: 温度別平均H2ダイアグラム
    vmax_t = max(m.max() for m in maps_temp.values()) * 0.5
    for ti, temp in enumerate(TEMPS):
        ax = fig3.add_subplot(gs[0, ti])
        m = apply_mask(maps_temp[temp])
        # log1p変換して可視化
        im = ax.imshow(np.log1p(m), origin='lower', cmap='viridis',
                       extent=[0, GRID_MAX, 0, GRID_MAX], aspect='equal',
                       vmin=0, vmax=np.log1p(vmax_t) if vmax_t > 0 else 1)
        ax.plot([0, GRID_MAX], [0, GRID_MAX], 'w--', lw=0.5)
        ax.set_title(f'{temp}K', fontsize=10)
        ax.set_xlabel('birth (Å)', fontsize=7)
        ax.set_ylabel('death (Å)', fontsize=7)
        ax.tick_params(labelsize=6)
        plt.colorbar(im, ax=ax, shrink=0.8)

    # 行2: 差分マップ (各温度 - 350K平均)
    ref_map = maps_temp[350]
    for ti, temp in enumerate(TEMPS):
        ax = fig3.add_subplot(gs[1, ti])
        diff = apply_mask(maps_temp[temp] - ref_map)
        vd = np.nanpercentile(np.abs(diff[~np.isnan(diff)]), 95) if not np.all(np.isnan(diff)) else 1
        im = ax.imshow(diff, origin='lower', cmap='RdBu_r',
                       extent=[0, GRID_MAX, 0, GRID_MAX], aspect='equal',
                       vmin=-vd, vmax=vd)
        ax.plot([0, GRID_MAX], [0, GRID_MAX], 'k--', lw=0.5)
        ax.set_title(f'{temp}K - 350K', fontsize=10)
        ax.set_xlabel('birth (Å)', fontsize=7)
        ax.set_ylabel('death (Å)', fontsize=7)
        ax.tick_params(labelsize=6)
        plt.colorbar(im, ax=ax, shrink=0.8)

    # 行3: BF4 vs TFSI, ef0 vs ef1, High D vs Low D の差分
    pairs = [
        (map_BF4 - map_TFSI, 'BF4 - TFSI'),
        (map_ef1 - map_ef0,  'ef=1 - ef=0'),
        (map_highD - map_lowD, f'High D - Low D\n(median={np.expm1(np.log1p(med)):.1f} Å²/ns)'),
    ]
    for pi, (diff_m, title) in enumerate(pairs):
        ax = fig3.add_subplot(gs[2, pi])
        diff = apply_mask(diff_m)
        vd = np.nanpercentile(np.abs(diff[~np.isnan(diff)]), 95) if not np.all(np.isnan(diff)) else 1
        im = ax.imshow(diff, origin='lower', cmap='RdBu_r',
                       extent=[0, GRID_MAX, 0, GRID_MAX], aspect='equal',
                       vmin=-vd, vmax=vd)
        ax.plot([0, GRID_MAX], [0, GRID_MAX], 'k--', lw=0.5)
        ax.axvline(x=1.42, color='gray', lw=0.7, ls=':', label='C-C')
        ax.axvline(x=2.35, color='orange', lw=0.7, ls=':', label='Na-O')
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('birth (Å)', fontsize=7)
        ax.set_ylabel('death (Å)', fontsize=7)
        ax.tick_params(labelsize=6)
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.legend(fontsize=6)

    fig3.suptitle('H2パーシステントホモロジー: 条件別平均・差分マップ\n(birth-death空間, 0-20Å)', fontsize=12)
    fig3.savefig(OUT_DIR / 'pd_h2_diagrams.png', dpi=150, bbox_inches='tight')
    print("  pd_h2_diagrams.png 保存")

    # -------- 図4: High D vs Low D の拡大 (0-10Å範囲) --------
    fig4, axes = plt.subplots(1, 3, figsize=(16, 5))

    zoom = 10.0 / GRID_MAX  # 0-10Å に対応するピクセル割合
    n_zoom = int(GRID_SIZE * zoom)  # 0-10Å に対応するビン数

    for axi, (m, title) in enumerate([
        (map_highD, f'High D (≥{np.expm1(med):.1f} Å²/ns)'),
        (map_lowD,  f'Low D (<{np.expm1(med):.1f} Å²/ns)'),
        (map_highD - map_lowD, 'High D - Low D (差分)'),
    ]):
        ax = axes[axi]
        m_zoom = apply_mask(m)[:n_zoom, :n_zoom]
        if axi < 2:
            im = ax.imshow(np.log1p(m_zoom), origin='lower', cmap='viridis',
                           extent=[0, 10, 0, 10], aspect='equal')
        else:
            vd = np.nanpercentile(np.abs(m_zoom[~np.isnan(m_zoom)]), 95) if not np.all(np.isnan(m_zoom)) else 1
            im = ax.imshow(m_zoom, origin='lower', cmap='RdBu_r',
                           extent=[0, 10, 0, 10], aspect='equal', vmin=-vd, vmax=vd)
        plt.colorbar(im, ax=ax)
        ax.plot([0, 10], [0, 10], 'w--' if axi < 2 else 'k--', lw=0.8)
        # 物理的スケールのアノテーション
        for x_ann, lbl, col in [(1.42, 'C-C', 'cyan'), (2.35, 'Na-O', 'lime'),
                                  (3.5, '第2殻', 'yellow'), (4.7, '第3殻', 'orange')]:
            ax.axvline(x=x_ann, color=col, lw=0.8, ls=':', label=lbl)
        ax.set_xlabel('birth (Å)')
        ax.set_ylabel('death (Å)')
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=6, loc='upper left')

    plt.suptitle('H2ダイアグラム (0-10Å拡大): 高D vs 低D 比較', fontsize=11)
    plt.tight_layout()
    fig4.savefig(OUT_DIR / 'pd_h2_highD_lowD.png', dpi=150, bbox_inches='tight')
    print("  pd_h2_highD_lowD.png 保存")

    # -------- 数値サマリー --------
    print("\n" + "=" * 70)
    print("=== 解析まとめ ===")
    print("=" * 70)
    print(f"\n[相関マップ]")
    print(f"  有意(p<0.05)なビン: {sig_mask.sum()}/2080 ({100*sig_mask.sum()/2080:.1f}%)")
    print(f"  最大正相関: r={corr_vec.max():.3f}")
    print(f"  最大負相関: r={corr_vec.min():.3f}")

    print(f"\n[残差解析]")
    print(f"  RDF+zdensity+phys CV R²    : {cv_rz:.4f}")
    print(f"  pd_vector → 残差 CV R²     : {cv_pd_res:.4f}  ← RDFが見逃す情報")
    print(f"  pd_vector → D直接 CV R²    : {cv_pd_D:.4f}")
    print(f"  全特徴量 CV R²              : {cv_all:.4f}")

    print(f"\n[H2特徴量のbirth-death分布]")
    # 高D vs 低D で差の大きなビン
    diff_hilo = map_highD - map_lowD
    diff_flat = diff_hilo[~lower_mask]
    top_diff_rows, top_diff_cols = np.unravel_index(
        np.argsort(np.abs(diff_hilo[~lower_mask].ravel()))[::-1][:5],
        (GRID_SIZE, GRID_SIZE)
    )
    # 正確な上三角インデックスで取り出し
    rows_t2, cols_t2 = np.triu_indices(GRID_SIZE, k=0)
    diff_vec = np.array([diff_hilo[r, c] for r, c in zip(rows_t2, cols_t2)])
    top5 = np.argsort(np.abs(diff_vec))[::-1][:5]
    print("  High D - Low D の差が大きいビン (birth, death):")
    for idx in top5:
        bi = rows_t2[idx]; di = cols_t2[idx]
        print(f"    birth={bin_centers[bi]:.2f}Å, death={bin_centers[di]:.2f}Å, "
              f"Δ={diff_vec[idx]:+.4f}")

    print("\n=== 完了 ===")

if __name__ == '__main__':
    main()
