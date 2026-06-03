"""
PH界面特化記述子 + RDF 比較ML:
新記述子: 最大永続性H2特徴量 (birth, death, persistence, n_pairs_gt3)
         + Na+グラフェン近傍密度 (z密度プロキシ)
比較: pd_vector_PCA vs 新PH記述子 vs RDF+zdensity
"""

import numpy as np
import pandas as pd
import subprocess, io, os, tempfile
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from concurrent.futures import ProcessPoolExecutor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold, train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO_ROOT  = Path(__file__).resolve().parent.parent
CACHE_DIR  = str(REPO_ROOT / 'cache')
HC_TXT_DIR = str(REPO_ROOT / 'cache' / 'homcloud_txt')
OUT_DIR    = REPO_ROOT / 'results'
DATA_DIR   = REPO_ROOT / 'data'
# Source folder on the original Windows host for raw homcloud txts.
RAW_WIN_BASE = r'D:\path\to\raw\trajectories'
PH_FEAT_CACHE = f'{CACHE_DIR}/ph_targeted_features.npy'
TEMPS      = [250, 300, 350, 400, 450]
SNAP_NS    = [100, 60, 70, 80, 90]

# ============================================================
# 1. homcloud txt ファイルを D: からコピー
# ============================================================
def copy_homcloud_txts():
    os.makedirs(HC_TXT_DIR, exist_ok=True)
    existing = len([f for f in os.listdir(HC_TXT_DIR) if f.endswith('.txt')])
    if existing >= 500:
        print(f"  homcloud txtキャッシュ済み: {existing}ファイル")
        return
    print(f"  {RAW_WIN_BASE} からhomcloud txtをコピー中... ({existing}/500 済み)")
    home = str(Path.home())
    distro = os.environ.get('WSL_DISTRO_NAME', 'Ubuntu')
    unc_dst = HC_TXT_DIR.replace(home, fr'\\wsl.localhost\{distro}{home}')
    ps_cmd = f'Copy-Item -Path "{RAW_WIN_BASE}\\homcloud\\*.txt" -Destination "{unc_dst}"'
    result = subprocess.run(['powershell.exe', '-Command', ps_cmd],
                            capture_output=True, text=True, timeout=300)
    done = len([f for f in os.listdir(HC_TXT_DIR) if f.endswith('.txt')])
    print(f"  コピー完了: {done}/500")

# ============================================================
# 2. HomCloud特徴量を1サンプル計算 (境界マップなし)
# ============================================================
def compute_ph_one(p_idx):
    """1サンプルのHoma量H2最大永続性特徴量を計算"""
    import homcloud.interface as hc
    import numpy as np, tempfile, os

    txt_path = os.path.join(HC_TXT_DIR, f'{p_idx:03d}.txt')
    if not os.path.exists(txt_path):
        return None

    try:
        coords = np.loadtxt(txt_path)
        if coords.shape[0] < 10:
            return None

        # グラフェン原子の特定 (z ≈ -3.35 Å)
        is_gra = np.abs(coords[:, 2] - (-3.35)) < 0.3
        n_gra = is_gra.sum()

        with tempfile.NamedTemporaryFile(suffix='.idiagram', delete=False) as f:
            tmp_path = f.name

        try:
            PDList = hc.PDList.from_alpha_filtration(
                coords, save_to=tmp_path, save_boundary_map=False)
            pd2 = PDList.dth_diagram(2)
            pairs = sorted(pd2.pairs(), key=lambda p: p.death - p.birth, reverse=True)
        finally:
            try: os.unlink(tmp_path)
            except: pass

        if not pairs:
            return {'p_idx': p_idx, 'birth': np.nan, 'death': np.nan,
                    'persistence': np.nan, 'n_pairs_gt3': 0, 'n_gra': n_gra}

        top = pairs[0]
        return {
            'p_idx':       p_idx,
            'birth':       top.birth,
            'death':       top.death,
            'persistence': top.death - top.birth,
            'n_pairs_gt3': sum(1 for p in pairs if (p.death - p.birth) > 3),
            'n_gra':       n_gra,
        }
    except Exception as e:
        return {'p_idx': p_idx, 'birth': np.nan, 'death': np.nan,
                'persistence': np.nan, 'n_pairs_gt3': 0, 'n_gra': 0}

# ============================================================
# 3. 並列バッチ計算
# ============================================================
def compute_all_ph_features():
    if os.path.exists(PH_FEAT_CACHE):
        print("  PH特徴量キャッシュを使用")
        return np.load(PH_FEAT_CACHE, allow_pickle=True).item()

    print("  並列HomCloud計算中 (500サンプル, 6コア)...")
    import time
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=6) as ex:
        results = list(ex.map(compute_ph_one, range(500)))

    elapsed = time.time() - t0
    print(f"  完了: {elapsed:.0f}秒 ({elapsed/60:.1f}分)")

    # 辞書に変換
    feat_dict = {}
    for r in results:
        if r is not None:
            feat_dict[r['p_idx']] = r

    np.save(PH_FEAT_CACHE, feat_dict)
    print(f"  PH特徴量保存: {PH_FEAT_CACHE}")
    return feat_dict

# ============================================================
# 補助関数
# ============================================================
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
                        rows.append({'sim_id': sim_id, 'il_type': il_type,
                                     'comp_idx': comp_idx, 'temp': temp,
                                     'efield': ef_idx, 'snap_idx': snap_idx,
                                     'snap_ns': SNAP_NS[snap_idx], 'p_idx': p_idx})
                    sim_id += 1
    return pd.DataFrame(rows)

def make_pca_features(X_raw, n_comp=20, name=''):
    vt = VarianceThreshold(threshold=1e-10)
    X_vt = vt.fit_transform(X_raw)
    sc = StandardScaler()
    X_sc = sc.fit_transform(X_vt)
    n = min(n_comp, X_sc.shape[1], X_sc.shape[0]-1)
    pca = PCA(n_components=n, random_state=42)
    Xp = pca.fit_transform(X_sc)
    print(f"  [{name}] {X_raw.shape[1]}→{X_vt.shape[1]}→PCA{n}, "
          f"累積寄与率={np.cumsum(pca.explained_variance_ratio_)[-1]:.3f}")
    return Xp, pca, sc, vt

def eval_model(X_tr, X_te, y_tr, y_te, g_tr, name):
    model = ExtraTreesRegressor(n_estimators=500, max_depth=5, max_features=0.5,
                                min_samples_leaf=2, random_state=42, n_jobs=-1)
    gkf = GroupKFold(n_splits=5)
    cv = cross_val_score(model, X_tr, y_tr, groups=g_tr, cv=gkf, scoring='r2')
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    r2_te = r2_score(y_te, y_pred)
    r2_tr = r2_score(y_tr, model.predict(X_tr))
    rmse_real = np.sqrt(mean_squared_error(np.expm1(y_te), np.expm1(y_pred)))
    print(f"[{name}]")
    print(f"  Test R²={r2_te:.4f}, Train R²={r2_tr:.4f}")
    print(f"  RMSE(実)={rmse_real:.2f} Å²/ns, CV R²={cv.mean():.4f}±{cv.std():.4f}")
    return {'name': name, 'R2_test': r2_te, 'R2_train': r2_tr,
            'RMSE_real': rmse_real, 'CV_R2': cv.mean(), 'CV_std': cv.std(),
            'model': model}

# ============================================================
# メイン
# ============================================================
def main():
    print("=" * 70)
    print("PH界面特化記述子 + RDF 比較ML")
    print("=" * 70)

    # ---- データ準備 ----
    print("\n=== データ読み込み ===")
    copy_homcloud_txts()
    ph_dict = compute_all_ph_features()

    df_sim   = build_sim_list()
    dc       = pd.read_csv(DATA_DIR / 'D_cumulative.csv').set_index('p_idx')
    pd_vecs  = np.load(f'{CACHE_DIR}/pd_vectors.npy')
    rdf_mat  = np.load(f'{CACHE_DIR}/rdf_matrix.npy')
    zd_mat   = np.load(f'{CACHE_DIR}/zdensity_matrix.npy')

    pidxs  = df_sim['p_idx'].values
    groups = df_sim['sim_id'].values
    y_raw  = np.array([dc.loc[p,'D_cumul'] if p in dc.index else np.nan for p in pidxs])
    y_log  = np.log1p(y_raw)

    valid = ~np.isnan(y_raw)
    print(f"有効サンプル: {valid.sum()}/500")

    df_v  = df_sim[valid].reset_index(drop=True)
    y     = y_log[valid]
    g     = groups[valid]

    # ---- 新PH記述子の構築 ----
    print("\n=== 新PH記述子 (birth, death, persistence, n_pairs_gt3, Na近傍密度) ===")

    # HomCloudから: birth, death, persistence, n_pairs_gt3
    ph_cols = ['birth', 'death', 'persistence', 'n_pairs_gt3']
    ph_raw = np.full((valid.sum(), len(ph_cols)), np.nan)
    for i, p_idx in enumerate(pidxs[valid]):
        if p_idx in ph_dict:
            r = ph_dict[p_idx]
            ph_raw[i] = [r['birth'], r['death'], r['persistence'], r['n_pairs_gt3']]

    # NaN補完 (中央値)
    for j in range(ph_raw.shape[1]):
        col = ph_raw[:, j]
        col[np.isnan(col)] = np.nanmedian(col)
        ph_raw[:, j] = col

    # z密度プロキシ: Na+のグラフェン近傍密度
    # グラフェンz≈-3.35Å → 換算座標≈0.57 → ビン28付近
    # 界面Na+は換算z=0.55-0.70 (ビン27-34) に集中
    zd_valid = np.nan_to_num(zd_mat[pidxs[valid]])
    na_near_gra_raw = zd_valid[:, 27:35].sum(axis=1, keepdims=True)  # グラフェン近傍Na+密度
    na_far_gra_raw  = zd_valid[:, :10].sum(axis=1, keepdims=True)    # ボックス端部

    # 全新PH記述子 = (birth, death, persistence, n_pairs_gt3, Na_near_gra)
    ph_all = np.hstack([ph_raw, na_near_gra_raw, na_far_gra_raw])

    print(f"birth 範囲: {ph_raw[:,0].min():.3f} ~ {ph_raw[:,0].max():.3f} Å²")
    print(f"persistence 範囲: {ph_raw[:,2].min():.3f} ~ {ph_raw[:,2].max():.3f} Å²")
    print(f"n_pairs_gt3 範囲: {ph_raw[:,3].min():.0f} ~ {ph_raw[:,3].max():.0f}")
    print(f"Na_near_gra 範囲: {na_near_gra_raw.min():.4f} ~ {na_near_gra_raw.max():.4f}")

    # 新PH記述子とD_cumulの相関
    print("\n各新記述子 vs log1p(D) Pearson相関:")
    col_names = ['birth', 'death', 'persistence', 'n_pairs_gt3', 'Na_near_gra', 'Na_far']
    for j, cname in enumerate(col_names):
        r, p = pearsonr(ph_all[:, j], y)
        print(f"  {cname:15s}: r={r:+.3f}, p={p:.2e}")

    # ---- 特徴量エンジニアリング ----
    print("\n=== 特徴量準備 ===")
    X_pd_pca, *_  = make_pca_features(pd_vecs[pidxs[valid]], 20, 'pd_vector')
    X_rdf_pca, *_ = make_pca_features(np.nan_to_num(rdf_mat[pidxs[valid]]), 20, 'RDF')
    X_zd_pca, *_  = make_pca_features(np.nan_to_num(zd_mat[pidxs[valid]]), 20, 'z密度')

    # 新PH記述子 (標準化)
    sc_ph = StandardScaler()
    X_ph_sc = sc_ph.fit_transform(ph_all)

    # 物理パラメータ
    sc_phys = StandardScaler()
    X_phys_raw = np.column_stack([
        (df_v['il_type'] == 'TFSI').astype(float),
        df_v['temp'].values / 450.0,
        df_v['comp_idx'].values / 4.0,
        df_v['efield'].values.astype(float),
        df_v['snap_ns'].values / 100.0,
    ])
    X_phys_sc = sc_phys.fit_transform(X_phys_raw)

    # ---- Train/Test 分割 ----
    unique_sims = np.unique(g)
    train_sims, test_sims = train_test_split(unique_sims, test_size=0.3, random_state=42)
    tr = np.isin(g, train_sims)
    te = np.isin(g, test_sims)
    g_tr = g[tr]

    def split(X): return X[tr], X[te]
    y_tr, y_te = y[tr], y[te]
    print(f"\nTrain: {tr.sum()} (シミュ: {len(train_sims)})")
    print(f"Test:  {te.sum()} (シミュ: {len(test_sims)})")

    # ---- MLモデル比較 ----
    print("\n" + "=" * 70)
    print("=== MLモデル比較 ===")
    print("=" * 70)
    results = []

    # 1. pd_vector + phys (v3参照)
    X1 = np.hstack([X_pd_pca, X_phys_sc])
    results.append(eval_model(*split(X1), y_tr, y_te, g_tr, 'pd_vector+phys (v3参照)'))

    # 2. 新PH記述子のみ + phys
    X2 = np.hstack([X_ph_sc, X_phys_sc])
    results.append(eval_model(*split(X2), y_tr, y_te, g_tr, '新PH記述子+phys'))

    # 3. pd_vector + 新PH記述子 + phys
    X3 = np.hstack([X_pd_pca, X_ph_sc, X_phys_sc])
    results.append(eval_model(*split(X3), y_tr, y_te, g_tr, 'pd_vector+新PH+phys'))

    # 4. RDF + zdensity + phys (現最良参照)
    X4 = np.hstack([X_rdf_pca, X_zd_pca, X_phys_sc])
    results.append(eval_model(*split(X4), y_tr, y_te, g_tr, 'RDF+zdensity+phys (現最良)'))

    # 5. RDF + zdensity + 新PH記述子 + phys
    X5 = np.hstack([X_rdf_pca, X_zd_pca, X_ph_sc, X_phys_sc])
    results.append(eval_model(*split(X5), y_tr, y_te, g_tr, 'RDF+zdensity+新PH+phys'))

    # 6. pd_vector + RDF + zdensity + phys
    X6 = np.hstack([X_pd_pca, X_rdf_pca, X_zd_pca, X_phys_sc])
    results.append(eval_model(*split(X6), y_tr, y_te, g_tr, 'pd+RDF+zdensity+phys'))

    # 7. 全特徴量
    X7 = np.hstack([X_pd_pca, X_ph_sc, X_rdf_pca, X_zd_pca, X_phys_sc])
    results.append(eval_model(*split(X7), y_tr, y_te, g_tr, '全特徴量'))

    # ---- まとめ ----
    df_res = pd.DataFrame(results).sort_values('CV_R2', ascending=False)
    df_res_out = df_res.drop(columns='model')

    print("\n" + "=" * 70)
    print("=== まとめ (CV R²降順) ===")
    print("=" * 70)
    print(df_res_out[['name','R2_test','R2_train','RMSE_real','CV_R2']].to_string(index=False))
    df_res_out.to_csv(OUT_DIR / 'ph_targeted_results.csv', index=False)

    # ---- 新PH記述子の寄与分析 ----
    print("\n=== 新PH記述子の特徴重要度 (RDF+zdensity+新PH+physモデル) ===")
    model5 = [r['model'] for r in results if r['name'] == 'RDF+zdensity+新PH+phys'][0]
    n_rdf = X_rdf_pca.shape[1]
    n_zd  = X_zd_pca.shape[1]
    n_ph  = X_ph_sc.shape[1]
    n_phy = X_phys_sc.shape[1]
    imp = model5.feature_importances_
    # グループ別重要度
    groups_imp = {
        'RDF_PCA':     imp[:n_rdf].sum(),
        'zdensity_PCA': imp[n_rdf:n_rdf+n_zd].sum(),
        '新PH記述子':   imp[n_rdf+n_zd:n_rdf+n_zd+n_ph].sum(),
        '物理パラメータ': imp[n_rdf+n_zd+n_ph:].sum(),
    }
    for k, v in groups_imp.items():
        print(f"  {k:20s}: {v:.4f} ({v*100:.1f}%)")
    # 新PH記述子の個別重要度
    ph_imp = imp[n_rdf+n_zd:n_rdf+n_zd+n_ph]
    print("\n  新PH記述子の個別重要度:")
    ph_full_names = ['birth_α(Å²)', 'death_α(Å²)', 'persistence(Å²)',
                     'n_pairs_gt3', 'Na近傍密度(z=0.55-0.70)', 'Na端部密度(z=0-0.19)']
    for nm, v in zip(ph_full_names, ph_imp):
        print(f"    {nm:30s}: {v:.4f} ({v*100:.1f}%)")

    # ---- プロット ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # (a) CV R² 比較棒グラフ
    ax = axes[0]
    names   = df_res['name'].values
    cv_vals = df_res['CV_R2'].values
    te_vals = df_res['R2_test'].values
    x = np.arange(len(names))
    bars1 = ax.bar(x-0.2, cv_vals, 0.38, label='CV R²', color='steelblue', alpha=0.85)
    bars2 = ax.bar(x+0.2, te_vals, 0.38, label='Test R²', color='coral', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha='right', fontsize=7.5)
    ax.set_ylabel('R²')
    ax.set_title('MLモデル比較 (CV R²降順)', fontsize=10)
    ax.legend(fontsize=8)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_ylim(-0.05, 0.90)
    for bar in bars1:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=6)

    # (b) 新PH記述子 vs D の散布図
    ax = axes[1]
    sc = ax.scatter(ph_raw[:, 0], ph_raw[:, 2],
                   c=y, cmap='coolwarm', s=10, alpha=0.7)
    plt.colorbar(sc, ax=ax, label='log1p(D_cumul)')
    ax.set_xlabel('birth α (Å²) — 最大永続性H2特徴', fontsize=9)
    ax.set_ylabel('persistence (Å²)', fontsize=9)
    ax.set_title('新PH記述子の分布\n色: log1p(D_cumul)', fontsize=9)
    # 250K/450K の注記
    ax.axvline(x=4.37, color='blue', lw=1.0, ls='--', alpha=0.7, label='250K界面特徴 birth')
    ax.axvline(x=4.89, color='red',  lw=1.0, ls='--', alpha=0.7, label='450Kバルク特徴 birth')
    ax.legend(fontsize=7)

    # (c) 特徴量グループ重要度 (RDF+zdensity+新PH+physモデル)
    ax = axes[2]
    group_names = list(groups_imp.keys())
    group_vals  = list(groups_imp.values())
    colors = ['steelblue', 'green', 'red', 'gray']
    ax.bar(group_names, group_vals, color=colors, alpha=0.8)
    ax.set_ylabel('特徴量重要度 (合計)', fontsize=9)
    ax.set_title('RDF+zdensity+新PH+phys\n特徴量グループ別重要度', fontsize=9)
    ax.tick_params(axis='x', rotation=20)
    for i, (nm, v) in enumerate(zip(group_names, group_vals)):
        ax.text(i, v+0.002, f'{v:.3f}\n({v*100:.1f}%)', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    fig.savefig(OUT_DIR / 'ph_targeted_ml.png', dpi=150, bbox_inches='tight')
    print("\n図保存: ph_targeted_ml.png")

    # ---- 新PH記述子の温度別統計 ----
    print("\n=== 新PH記述子の温度依存性 ===")
    print(f"{'温度':>5} {'BF4 birth':>11} {'BF4 persist':>12} {'BF4 n_gt3':>10} {'TFSI birth':>11} {'TFSI persist':>13}")
    for temp in TEMPS:
        for il, vals in [('BF4', []), ('TFSI', [])]:
            mask = (df_v['temp']==temp) & (df_v['il_type']==il) & (df_v['efield']==0)
            if mask.sum() > 0:
                b = ph_raw[mask, 0].mean()
                p = ph_raw[mask, 2].mean()
                n = ph_raw[mask, 3].mean()
                if il == 'BF4':
                    print(f"{temp:>5} {b:>11.4f} {p:>12.4f} {n:>10.2f}", end='')
                else:
                    print(f" {b:>11.4f} {p:>13.4f}")

    print("\n=== 完了 ===")

if __name__ == '__main__':
    main()
