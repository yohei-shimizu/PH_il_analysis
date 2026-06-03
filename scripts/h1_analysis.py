"""
H1（1次パーシステントホモロジー）解析
H2 で実施した解析をすべて H1 に適用して比較する

H2 との対応:
  pd_vector_analysis.py → 本スクリプト (H1版)
  ph_targeted_ml.py     → 本スクリプト (H1版)
  homcloud_inverse_analysis.py → 本スクリプト (H1版、既存 idiagram 使用)

出力ファイル:
  cache/h1_vectors.npy          : H1 PIベクトル (500×上三角次元)
  cache/h1_features.npy         : H1 targeted features (500×6)
  h1_ml_results.csv             : ML モデル比較
  h1_corr_map.png               : H1 ビン vs D 相関マップ
  h1_pd_diagrams.png            : 条件別 H1 パーシステンス図
  h1_pd_comparison.png          : 250K vs 450K 比較
  h1_targeted_ml.png            : 新 H1 記述子 ML 比較
  h1_inverse_analysis.png       : H1 逆解析（境界原子可視化）
"""

import os, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from multiprocessing import Pool, cpu_count
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
import homcloud.interface as hc
warnings.filterwarnings('ignore')

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE = str(REPO_ROOT / 'cache')
TXT   = str(REPO_ROOT / 'cache' / 'homcloud_txt')
OUT_DIR = REPO_ROOT / 'results'
DATA_DIR = REPO_ROOT / 'data'

plt.rcParams.update({
    'font.family': ['DejaVu Sans','sans-serif'],
    'axes.spines.top': False,'axes.spines.right': False,
    'axes.grid': True,'grid.alpha':0.3,'grid.linestyle':'--',
})
BLUE='#2E75B6'; GREEN='#378644'; ORANGE='#C05020'
GOLD='#FFC000'; PURPLE='#6B0AAC'; TEAL='#00707F'
LBLUE='#BDD7EE'; LGRAY='#F2F2F2'

# ─────────────────────────────────────────────────────────────
# 1. アルファ複体 H1 計算（並列）
# ─────────────────────────────────────────────────────────────
GRID_RANGE = (0, 10)   # Å²: H1 は H2 より小さい範囲に集中
GRID_N     = 64
SIGMA      = 0.002
WEIGHT     = ("atan", 0.01, 1.0)   # persistence < 1 はほぼ 0

def compute_h1_for_sample(idx):
    fpath = f'{TXT}/{idx:03d}.txt'
    pts   = np.loadtxt(fpath)
    pd_   = hc.PDList.from_alpha_filtration(pts, save_to=None, save_boundary_map=False)
    d1    = pd_.dth_diagram(1)
    pairs = d1.pairs()

    # Targeted features
    births  = np.array([p.birth for p in pairs])
    deaths  = np.array([p.death for p in pairs])
    persist = deaths - births
    fin     = persist > 0
    if fin.sum() == 0:
        return idx, np.zeros(6), []

    bi, de, pe = births[fin], deaths[fin], persist[fin]
    top_idx = np.argmax(pe)
    n_gt1   = (pe > 1.0).sum()
    n_gt2   = (pe > 2.0).sum()

    feats = [bi[top_idx], de[top_idx], pe[top_idx],
             n_gt1, n_gt2, pe.mean()]

    raw_pairs = list(zip(bi.tolist(), de.tolist()))
    return idx, feats, raw_pairs

def vectorize_h1_diagram(raw_pairs):
    """birth-death ペアリストを PI ベクトルに変換"""
    from homcloud.interface import PIVectorizerMesh
    vz = PIVectorizerMesh(GRID_RANGE, GRID_N, sigma=SIGMA, weight=WEIGHT)
    # homcloudのvectorizerはpairs形式で入力
    # birth,death配列を作成
    births = np.array([p[0] for p in raw_pairs])
    deaths = np.array([p[1] for p in raw_pairs])
    # 上三角インデックスのグリッドを直接手動で計算
    g_min, g_max = GRID_RANGE
    step = (g_max - g_min) / GRID_N
    centers = np.arange(g_min + step/2, g_max, step)
    # 64×64 グリッド（上三角部分）
    vec = np.zeros((GRID_N, GRID_N))
    for b, d in zip(births, deaths):
        p = d - b
        # weight
        if WEIGHT and WEIGHT[0] == "atan":
            w = np.arctan(WEIGHT[1] * max(p - WEIGHT[2], 0))
        else:
            w = 1.0
        if w < 1e-10:
            continue
        bi_idx = int((b - g_min) / step)
        de_idx = int((d - g_min) / step)
        bi_idx = max(0, min(GRID_N-1, bi_idx))
        de_idx = max(0, min(GRID_N-1, de_idx))
        # Gaussian smearing
        for gi in range(max(0, bi_idx-3), min(GRID_N, bi_idx+4)):
            for gj in range(max(0, de_idx-3), min(GRID_N, de_idx+4)):
                if gj <= gi:  # upper triangle (death >= birth)
                    continue
                dist2 = ((centers[gi]-b)**2 + (centers[gj]-d)**2)/(2*SIGMA**2 + 1e-10)
                if dist2 < 20:
                    vec[gi, gj] += w * np.exp(-dist2)
    # upper triangle → 1D
    idx_u = np.triu_indices(GRID_N, k=1)
    return vec[idx_u]

# ─────────────────────────────────────────────────────────────
# Cache check
# ─────────────────────────────────────────────────────────────
h1_vec_path   = f'{CACHE}/h1_vectors.npy'
h1_feat_path  = f'{CACHE}/h1_features.npy'
h1_pairs_path = f'{CACHE}/h1_raw_pairs.npy'

if os.path.exists(h1_vec_path) and os.path.exists(h1_feat_path):
    print("Loading H1 cache...")
    h1_vectors  = np.load(h1_vec_path)
    h1_features = np.load(h1_feat_path)
else:
    print(f"Computing H1 for 500 samples (6 cores)...")
    t0 = time.time()
    N_CORES = min(6, cpu_count())
    with Pool(N_CORES) as pool:
        results = pool.map(compute_h1_for_sample, range(500))
    t1 = time.time()
    print(f"  Alpha complex: {t1-t0:.1f}s = {(t1-t0)/60:.1f} min")

    results.sort(key=lambda x: x[0])
    h1_features = np.array([r[1] for r in results])   # 500×6
    raw_pairs_all = [r[2] for r in results]

    print("Vectorizing H1 diagrams...")
    t2 = time.time()
    h1_vectors = np.array([vectorize_h1_diagram(p) for p in raw_pairs_all])
    t3 = time.time()
    print(f"  Vectorization: {t3-t2:.1f}s")
    print(f"  H1 vector shape: {h1_vectors.shape}")

    np.save(h1_vec_path,  h1_vectors)
    np.save(h1_feat_path, h1_features)
    print(f"Cached: {h1_vec_path}, {h1_feat_path}")

print(f"H1 vectors: {h1_vectors.shape}   H1 features: {h1_features.shape}")

# ─────────────────────────────────────────────────────────────
# 2. データ読み込み
# ─────────────────────────────────────────────────────────────
dc   = pd.read_csv(DATA_DIR / 'D_cumulative.csv').sort_values('p_idx').reset_index(drop=True)
logD = np.log1p(dc['D_cumul'].values)
phys = dc[['il_type','temp','comp_idx','efield','snap_ns']].copy()
phys['il_bin'] = (phys['il_type']=='TFSI').astype(int)
phys_arr = phys[['il_bin','temp','comp_idx','efield','snap_ns']].values.astype(float)
groups   = dc['sim_id'].values

rdf_mat  = np.load(f'{CACHE}/rdf_matrix.npy')
zdm      = np.load(f'{CACHE}/zdensity_matrix.npy')
pd2_vec  = np.load(f'{CACHE}/pd_vectors.npy')   # H2 vectors for comparison

# H1 features 列名
H1_FEAT_NAMES = ['H1_birth','H1_death','H1_persistence','H1_n_gt1','H1_n_gt2','H1_mean_persist']

# ─────────────────────────────────────────────────────────────
# 3. 相関解析
# ─────────────────────────────────────────────────────────────
from scipy.stats import pearsonr

# H1 ビン vs logD
corr_h1 = np.array([pearsonr(h1_vectors[:,j], logD)[0]
                    for j in range(h1_vectors.shape[1])])
corr_h2 = np.array([pearsonr(pd2_vec[:,j], logD)[0]
                    for j in range(pd2_vec.shape[1])])

# H1 targeted features vs logD
feat_corr = {n: pearsonr(h1_features[:,i], logD)[0]
             for i, n in enumerate(H1_FEAT_NAMES)}
print("\n--- H1 targeted features vs logD (Pearson r) ---")
for k, v in feat_corr.items():
    print(f"  {k}: {v:+.3f}")

# ─────────────────────────────────────────────────────────────
# 4. ML パイプライン
# ─────────────────────────────────────────────────────────────
def make_X(descs, phys_arr):
    parts = []
    for d in descs:
        sel = VarianceThreshold(1e-10).fit_transform(d)
        scl = StandardScaler().fit_transform(sel)
        pca = PCA(n_components=min(20, scl.shape[1])).fit_transform(scl)
        parts.append(pca)
    parts.append(StandardScaler().fit_transform(phys_arr))
    return np.hstack(parts)

def run_ml(X, logD, groups, test_size=0.3):
    unique_g = np.unique(groups)
    n_test   = int(len(unique_g) * test_size)
    np.random.seed(42)
    test_g   = np.random.choice(unique_g, n_test, replace=False)
    tr       = ~np.isin(groups, test_g)
    te       = np.isin(groups, test_g)

    est = ExtraTreesRegressor(n_estimators=500, max_depth=5,
                              max_features=0.5, min_samples_leaf=2,
                              n_jobs=-1, random_state=42)
    cv = GroupKFold(n_splits=5)
    cv_r2 = cross_val_score(est, X[tr], logD[tr], groups=groups[tr],
                            cv=cv, scoring='r2').mean()
    est.fit(X[tr], logD[tr])
    pred_te = est.predict(X[te])
    te_r2   = r2_score(logD[te], pred_te)
    pred_real  = np.expm1(pred_te)
    true_real  = np.expm1(logD[te])
    rmse_real  = np.sqrt(mean_squared_error(true_real, pred_real))
    return cv_r2, te_r2, rmse_real, est, pred_te, logD[te]

print("\n--- ML 比較 ---")
model_defs = [
    ('H1 vector + 物理',          [h1_vectors],                     phys_arr),
    ('H1 targeted + 物理',        [h1_features],                    phys_arr),
    ('H2 vector + 物理 (参照)',    [pd2_vec],                        phys_arr),
    ('H1 vector + H2 vector + 物理', [h1_vectors, pd2_vec],         phys_arr),
    ('RDF + z密度 + 物理 (最良)', [rdf_mat, zdm],                   phys_arr),
    ('H1 vector + RDF + z密度 + 物理', [h1_vectors, rdf_mat, zdm],  phys_arr),
    ('H1 targeted + RDF + z密度 + 物理', [h1_features, rdf_mat, zdm], phys_arr),
]

ml_results = []
best_pred, best_true = None, None
for name, descs, phys in model_defs:
    X   = make_X(descs, phys_arr)
    cv, te, rmse, est, pred, true = run_ml(X, logD, groups)
    ml_results.append({'name': name, 'CV_R2': cv, 'Test_R2': te, 'RMSE': rmse})
    if name == 'H1 vector + 物理':
        best_pred, best_true = pred, true
    print(f"  {name:40s}  CV R²={cv:.3f}  Test R²={te:.3f}  RMSE={rmse:.0f}")

df_ml = pd.DataFrame(ml_results)
df_ml.to_csv(OUT_DIR / 'h1_ml_results.csv', index=False)
print(f"Saved: h1_ml_results.csv")

# ─────────────────────────────────────────────────────────────
# 5. 可視化 1: 相関マップ + H1 diagram 条件別
# ─────────────────────────────────────────────────────────────
print("\nGenerating figures...")

# --- h1_corr_map.png ---
N_TRI = h1_vectors.shape[1]
# upper triangle indices (GRID_N=64)
idx_u = np.triu_indices(GRID_N, k=1)
corr_map = np.full((GRID_N, GRID_N), np.nan)
for k, (i, j) in enumerate(zip(*idx_u)):
    if k < len(corr_h1):
        corr_map[i, j] = corr_h1[k]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor(LGRAY)
fig.suptitle('H1 パーシステンス図ベクトル vs log1p(D_cumul) 相関解析', fontsize=14, fontweight='bold', color='#1F497D')

ax = axes[0]; ax.set_facecolor('white')
vmax = np.nanmax(np.abs(corr_h1))
step = (GRID_RANGE[1]-GRID_RANGE[0])/GRID_N
ext  = [GRID_RANGE[0], GRID_RANGE[1], GRID_RANGE[0], GRID_RANGE[1]]
im   = ax.imshow(corr_map, origin='lower', extent=ext,
                 cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
plt.colorbar(im, ax=ax, label='Pearson r')
ax.set_xlabel('birth α (Å²)', fontsize=11); ax.set_ylabel('death α (Å²)', fontsize=11)
ax.set_title('H1: birth-death グリッド vs D 相関マップ', fontsize=12, fontweight='bold')
ax.plot([GRID_RANGE[0],GRID_RANGE[1]],[GRID_RANGE[0],GRID_RANGE[1]], 'k--', lw=0.8, alpha=0.5)
max_k = np.nanargmax(np.abs(corr_h1))
bi_i, de_j = idx_u[0][max_k], idx_u[1][max_k]
bi_c = GRID_RANGE[0] + (bi_i+0.5)*step
de_c = GRID_RANGE[0] + (de_j+0.5)*step
ax.plot(bi_c, de_c, 'k*', ms=12, label=f'max |r|={abs(corr_h1[max_k]):.3f}')
ax.legend(fontsize=9)

# H1 vs H2 distribution comparison
ax = axes[1]; ax.set_facecolor('white')
ax.hist(np.abs(corr_h1[np.isfinite(corr_h1)]), bins=40, alpha=0.7, color=TEAL, label=f'H1 (n={len(corr_h1)})', density=True)
ax.hist(np.abs(corr_h2[np.isfinite(corr_h2)]), bins=40, alpha=0.7, color=ORANGE, label=f'H2 (n={len(corr_h2)})', density=True)
ax.set_xlabel('|Pearson r| vs log1p(D)', fontsize=11)
ax.set_ylabel('密度', fontsize=11)
ax.set_title('H1 vs H2: ビン相関係数の分布比較', fontsize=12, fontweight='bold')
ax.axvline(np.nanmax(np.abs(corr_h1)), color=TEAL, ls='--', lw=2, label=f'H1 max: {np.nanmax(np.abs(corr_h1)):.3f}')
ax.axvline(np.nanmax(np.abs(corr_h2)), color=ORANGE, ls='--', lw=2, label=f'H2 max: {np.nanmax(np.abs(corr_h2)):.3f}')
ax.legend(fontsize=9)

fig.tight_layout()
fig.savefig(OUT_DIR / 'h1_corr_map.png', dpi=150, bbox_inches='tight')
plt.close(); print("  h1_corr_map.png")

# --- h1_pd_diagrams.png ---
# 条件別に H1 persistence 図を可視化
fig2, axes2 = plt.subplots(2, 4, figsize=(18, 9))
fig2.patch.set_facecolor(LGRAY)
fig2.suptitle('H1 パーシステンス図 条件別（温度・IL種・電場・D高低）', fontsize=14, fontweight='bold', color='#1F497D')

# 再計算（軽量版: 各条件の代表サンプルのみ）
cond_samples = [
    (dc[dc['temp']==250].index[0], '250 K', LBLUE),
    (dc[dc['temp']==300].index[0], '300 K', '#9DC3E6'),
    (dc[dc['temp']==350].index[0], '350 K', BLUE),
    (dc[dc['temp']==450].index[0], '450 K', '#1F497D'),
    (dc[dc['il_type']=='BF4'].index[0], 'BF4 (ef=0)', GREEN),
    (dc[(dc['il_type']=='BF4')&(dc['efield']==1)].index[0], 'BF4 (ef=1)', '#70AD47'),
    (dc[dc['D_cumul']<dc['D_cumul'].quantile(0.1)].index[0], '低D系 (<10%)', BLUE),
    (dc[dc['D_cumul']>dc['D_cumul'].quantile(0.9)].index[0], '高D系 (>90%)', ORANGE),
]

for ax, (idx, label, col) in zip(axes2.flat, cond_samples):
    ax.set_facecolor('white')
    fpath = f'{TXT}/{idx:03d}.txt'
    pts   = np.loadtxt(fpath)
    pd_   = hc.PDList.from_alpha_filtration(pts, save_to=None, save_boundary_map=False)
    d1    = pd_.dth_diagram(1)
    pairs = d1.pairs()
    b_all = np.array([p.birth for p in pairs])
    d_all = np.array([p.death for p in pairs])
    pe_all= d_all - b_all
    fin   = pe_all > 0

    sc = ax.scatter(b_all[fin], d_all[fin], c=pe_all[fin], cmap='viridis',
                    s=8, alpha=0.6, rasterized=True)
    plt.colorbar(sc, ax=ax, label='persistence')
    ax.plot([0,10],[0,10],'k--',lw=0.8,alpha=0.5)
    ax.axhline(b_all[fin].min()+1, color='gray', ls=':', lw=0.8)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.set_xlabel('birth α (Å²)', fontsize=9); ax.set_ylabel('death α (Å²)', fontsize=9)
    d_val = dc.loc[idx,'D_cumul']
    ax.set_title(f'{label}\nD={d_val:.1f} Å²/ns, n_pairs={fin.sum()}', fontsize=10, fontweight='bold', color=col)

    # max persistence 点を強調
    top = np.argmax(pe_all[fin])
    ax.scatter(b_all[fin][top], d_all[fin][top], marker='*', s=150, color='red', zorder=5,
               label=f'max p={pe_all[fin][top]:.2f}')
    ax.legend(fontsize=7)

fig2.tight_layout()
fig2.savefig(OUT_DIR / 'h1_pd_diagrams.png', dpi=150, bbox_inches='tight')
plt.close(); print("  h1_pd_diagrams.png")

# --- h1_pd_comparison.png (250K vs 450K) ---
fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5))
fig3.patch.set_facecolor(LGRAY)
fig3.suptitle('H1 パーシステンス図: 250K (低D) vs 450K (高D) — BF4_gra No.01 100ns',
              fontsize=13, fontweight='bold', color='#1F497D')

for ax, (fname, label, col) in zip(axes3, [
    ('BF4_No01_250K_100ns.idiagram', '250 K — 低D系 (D≈8.9 Å²/ns)', BLUE),
    ('BF4_No01_450K_100ns.idiagram', '450 K — 高D系 (D≈数千 Å²/ns)', ORANGE),
]):
    ax.set_facecolor('white')
    pd_ = hc.PDList(f'{CACHE}/homcloud/{fname}')
    d1  = pd_.dth_diagram(1)
    pairs = d1.pairs()
    b   = np.array([p.birth for p in pairs])
    d   = np.array([p.death for p in pairs])
    pe  = d - b; fin = pe > 0

    sc  = ax.scatter(b[fin], d[fin], c=pe[fin], cmap='plasma',
                     s=15, alpha=0.7, rasterized=True)
    plt.colorbar(sc, ax=ax, label='persistence (Å²)')
    ax.plot([0,14],[0,14],'k--',lw=1,alpha=0.5)
    ax.axvline(1.0, color='gray', ls=':', lw=0.8, alpha=0.7, label='birth=1.0')
    ax.set_xlim(0,14); ax.set_ylim(0,14)
    ax.set_xlabel('birth α (Å²)', fontsize=11); ax.set_ylabel('death α (Å²)', fontsize=11)

    top = np.argmax(pe[fin])
    top_b, top_d, top_p = b[fin][top], d[fin][top], pe[fin][top]
    ax.scatter(top_b, top_d, marker='★' if False else '*', s=300, color='red', zorder=6,
               label=f'max: birth={top_b:.2f}, p={top_p:.2f}')

    gt1 = (pe[fin] > 1.0).sum(); gt2 = (pe[fin] > 2.0).sum()
    ax.set_title(f'{label}\n全 {fin.sum()} ペア  |  persist>1: {gt1}  |  persist>2: {gt2}',
                 fontsize=11, fontweight='bold', color=col)
    ax.legend(fontsize=9)

fig3.tight_layout()
fig3.savefig(OUT_DIR / 'h1_pd_comparison.png', dpi=150, bbox_inches='tight')
plt.close(); print("  h1_pd_comparison.png")

# ─────────────────────────────────────────────────────────────
# 6. 可視化 2: ML 結果
# ─────────────────────────────────────────────────────────────
fig4, axes4 = plt.subplots(1, 3, figsize=(18, 5))
fig4.patch.set_facecolor(LGRAY)
fig4.suptitle('H1 記述子 vs H2 記述子 機械学習比較 (ExtraTrees, GroupKFold-5)',
              fontsize=14, fontweight='bold', color='#1F497D')

names  = [r['name'] for r in ml_results]
cvr2s  = [r['CV_R2'] for r in ml_results]
ter2s  = [r['Test_R2'] for r in ml_results]
rmses  = [r['RMSE'] for r in ml_results]
colors = [TEAL if 'H1 vector' in n and '物理' in n and 'RDF' not in n and 'H2' not in n else
          '#70AD47' if 'H1 targeted' in n and 'RDF' not in n else
          ORANGE if 'H2' in n else
          PURPLE if 'H1 vector + H2' in n else
          GREEN for n in names]

x = np.arange(len(names))
ax = axes4[0]; ax.set_facecolor('white')
w = 0.38
b1 = ax.bar(x-w/2, cvr2s, w, color=colors, alpha=0.85, label='CV R²')
b2 = ax.bar(x+w/2, ter2s, w, color=colors, alpha=0.5,  label='Test R²', edgecolor='gray')
ax.set_xticks(x); ax.set_xticklabels([n.replace(' + ','\\n+') for n in names], fontsize=7, rotation=15, ha='right')
ax.set_ylim(0, 0.85); ax.set_ylabel('R²', fontsize=11)
ax.set_title('CV R² vs Test R²', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.axhline(0.727, color=GREEN, ls=':', lw=1.5, label='H2+RDF最良 (0.727)')
for i, (v1, v2) in enumerate(zip(cvr2s, ter2s)):
    ax.text(i-w/2, v1+0.008, f'{v1:.3f}', ha='center', fontsize=7, color='#1F497D')

ax = axes4[1]; ax.set_facecolor('white')
ax.bar(x, rmses, color=colors, alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels([n.replace(' + ','\\n+') for n in names], fontsize=7, rotation=15, ha='right')
ax.set_ylabel('RMSE (Å²/ns)', fontsize=11)
ax.set_title('RMSE（実スケール）', fontsize=12, fontweight='bold')
ax.axhline(334, color=GREEN, ls=':', lw=1.5, label='H2+RDF最良 (334)')
ax.legend(fontsize=9)
for bar, v in zip(ax.patches, rmses):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+8,
            f'{int(v)}', ha='center', fontsize=8)

# Scatter: H1 pred vs true (H1 vector + 物理)
ax = axes4[2]; ax.set_facecolor('white')
ax.scatter(best_true, best_pred, alpha=0.5, s=20, color=TEAL, label='H1 vector + 物理')
lims = [min(best_true.min(), best_pred.min())-0.2, max(best_true.max(), best_pred.max())+0.2]
ax.plot(lims, lims, 'k--', lw=1)
ax.set_xlabel('実測 log1p(D_cumul)', fontsize=11); ax.set_ylabel('予測 log1p(D_cumul)', fontsize=11)
h1_te_r2 = ml_results[0]['Test_R2']
ax.set_title(f'H1 vector + 物理: 予測 vs 実測\nTest R² = {h1_te_r2:.3f}', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)

fig4.tight_layout()
fig4.savefig(OUT_DIR / 'h1_targeted_ml.png', dpi=150, bbox_inches='tight')
plt.close(); print("  h1_targeted_ml.png")

# ─────────────────────────────────────────────────────────────
# 7. H1 逆解析（既存 idiagram ファイルを使用）
# ─────────────────────────────────────────────────────────────
print("\nH1 inverse analysis (using cached idiagram with boundary map)...")

def load_lammps_snapshot(data_path):
    """LAMMPS data ファイルから原子座標と原子種を読み込む"""
    import re
    coords, types = [], []
    in_atoms = False
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('Atoms'):
                in_atoms = True; next(f); continue
            if in_atoms:
                if line == '' and len(coords) > 0:
                    break
                parts = line.split()
                if len(parts) >= 8:
                    try:
                        atype = int(parts[2])
                        x, y, z = float(parts[4]), float(parts[5]), float(parts[6])
                        coords.append([x, y, z]); types.append(atype)
                    except: pass
    return np.array(coords), np.array(types)

type_map = {1:'Na+', 2:'N(EMI)', 3:'C(EMI)', 4:'N(EMI)', 5:'C(EMI)',
            6:'C(EMI)', 7:'H(EMI)', 8:'C(EMI)', 9:'H(EMI)', 10:'C(EMI)',
            11:'H(EMI)', 12:'H(EMI)', 13:'H(EMI)', 14:'B(BF4)', 15:'F(BF4)', 16:'C(gra)'}

# Raw LAMMPS data file tree on the Windows host (only needed for inverse analysis).
RAW_WIN_BASE = r'D:\path\to\raw\trajectories'
conditions = [
    ('BF4_No01_250K_100ns', '250 K (低D: D≈8.9 Å²/ns)', BLUE,
     fr'{RAW_WIN_BASE}\BF4_gra\Na_EMI_BF4_gra_250K\No.01_250K\data.eq4.100000000.data'),
    ('BF4_No01_450K_100ns', '450 K (高D: D≈数千 Å²/ns)', ORANGE,
     fr'{RAW_WIN_BASE}\BF4_gra\Na_EMI_BF4_gra_450K\No.01_450K\data.eq4.100000000.data'),
]

import subprocess

fig5, axes5 = plt.subplots(2, 3, figsize=(18, 10))
fig5.patch.set_facecolor(LGRAY)
fig5.suptitle('H1 逆解析: 最大永続性 H1 ループの境界原子同定 (BF4_gra No.01 / 100ns)',
              fontsize=14, fontweight='bold', color='#1F497D')

type_colors = {'Na+':'red','N(EMI)':'blue','C(EMI)':'cyan',
               'H(EMI)':'lightblue','B(BF4)':'yellow','F(BF4)':'green','C(gra)':'black'}

for row, (base, label, col, win_path) in enumerate(conditions):
    idiag = f'{CACHE}/homcloud/{base}.idiagram'
    # LAMMPSデータ読み込み（PowerShell経由）
    wsl_path = None  # raw data is read via PowerShell on the Windows side
    # PowerShell で読む
    try:
        result = subprocess.run(
            ['powershell.exe', '-Command', f"Get-Content '{win_path}'"],
            capture_output=True, text=True, timeout=30)
        lines = result.stdout.strip().split('\n')
        coords, atypes = [], []
        in_atoms = False
        for line in lines:
            line = line.strip()
            if line.startswith('Atoms'): in_atoms = True; continue
            if in_atoms and line == '' and len(coords) > 0: break
            if in_atoms:
                parts = line.split()
                if len(parts) >= 8:
                    try:
                        at = int(parts[2])
                        x, y, z = float(parts[4]), float(parts[5]), float(parts[6])
                        coords.append([x,y,z]); atypes.append(at)
                    except: pass
        coords = np.array(coords); atypes = np.array(atypes)
        has_coords = len(coords) > 0
    except Exception as e:
        print(f"  Warning: could not load coords ({e})")
        has_coords = False

    # H1 inverse analysis
    pd_  = hc.PDList(idiag)
    d1   = pd_.dth_diagram(1)
    pairs = d1.pairs()
    b    = np.array([p.birth for p in pairs])
    de   = np.array([p.death for p in pairs])
    pe   = de - b
    fin  = pe > 0
    top_idx = np.argmax(pe[fin])
    top_pair = [p for p in pairs if (p.death - p.birth) == pe[fin][top_idx]][0]

    print(f"\n[{label}] H1 max persistence pair:")
    print(f"  birth={b[fin][top_idx]:.3f} Å²  death={de[fin][top_idx]:.3f} Å²  "
          f"persistence={pe[fin][top_idx]:.3f} Å²")
    print(f"  persistence > 1.0: {(pe[fin]>1.0).sum()} pairs")

    try:
        bdy_atoms = top_pair.boundary_points()
        bdy_idx   = np.array([a.index for a in bdy_atoms])
        print(f"  Boundary atoms: {len(bdy_idx)}")
        has_boundary = has_coords and len(bdy_idx) > 0

        if has_boundary and len(coords) > 0:
            bdy_coords = coords[bdy_idx]
            bdy_types  = [type_map.get(atypes[i], f'type{atypes[i]}') for i in bdy_idx]
            type_count = {}
            for t in bdy_types:
                type_count[t] = type_count.get(t, 0) + 1
            print(f"  Type counts: {type_count}")
        else:
            has_boundary = False
    except Exception as e:
        print(f"  Boundary atoms not available: {e}")
        has_boundary = False

    # Plot 1: H1 persistence diagram
    ax = axes5[row, 0]
    ax.set_facecolor('white')
    sc = ax.scatter(b[fin], de[fin], c=pe[fin], cmap='plasma', s=10, alpha=0.7)
    plt.colorbar(sc, ax=ax, label='persistence (Å²)')
    ax.plot([0,14],[0,14],'k--',lw=1,alpha=0.5)
    ax.scatter(b[fin][top_idx], de[fin][top_idx], marker='*', s=300, color='red',
               zorder=6, label=f'最大 p={pe[fin][top_idx]:.2f}')
    ax.set_xlim(0,14); ax.set_ylim(0,14)
    ax.set_xlabel('birth α (Å²)', fontsize=10); ax.set_ylabel('death α (Å²)', fontsize=10)
    ax.set_title(f'{label}\nH1 パーシステンス図', fontsize=10, fontweight='bold', color=col)
    ax.legend(fontsize=8)

    # Plot 2: 3D visualization of boundary atoms
    ax = axes5[row, 1]
    ax.set_facecolor('white')
    if has_boundary:
        # XY projection
        all_colors = [type_colors.get(type_map.get(t, 'other'), 'gray') for t in atypes]
        ax.scatter(coords[:,0], coords[:,1], c='lightgray', s=1, alpha=0.15, rasterized=True)
        for tname, tcolor in type_colors.items():
            mask_all = np.array([type_map.get(at, '') == tname for at in atypes])
            bdy_mask = np.array([t == tname for t in bdy_types])
            if bdy_mask.sum() > 0:
                ax.scatter(bdy_coords[bdy_mask,0], bdy_coords[bdy_mask,1],
                           c=tcolor, s=50, alpha=0.9, label=f'{tname}({bdy_mask.sum()})', zorder=5)
        ax.set_xlabel('x (Å)', fontsize=10); ax.set_ylabel('y (Å)', fontsize=10)
        ax.set_title(f'H1 境界原子 XY 投影\n全{len(bdy_idx)}個', fontsize=10, fontweight='bold', color=col)
        ax.legend(fontsize=7, ncol=2)
    else:
        ax.text(0.5, 0.5, '座標データ\n読み込み不可\n(D:ドライブ未接続)', ha='center', va='center',
                transform=ax.transAxes, fontsize=12, color='gray')
        ax.set_title(f'H1 境界原子 XY 投影', fontsize=10, fontweight='bold', color=col)

    # Plot 3: Bar chart of atom type composition
    ax = axes5[row, 2]
    ax.set_facecolor('white')
    if has_boundary:
        tc_sorted = sorted(type_count.items(), key=lambda x: x[1], reverse=True)
        names_tc = [t for t, _ in tc_sorted]
        counts_tc= [c for _, c in tc_sorted]
        pcts_tc  = [c/len(bdy_idx)*100 for c in counts_tc]
        bcolors  = [type_colors.get(n, 'gray') for n in names_tc]
        bars = ax.bar(range(len(names_tc)), pcts_tc, color=bcolors, alpha=0.85)
        ax.set_xticks(range(len(names_tc))); ax.set_xticklabels(names_tc, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel('境界原子の割合 (%)', fontsize=10)
        ax.set_title(f'H1 境界原子の組成\n(n={len(bdy_idx)}, 1D ループ形成原子)', fontsize=10, fontweight='bold', color=col)
        for bar, pct, cnt in zip(bars, pcts_tc, counts_tc):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                    f'{cnt}\n({pct:.0f}%)', ha='center', fontsize=8)
    else:
        ax.text(0.5, 0.5, '境界情報なし\n(D:ドライブ必要)', ha='center', va='center',
                transform=ax.transAxes, fontsize=12, color='gray')
        ax.set_title('H1 境界原子の組成', fontsize=10, fontweight='bold', color=col)

fig5.tight_layout()
fig5.savefig(OUT_DIR / 'h1_inverse_analysis.png', dpi=120, bbox_inches='tight')
plt.close(); print("  h1_inverse_analysis.png")

# ─────────────────────────────────────────────────────────────
# 8. H1 vs H2 総合比較図
# ─────────────────────────────────────────────────────────────
fig6, axes6 = plt.subplots(2, 3, figsize=(18, 10))
fig6.patch.set_facecolor(LGRAY)
fig6.suptitle('H1（1次: ループ）vs H2（2次: 空洞）— 総合比較', fontsize=14, fontweight='bold', color='#1F497D')

# (0,0) 特徴量 vs D 相関棒グラフ
ax = axes6[0,0]; ax.set_facecolor('white')
feat_names_all = H1_FEAT_NAMES + ['H2_birth','H2_death','H2_persist','H2_n_gt3','H2_Na_prox','H2_Na_edge']
# H2 targeted features from ph_targeted_ml cache
try:
    h2_feat = np.load(f'{CACHE}/ph_targeted_features.npy')
    feat_corr_h2 = {f'H2_feat{i}': pearsonr(h2_feat[:,i], logD)[0] for i in range(h2_feat.shape[1])}
except:
    feat_corr_h2 = {}
feat_names_plot = list(feat_corr.keys()) + list(feat_corr_h2.keys())[:4]
feat_vals_plot  = list(feat_corr.values()) + list(feat_corr_h2.values())[:4]
colors_f = [TEAL]*len(feat_corr) + [ORANGE]*4
ax.bar(range(len(feat_names_plot)), feat_vals_plot, color=colors_f, alpha=0.85)
ax.set_xticks(range(len(feat_names_plot)))
ax.set_xticklabels([n.replace('H1_','').replace('H2_','') for n in feat_names_plot],
                   rotation=30, ha='right', fontsize=9)
ax.set_ylabel('Pearson r vs log1p(D)', fontsize=10)
ax.set_title('Targeted 特徴量 vs D 相関\n青=H1, 橙=H2 targeted', fontsize=11, fontweight='bold')
ax.axhline(0, color='black', lw=0.8)

# (0,1) H1 persistence 統計 vs 温度
ax = axes6[0,1]; ax.set_facecolor('white')
h1_max_p = h1_features[:,2]  # max persistence
for il, col in [('BF4', BLUE), ('TFSI', ORANGE)]:
    mask = dc['il_type'] == il
    for t in sorted(dc['temp'].unique()):
        tm = mask & (dc['temp']==t)
        ax.scatter([t]*tm.sum(), h1_max_p[tm], color=col, alpha=0.3, s=10)
    # median line
    meds = [np.median(h1_max_p[mask & (dc['temp']==t)]) for t in sorted(dc['temp'].unique())]
    ax.plot(sorted(dc['temp'].unique()), meds, color=col, lw=2.5, marker='o', ms=7, label=il)
ax.set_xlabel('温度 (K)', fontsize=10); ax.set_ylabel('H1 最大 persistence (Å²)', fontsize=10)
ax.set_title('H1 最大永続性 vs 温度\n(H2と同様の温度依存性か?)', fontsize=11, fontweight='bold')
ax.legend(fontsize=9)

# (0,2) H1 max persistence vs H2 max persistence scatter
ax = axes6[0,2]; ax.set_facecolor('white')
try:
    h2_feat_data = np.load(f'{CACHE}/ph_targeted_features.npy')
    h2_max_p = h2_feat_data[:,2] if h2_feat_data.shape[1] > 2 else None
except:
    h2_max_p = None

if h2_max_p is not None:
    sc = ax.scatter(h2_max_p, h1_max_p, c=logD, cmap='viridis', s=15, alpha=0.6)
    plt.colorbar(sc, ax=ax, label='log1p(D)')
    ax.set_xlabel('H2 最大 persistence (Å²)', fontsize=10)
    ax.set_ylabel('H1 最大 persistence (Å²)', fontsize=10)
    r_val = pearsonr(h2_max_p, h1_max_p)[0]
    ax.set_title(f'H1 vs H2 最大 persistence\nPearson r={r_val:.3f}', fontsize=11, fontweight='bold')
else:
    ax.text(0.5, 0.5, 'H2 データ不可', ha='center', va='center', transform=ax.transAxes)

# (1,0) ML 比較棒グラフ
ax = axes6[1,0]; ax.set_facecolor('white')
ml_names_short = ['H1 vec\n+物理', 'H1 tgt\n+物理', 'H2 vec\n+物理(参照)',
                  'H1+H2\n+物理', 'RDF+z\n+物理(最良)', 'H1+RDF\n+物理', 'H1tgt+RDF\n+物理']
x_ml = np.arange(len(ml_results))
ax.bar(x_ml-0.2, [r['CV_R2'] for r in ml_results], 0.38, color=TEAL, alpha=0.85, label='CV R²')
ax.bar(x_ml+0.2, [r['Test_R2'] for r in ml_results], 0.38, color=ORANGE, alpha=0.7, label='Test R²')
ax.set_xticks(x_ml); ax.set_xticklabels(ml_names_short, fontsize=8)
ax.axhline(0.727, color=GREEN, ls=':', lw=1.5, label='H2+RDF最良(0.727)')
ax.set_ylabel('R²', fontsize=10); ax.set_title('ML モデル比較', fontsize=11, fontweight='bold')
ax.legend(fontsize=8); ax.set_ylim(0, 0.85)

# (1,1) H1 vs H2 ペア数 vs D
ax = axes6[1,1]; ax.set_facecolor('white')
h1_n_gt1 = h1_features[:,3]
ax.scatter(h1_n_gt1, logD, c=dc['temp'], cmap='coolwarm', s=15, alpha=0.5, label='H1 n_gt1')
ax.set_xlabel('H1 n_pairs (persist>1)', fontsize=10)
ax.set_ylabel('log1p(D_cumul)', fontsize=10)
r_n = pearsonr(h1_n_gt1, logD)[0]
ax.set_title(f'H1 有意ペア数 vs D\nPearson r={r_n:.3f}', fontsize=11, fontweight='bold')
sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(250,450))
plt.colorbar(sm, ax=ax, label='温度 (K)')

# (1,2) RMSE比較
ax = axes6[1,2]; ax.set_facecolor('white')
rmse_vals = [r['RMSE'] for r in ml_results]
bar_colors_r = [GREEN if v==min(rmse_vals) else TEAL if 'H1' in ml_results[i]['name'] else ORANGE
                for i, v in enumerate(rmse_vals)]
bars = ax.bar(x_ml, rmse_vals, color=bar_colors_r, alpha=0.85)
ax.set_xticks(x_ml); ax.set_xticklabels(ml_names_short, fontsize=8)
ax.axhline(334, color=GREEN, ls=':', lw=1.5, label='H2+RDF最良(334)')
ax.set_ylabel('RMSE (Å²/ns)', fontsize=10)
ax.set_title('RMSE 比較', fontsize=11, fontweight='bold')
ax.legend(fontsize=8)
for bar, v in zip(bars, rmse_vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+8,
            f'{int(v)}', ha='center', fontsize=8)

fig6.tight_layout()
fig6.savefig(OUT_DIR / 'h1_h2_comparison.png', dpi=150, bbox_inches='tight')
plt.close(); print("  h1_h2_comparison.png")

print("\n=== 完了 ===")
print(f"出力ファイル:")
for f in ['h1_ml_results.csv','h1_corr_map.png','h1_pd_diagrams.png',
          'h1_pd_comparison.png','h1_targeted_ml.png','h1_inverse_analysis.png','h1_h2_comparison.png']:
    path = str(OUT_DIR / f)
    size = os.path.getsize(path)//1024 if os.path.exists(path) else 0
    print(f"  {f}: {size} KB")
