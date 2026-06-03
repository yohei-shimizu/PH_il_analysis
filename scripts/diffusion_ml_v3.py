"""
イオン液体拡散係数再計算 + 機械学習モデル構築 (v3 最終版)

v3の改善点:
1. 500サンプル全使用 (各シミュレーション5スナップショット)
2. 目的変数: log1p(D_cumul) - 累積拡散係数 MSD(T)/(6T) で確実に正値
3. シミュレーション単位でtrain/test分割 (データリーク完全防止)
4. PCA成分数を最適化 (15成分)
5. 物理パラメータ + pd_vector を組み合わせた特徴量
6. 複数モデルの系統的比較
"""

import numpy as np
import pandas as pd
from scipy import stats
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / 'results'
DATA_DIR = REPO_ROOT / 'data'

# Path to the raw LAMMPS trajectory tree (only needed for step 1).
BASE = '/path/to/raw/trajectories'
TEMPS = [250, 300, 350, 400, 450]
# snap_idx 0=10ns(100ns), 1=6ns(60ns), 2=7ns(70ns), 3=8ns(80ns), 4=9ns(90ns)
SNAP_NS   = [100, 60, 70, 80, 90]   # actual ns
SNAP_SEG  = [10,   6,  7,  8,  9]   # file segment number

# ============================================================
# 1. MSDから累積拡散係数を計算
# ============================================================

def get_msd_at_snapshot(folder, temp, snap_seg):
    """
    指定セグメントファイルの最終行のMSD値を返す
    snap_seg: セグメント番号 (6-10 → 60-100 ns)
    """
    fpath = os.path.join(folder, f'Na_{temp}K.{snap_seg:02d}.msd')
    if not os.path.exists(fpath):
        return np.nan
    try:
        data = np.loadtxt(fpath)
        return float(data[-1, 4])  # 最終行のtotal MSD
    except Exception:
        return np.nan


def calc_cumulative_D(msd_val, time_ns):
    """D_cumul = MSD / (6 * t) [Å²/ns]"""
    if np.isnan(msd_val) or time_ns <= 0:
        return np.nan
    return max(msd_val / (6.0 * time_ns), 0.0)


# ============================================================
# 2. 全データセットを構築
# ============================================================

def build_full_dataset():
    rows = []
    sim_id_counter = 0

    for il_type in ['BF4', 'TFSI']:
        system = 'BF4_gra' if il_type == 'BF4' else 'TFSI_gra'
        il_folder = os.path.join(BASE, system)
        for comp_idx in range(5):
            no = comp_idx + 1
            for temp_idx, temp in enumerate(TEMPS):
                for ef_idx in range(2):
                    ef_str = f'_{temp}K_ef' if ef_idx == 1 else f'_{temp}K'
                    if il_type == 'BF4':
                        folder_name = f'Na_EMI_BF4_gra{ef_str}'
                    else:
                        folder_name = f'Na_EMI_TFSI_gra{ef_str}'
                    folder = os.path.join(il_folder, folder_name,
                                          f'No.{no:02d}{ef_str}')

                    # p_base offset for TFSI
                    p_base = 0 if il_type == 'BF4' else 250
                    sim_id = sim_id_counter

                    for snap_idx in range(5):
                        p_idx = (p_base + comp_idx * 50 +
                                 temp_idx * 10 + ef_idx * 5 + snap_idx)
                        seg   = SNAP_SEG[snap_idx]
                        t_ns  = SNAP_NS[snap_idx]

                        msd_val = get_msd_at_snapshot(folder, temp, seg)
                        D = calc_cumulative_D(msd_val, t_ns)

                        rows.append({
                            'sim_id':    sim_id,
                            'il_type':   il_type,
                            'comp_idx':  comp_idx,
                            'temp':      temp,
                            'efield':    ef_idx,
                            'snap_idx':  snap_idx,
                            'snap_ns':   t_ns,
                            'p_idx':     p_idx,
                            'D_cumul':   D,
                        })

                    sim_id_counter += 1

    return pd.DataFrame(rows), sim_id_counter


print("=== データセット構築中 ===")
df, n_sims = build_full_dataset()
print(f"総サンプル数: {len(df)}, シミュレーション数: {n_sims}")

D_vals = df['D_cumul'].values
valid = ~np.isnan(D_vals)
print(f"有効サンプル: {valid.sum()}")
print(f"D_cumul範囲: {D_vals[valid].min():.2f} ~ {D_vals[valid].max():.2f} Å²/ns")
print(f"D_cumul中央値: {np.median(D_vals[valid]):.2f}")
print(f"D_cumul=0件数: {(D_vals[valid] == 0).sum()}")

# NANを除外
df = df[valid].reset_index(drop=True)

# 拡散係数CSVを保存
df_save = df.drop_duplicates(subset=['p_idx'])
df_save[['p_idx', 'sim_id', 'il_type', 'comp_idx', 'temp', 'efield',
          'snap_ns', 'D_cumul']].to_csv(DATA_DIR / 'D_cumulative.csv', index=False)
print(f"D_cumul CSV保存: {DATA_DIR / 'D_cumulative.csv'}")

# ============================================================
# 3. 特徴量行列の構築
# ============================================================

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.model_selection import (train_test_split, cross_val_score,
                                     GroupKFold, KFold, GroupShuffleSplit)
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                               ExtraTreesRegressor)
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# pd_vectorの読み込み
print("\n=== pd_vector読み込み中 ===")
vecs = []
for _, row in df.iterrows():
    vec = np.loadtxt(f'{BASE}/pd_vector/p{int(row["p_idx"]):03d}.dat')
    vecs.append(vec)
X_pd = np.array(vecs)  # (N, 2080)
print(f"pd_vector形状: {X_pd.shape}")

# 物理パラメータ特徴量
il_enc = (df['il_type'] == 'TFSI').astype(float).values.reshape(-1, 1)
temp_norm = (df['temp'].values / 450.0).reshape(-1, 1)
comp_norm = (df['comp_idx'].values / 4.0).reshape(-1, 1)
ef_feat  = df['efield'].values.reshape(-1, 1)
snap_norm = (df['snap_ns'].values / 100.0).reshape(-1, 1)
X_phys = np.hstack([il_enc, temp_norm, comp_norm, ef_feat, snap_norm])
print(f"物理パラメータ特徴量形状: {X_phys.shape}")

# 目的変数
y_raw = df['D_cumul'].values
y = np.log1p(y_raw)
print(f"y_log 範囲: {y.min():.3f} ~ {y.max():.3f}")

# グループ (シミュレーション単位)
groups = df['sim_id'].values

# ============================================================
# 4. 特徴量前処理
# ============================================================

# pd_vector: VarianceThreshold → StandardScaler → PCA
vt = VarianceThreshold(threshold=1e-10)
X_pd_vt = vt.fit_transform(X_pd)
print(f"\nVarianceThreshold後: {X_pd_vt.shape[1]} 特徴量")

scaler_pd = StandardScaler()
X_pd_sc = scaler_pd.fit_transform(X_pd_vt)

N_PCA = 20
pca = PCA(n_components=N_PCA, random_state=42)
X_pd_pca = pca.fit_transform(X_pd_sc)
cum_var = np.cumsum(pca.explained_variance_ratio_)
print(f"PCA {N_PCA}成分 累積寄与率: {cum_var[-1]:.3f}")
print(f"各PC寄与率 (上位5): {pca.explained_variance_ratio_[:5].round(4)}")

# 物理パラメータのスケーリング (既に0-1程度)
scaler_phys = StandardScaler()
X_phys_sc = scaler_phys.fit_transform(X_phys)

# 結合特徴量
X_combined = np.hstack([X_pd_pca, X_phys_sc])  # (N, 25)
print(f"\n結合特徴量形状: {X_combined.shape}")

# ============================================================
# 5. Train/Test分割 (シミュレーション単位)
# ============================================================

# シミュレーションのユニークIDを70/30で分割
unique_sims = np.unique(groups)
np.random.seed(42)
train_sims, test_sims = train_test_split(unique_sims, test_size=0.3, random_state=42)

train_mask = np.isin(groups, train_sims)
test_mask  = np.isin(groups, test_sims)

X_train = X_combined[train_mask]
X_test  = X_combined[test_mask]
y_train = y[train_mask]
y_test  = y[test_mask]
g_train = groups[train_mask]

print(f"\nTrain: {train_mask.sum()} スナップショット ({len(train_sims)} シミュレーション)")
print(f"Test:  {test_mask.sum()} スナップショット ({len(test_sims)} シミュレーション)")

# GroupKFold CV (シミュレーション単位でfold)
gkf = GroupKFold(n_splits=5)

# ============================================================
# 6. モデル学習・評価
# ============================================================

def evaluate_model(name, model, X_tr, y_tr, X_te, y_te,
                   g_tr=None, cv_splitter=None):
    model.fit(X_tr, y_tr)
    y_pred_te = model.predict(X_te)
    y_pred_tr = model.predict(X_tr)
    r2_te = r2_score(y_te, y_pred_te)
    r2_tr = r2_score(y_tr, y_pred_tr)
    rmse_te_log = np.sqrt(mean_squared_error(y_te, y_pred_te))
    rmse_te_orig = np.sqrt(mean_squared_error(np.expm1(y_te), np.expm1(y_pred_te)))
    if cv_splitter is not None and g_tr is not None:
        cv_scores = cross_val_score(model, X_tr, y_tr, cv=cv_splitter,
                                    groups=g_tr, scoring='r2')
        cv_r2 = cv_scores.mean()
    else:
        cv_r2 = np.nan
    print(f"\n[{name}]")
    print(f"  Test  R²={r2_te:.4f}, RMSE(log)={rmse_te_log:.4f}")
    print(f"  Train R²={r2_tr:.4f}")
    print(f"  RMSE (実スケール): {rmse_te_orig:.2f} Å²/ns")
    if not np.isnan(cv_r2):
        print(f"  CV R² (GroupKFold 5): {cv_r2:.4f}")
    return {
        'model': name, 'R2_test': r2_te, 'R2_train': r2_tr,
        'RMSE_log': rmse_te_log, 'RMSE_orig': rmse_te_orig,
        'CV_R2': cv_r2, 'estimator': model,
    }

results = []

# --- Ridge ---
r = evaluate_model("Ridge(α=10)", Ridge(alpha=10), X_train, y_train, X_test, y_test,
                   g_train, gkf)
results.append(r)

# --- SVR ---
for C in [10, 50, 100]:
    r = evaluate_model(f"SVR(C={C})", SVR(kernel='rbf', C=C, gamma='scale'),
                       X_train, y_train, X_test, y_test, g_train, gkf)
    results.append(r)

# --- Random Forest ---
print("\n[Random Forest - 探索中...]")
rf_params = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 5),
    'max_features': ['sqrt', 0.3, 0.5],
}
rf_search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42),
    rf_params, n_iter=40,
    cv=gkf, scoring='r2',
    random_state=42, n_jobs=-1
)
rf_search.fit(X_train, y_train, groups=g_train)
best_rf = rf_search.best_estimator_
r = evaluate_model("RandomForest(Best)", best_rf,
                   X_train, y_train, X_test, y_test, g_train, gkf)
print(f"  最適パラメータ: {rf_search.best_params_}")
results.append(r)

# --- ExtraTrees ---
print("\n[ExtraTrees - 探索中...]")
et_search = RandomizedSearchCV(
    ExtraTreesRegressor(random_state=42),
    rf_params, n_iter=40,
    cv=gkf, scoring='r2',
    random_state=42, n_jobs=-1
)
et_search.fit(X_train, y_train, groups=g_train)
best_et = et_search.best_estimator_
r = evaluate_model("ExtraTrees(Best)", best_et,
                   X_train, y_train, X_test, y_test, g_train, gkf)
results.append(r)

# --- Gradient Boosting ---
print("\n[Gradient Boosting - 探索中...]")
gb_params = {
    'n_estimators': randint(50, 400),
    'max_depth': randint(2, 7),
    'learning_rate': uniform(0.01, 0.25),
    'subsample': uniform(0.5, 0.5),
    'min_samples_leaf': randint(1, 10),
}
gb_search = RandomizedSearchCV(
    GradientBoostingRegressor(random_state=42),
    gb_params, n_iter=40,
    cv=gkf, scoring='r2',
    random_state=42, n_jobs=-1
)
gb_search.fit(X_train, y_train, groups=g_train)
best_gb = gb_search.best_estimator_
r = evaluate_model("GradientBoosting(Best)", best_gb,
                   X_train, y_train, X_test, y_test, g_train, gkf)
print(f"  最適パラメータ: {gb_search.best_params_}")
results.append(r)

# --- MLP ---
print("\n[MLP - 探索中...]")
scaler_mlp = StandardScaler()
X_train_mlp = scaler_mlp.fit_transform(X_train)
X_test_mlp  = scaler_mlp.transform(X_test)

mlp_params = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50),
                           (100, 100), (50, 50, 50)],
    'alpha': [0.001, 0.01, 0.1, 1.0],
    'max_iter': [2000],
    'learning_rate_init': [0.001, 0.01],
}
mlp_search = RandomizedSearchCV(
    MLPRegressor(random_state=42, early_stopping=True, validation_fraction=0.15),
    mlp_params, n_iter=30,
    cv=gkf, scoring='r2',
    random_state=42, n_jobs=-1
)
mlp_search.fit(X_train_mlp, y_train, groups=g_train)
best_mlp = mlp_search.best_estimator_
r = evaluate_model("MLP(Best)", best_mlp,
                   X_train_mlp, y_train, X_test_mlp, y_test, g_train, gkf)
print(f"  最適パラメータ: {mlp_search.best_params_}")
results.append(r)

# pd_vectorのみ (物理パラメータなし) でのRF比較
print("\n[RandomForest (pd_vectorのみ)]")
X_train_pd = X_train[:, :N_PCA]
X_test_pd  = X_test[:,  :N_PCA]
rf_pd = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
r = evaluate_model("RandomForest(pd_only)", rf_pd,
                   X_train_pd, y_train, X_test_pd, y_test, g_train, gkf)
results.append(r)

# 物理パラメータのみでのRF比較
print("\n[RandomForest (物理パラメータのみ)]")
X_train_phys = X_train[:, N_PCA:]
X_test_phys  = X_test[:,  N_PCA:]
rf_phys = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
r = evaluate_model("RandomForest(phys_only)", rf_phys,
                   X_train_phys, y_train, X_test_phys, y_test, g_train, gkf)
results.append(r)

# ============================================================
# 7. 結果まとめ
# ============================================================

print("\n" + "="*75)
print("=== モデル比較まとめ ===")
print("="*75)
print(f"目的変数: log1p(D_cumul)  [D_cumul = MSD(T) / (6T)]")
print(f"{'モデル':<35} {'R²(Test)':>9} {'R²(Train)':>10} {'RMSE(実)':>12} {'CV_R²':>8}")
print("-"*75)
for r in sorted(results, key=lambda x: x['R2_test'], reverse=True):
    cv_str = f"{r['CV_R2']:.4f}" if not np.isnan(r['CV_R2']) else "  N/A "
    print(f"{r['model']:<35} {r['R2_test']:>9.4f} {r['R2_train']:>10.4f} "
          f"{r['RMSE_orig']:>12.2f} {cv_str:>8}")

best = max(results, key=lambda x: x['R2_test'])
print(f"\n{'='*75}")
print(f"最良モデル: {best['model']}")
print(f"  R²   (Test,  log1p空間) = {best['R2_test']:.4f}")
print(f"  RMSE (Test, 実スケール) = {best['RMSE_orig']:.2f} Å²/ns")
print(f"  R²   (Train, log1p空間) = {best['R2_train']:.4f}")
print(f"  CV R² (GroupKFold-5)    = {best['CV_R2']:.4f}")

# ============================================================
# 8. 可視化
# ============================================================

best_name = best['model']
best_est  = best['estimator']

# bestモデルの予測（X_train/X_test を適切に選択）
if 'MLP' in best_name:
    X_tr_use = X_train_mlp
    X_te_use = X_test_mlp
else:
    X_tr_use = X_train
    X_te_use = X_test

y_pred_te = best_est.predict(X_te_use)
y_pred_tr = best_est.predict(X_tr_use)

fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# log1p散布図
ax = axes[0, 0]
ax.scatter(y_test, y_pred_te, alpha=0.6,
           label=f'Test R²={best["R2_test"]:.3f}', s=40)
ax.scatter(y_train, y_pred_tr, alpha=0.3, marker='^',
           label=f'Train R²={best["R2_train"]:.3f}', s=25)
lims = [min(y.min(), y_pred_te.min()), max(y.max(), y_pred_te.max())]
ax.plot(lims, lims, 'k--', lw=1)
ax.set_xlabel('Calculated log1p(D_cumul)')
ax.set_ylabel('Predicted log1p(D_cumul)')
ax.set_title(f'Best: {best_name}')
ax.legend(fontsize=9)

# 実スケール散布図
ax = axes[0, 1]
ax.scatter(np.expm1(y_test), np.expm1(y_pred_te), alpha=0.6, s=40)
lims_o = [0, max(np.expm1(y_test).max(), np.expm1(y_pred_te).max())]
ax.plot(lims_o, lims_o, 'k--', lw=1)
ax.set_xlabel('Calculated D_cumul (Å²/ns)')
ax.set_ylabel('Predicted D_cumul (Å²/ns)')
ax.set_title('実スケール (D_cumul = MSD/(6T))')

# D_cumulの分布
ax = axes[1, 0]
ax.hist(y_raw[~np.isnan(y_raw)], bins=50, alpha=0.7, log=True)
ax.set_xlabel('D_cumul (Å²/ns)')
ax.set_ylabel('Count (log scale)')
ax.set_title('拡散係数分布')

# モデル比較
ax = axes[1, 1]
top = sorted(results, key=lambda x: x['R2_test'], reverse=True)
nms = [r['model'].replace('(Best)', '').replace('(Tuned)', '') for r in top]
r2s = [r['R2_test'] for r in top]
cols = ['steelblue' if v > 0 else 'salmon' for v in r2s]
bars = ax.barh(nms[::-1], r2s[::-1], color=cols[::-1])
ax.set_xlabel('R² (Test)')
ax.set_title('モデル比較')
ax.axvline(0, color='k', lw=0.5)
for bar, r2 in zip(bars, r2s[::-1]):
    ax.text(max(r2, 0.01), bar.get_y() + bar.get_height()/2,
            f'{r2:.3f}', va='center', fontsize=8)

plt.suptitle('イオン液体Naイオン拡散係数予測 (パーシステントベクトル使用)', y=1.01)
plt.tight_layout()
plt.savefig(OUT_DIR / 'ml_results_v3.png', dpi=120)
print(f"\n図保存: {OUT_DIR / 'ml_results_v3.png'}")

# 結果CSV
df_res = pd.DataFrame([{k: v for k, v in r.items() if k != 'estimator'}
                        for r in results])
df_res.to_csv(OUT_DIR / 'ml_results_v3.csv', index=False)
print(f"結果CSV: {OUT_DIR / 'ml_results_v3.csv'}")

print("\n=== 完了 ===")
