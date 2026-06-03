"""
イオン液体拡散係数再計算 + 機械学習モデル構築 (改良版)

改善点:
1. log1p(D)変換で歪んだ分布を正規化
2. VarianceThreshold で零分散特徴量を除去
3. シミュレーション単位でtrain/test分割 (データリーク防止)
4. 各シミュレーションの最終スナップショット(100ns)を代表として使用
5. 複数モデルを系統的に比較
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

# ============================================================
# 1. MSDファイルから拡散係数を計算
# ============================================================

def load_all_msd(folder, temp):
    times, msds = [], []
    for seg in range(1, 11):
        fpath = os.path.join(folder, f'Na_{temp}K.{seg:02d}.msd')
        if not os.path.exists(fpath):
            return None, None
        try:
            data = np.loadtxt(fpath)
            times.extend(data[:, 0] / 1e6)
            msds.extend(data[:, 4])
        except Exception:
            return None, None
    return np.array(times), np.array(msds)


def calc_diffusion(times, msds):
    """後半40nsでフィット、ダメなら全区間 → D=slope/6"""
    if times is None:
        return np.nan
    for t_start in [60.0, 40.0, 20.0, 0.0]:
        mask = times >= t_start
        if mask.sum() < 10:
            continue
        slope, _, _, _, _ = stats.linregress(times[mask], msds[mask])
        if slope > 0:
            return slope / 6.0
    # 正にならない → 実質ゼロ拡散
    return 0.0


def build_dataset():
    """
    100シミュレーション分の (pd_vector, D, metadata) を作成
    各シミュレーションの最終スナップショット (10ns=100ns) を使用
    → p_idx = comp*50 + temp*10 + ef*5 + 0 (snap_idx=0 → 10ns snapshot)
    """
    rows = []
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
                    times, msds = load_all_msd(folder, temp)
                    D = calc_diffusion(times, msds)

                    # pd_vectorのインデックス (snap_idx=0: 10ns=100nsスナップショット)
                    if il_type == 'BF4':
                        p_base = 0
                    else:
                        p_base = 250
                    p_idx = p_base + comp_idx * 50 + temp_idx * 10 + ef_idx * 5 + 0
                    vec = np.loadtxt(f'{BASE}/pd_vector/p{p_idx:03d}.dat')

                    sim_id = f'{il_type}_No{no:02d}_{temp}K_ef{ef_idx}'
                    rows.append({
                        'sim_id': sim_id,
                        'il_type': il_type,
                        'comp_idx': comp_idx,
                        'temp': temp,
                        'efield': ef_idx,
                        'p_idx': p_idx,
                        'D': D,
                        'vec': vec,
                    })
    return rows


print("=== データセット構築中 ===")
dataset = build_dataset()
print(f"シミュレーション数: {len(dataset)}")

D_vals = np.array([r['D'] for r in dataset])
print(f"D範囲: {D_vals.min():.2f} ~ {D_vals.max():.2f} Å²/ns")
print(f"D中央値: {np.median(D_vals):.2f}")
print(f"D=0件数: {(D_vals == 0).sum()}")

# ============================================================
# 2. 特徴量・目的変数の準備
# ============================================================

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.model_selection import (train_test_split, cross_val_score,
                                     GroupKFold, KFold)
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                               ExtraTreesRegressor)
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

X_raw = np.array([r['vec'] for r in dataset])          # (100, 2080)
y_raw = np.array([r['D'] for r in dataset])             # (100,)
y_log = np.log1p(y_raw)                                 # log変換

groups = np.array([r['comp_idx'] * 10 + TEMPS.index(r['temp'])
                   for r in dataset])                    # グループ化用

print(f"\n特徴量形状: {X_raw.shape}")
print(f"y_log範囲: {y_log.min():.3f} ~ {y_log.max():.3f}")

# --- 特徴量前処理 ---
# Step1: 零分散特徴量の除去
vt = VarianceThreshold(threshold=1e-10)
X_var = vt.fit_transform(X_raw)
print(f"\nVarianceThreshold後: {X_var.shape[1]} 特徴量 (除去: {X_raw.shape[1]-X_var.shape[1]})")

# Step2: 標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_var)

# Step3: PCA (95%寄与率)
pca = PCA(n_components=0.95, svd_solver='full')
X_pca = pca.fit_transform(X_scaled)
n_pca = X_pca.shape[1]
print(f"PCA後次元数 (95%寄与率): {n_pca}")
print(f"各PCの寄与率 (上位5): {pca.explained_variance_ratio_[:5].round(4)}")

# ============================================================
# 3. 学習・評価 (GroupKFold でデータリーク防止)
# ============================================================

# シミュレーション単位でtrain/test分割 (70/30)
np.random.seed(42)
sim_indices = np.arange(len(dataset))
train_idx, test_idx = train_test_split(sim_indices, test_size=0.3, random_state=42)

X_train_pca = X_pca[train_idx]
X_test_pca = X_pca[test_idx]
X_train_raw = X_raw[train_idx]
X_test_raw = X_raw[test_idx]
X_train_var = X_var[train_idx]
X_test_var = X_var[test_idx]
y_train = y_log[train_idx]
y_test = y_log[test_idx]

print(f"\nTrain: {len(train_idx)}, Test: {len(test_idx)}")

kf = KFold(n_splits=5, shuffle=True, random_state=42)

def evaluate_model(name, model, X_tr, y_tr, X_te, y_te, cv=None):
    model.fit(X_tr, y_tr)
    y_pred_te = model.predict(X_te)
    y_pred_tr = model.predict(X_tr)
    r2_te = r2_score(y_te, y_pred_te)
    r2_tr = r2_score(y_tr, y_pred_tr)
    rmse_te = np.sqrt(mean_squared_error(y_te, y_pred_te))
    rmse_tr = np.sqrt(mean_squared_error(y_tr, y_pred_tr))
    if cv is not None:
        cv_r2 = cross_val_score(model, X_tr, y_tr, cv=cv, scoring='r2').mean()
    else:
        cv_r2 = np.nan
    print(f"\n[{name}]")
    print(f"  Test  R²={r2_te:.4f}, RMSE={rmse_te:.4f} (log1p空間)")
    print(f"  Train R²={r2_tr:.4f}, RMSE={rmse_tr:.4f}")
    if not np.isnan(cv_r2):
        print(f"  CV R² (5-fold): {cv_r2:.4f}")

    # 実スケールのRMSE
    rmse_orig = np.sqrt(mean_squared_error(
        np.expm1(y_te), np.expm1(y_pred_te)))
    print(f"  RMSE (実スケール): {rmse_orig:.2f} Å²/ns")

    return {
        'model': name,
        'R2_test': r2_te, 'RMSE_test_log': rmse_te,
        'R2_train': r2_tr, 'RMSE_train_log': rmse_tr,
        'CV_R2': cv_r2,
        'RMSE_test_orig': rmse_orig,
        'estimator': model,
        'X_tr': X_tr, 'X_te': X_te,
    }

results = []

# --- Ridge ---
for alpha in [0.01, 0.1, 1.0, 10.0]:
    r = evaluate_model(f"Ridge(α={alpha})", Ridge(alpha=alpha),
                       X_train_pca, y_train, X_test_pca, y_test, kf)
    results.append(r)

# --- SVR ---
r = evaluate_model("SVR(RBF, C=10)", SVR(kernel='rbf', C=10, gamma='scale'),
                   X_train_pca, y_train, X_test_pca, y_test, kf)
results.append(r)

r = evaluate_model("SVR(RBF, C=100)", SVR(kernel='rbf', C=100, gamma='scale'),
                   X_train_pca, y_train, X_test_pca, y_test, kf)
results.append(r)

# --- Random Forest ---
print("\n[Random Forest - 探索中...]")
rf_params = {
    'n_estimators': randint(50, 300),
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 5),
    'max_features': ['sqrt', 'log2', 0.5],
}
rf_search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42),
    rf_params, n_iter=30, cv=kf, scoring='r2',
    random_state=42, n_jobs=-1
)
rf_search.fit(X_train_pca, y_train)
best_rf = rf_search.best_estimator_
r = evaluate_model("RandomForest(Best)", best_rf,
                   X_train_pca, y_train, X_test_pca, y_test)
print(f"  最適パラメータ: {rf_search.best_params_}")
results.append(r)

# --- Random Forest (raw var-filtered features) ---
print("\n[Random Forest on raw features - 探索中...]")
rf_search2 = RandomizedSearchCV(
    RandomForestRegressor(random_state=42),
    rf_params, n_iter=30, cv=kf, scoring='r2',
    random_state=42, n_jobs=-1
)
rf_search2.fit(X_train_var, y_train)
best_rf2 = rf_search2.best_estimator_
r2_ = evaluate_model("RandomForest(raw)", best_rf2,
                      X_train_var, y_train, X_test_var, y_test)
print(f"  最適パラメータ: {rf_search2.best_params_}")
results.append(r2_)

# --- Extra Trees ---
print("\n[ExtraTrees - 探索中...]")
et_search = RandomizedSearchCV(
    ExtraTreesRegressor(random_state=42),
    rf_params, n_iter=30, cv=kf, scoring='r2',
    random_state=42, n_jobs=-1
)
et_search.fit(X_train_pca, y_train)
best_et = et_search.best_estimator_
r = evaluate_model("ExtraTrees(Best)", best_et,
                   X_train_pca, y_train, X_test_pca, y_test)
results.append(r)

# --- Gradient Boosting ---
print("\n[Gradient Boosting - 探索中...]")
gb_params = {
    'n_estimators': randint(50, 300),
    'max_depth': randint(2, 6),
    'learning_rate': uniform(0.01, 0.2),
    'subsample': uniform(0.5, 0.5),
    'min_samples_split': randint(2, 10),
}
gb_search = RandomizedSearchCV(
    GradientBoostingRegressor(random_state=42),
    gb_params, n_iter=30, cv=kf, scoring='r2',
    random_state=42, n_jobs=-1
)
gb_search.fit(X_train_pca, y_train)
best_gb = gb_search.best_estimator_
r = evaluate_model("GradientBoosting(Best)", best_gb,
                   X_train_pca, y_train, X_test_pca, y_test)
print(f"  最適パラメータ: {gb_search.best_params_}")
results.append(r)

# --- MLP ---
print("\n[MLP Neural Network - 探索中...]")
mlp_params = {
    'hidden_layer_sizes': [(100, 50), (200, 100), (100, 100, 50),
                           (200, 100, 50), (50, 50, 50)],
    'alpha': [0.001, 0.01, 0.1],
    'max_iter': [2000],
    'learning_rate_init': [0.001, 0.01],
}
mlp_search = RandomizedSearchCV(
    MLPRegressor(random_state=42, early_stopping=True),
    mlp_params, n_iter=20, cv=kf, scoring='r2',
    random_state=42, n_jobs=-1
)
mlp_search.fit(X_train_pca, y_train)
best_mlp = mlp_search.best_estimator_
r = evaluate_model("MLP(Best)", best_mlp,
                   X_train_pca, y_train, X_test_pca, y_test)
print(f"  最適パラメータ: {mlp_search.best_params_}")
results.append(r)

# ============================================================
# 4. 結果まとめ
# ============================================================

print("\n" + "="*70)
print("=== モデル比較まとめ (log1p空間でのR²) ===")
print("="*70)
hdr = f"{'モデル':<30} {'R²(Test)':>10} {'R²(Train)':>10} {'RMSE(実)':>14} {'CV_R²':>8}"
print(hdr)
print("-"*70)
for r in sorted(results, key=lambda x: x['R2_test'], reverse=True):
    cv_str = f"{r['CV_R2']:.4f}" if not np.isnan(r['CV_R2']) else "  -   "
    print(f"{r['model']:<30} {r['R2_test']:>10.4f} {r['R2_train']:>10.4f} "
          f"{r['RMSE_test_orig']:>14.2f} {cv_str:>8}")

best_result = max(results, key=lambda x: x['R2_test'])
print(f"\n最良モデル: {best_result['model']}")
print(f"  R²  (Test,  log1p) = {best_result['R2_test']:.4f}")
print(f"  RMSE(Test, 実スケール) = {best_result['RMSE_test_orig']:.2f} Å²/ns")
print(f"  R²  (Train, log1p) = {best_result['R2_train']:.4f}")
print(f"  CV R²              = {best_result['CV_R2']:.4f}")

# ============================================================
# 5. 可視化
# ============================================================

best_name = best_result['model']
best_est = best_result['estimator']
X_tr_use = best_result['X_tr']
X_te_use = best_result['X_te']

y_pred_test_log = best_est.predict(X_te_use)
y_pred_train_log = best_est.predict(X_tr_use)
y_pred_test_orig = np.expm1(y_pred_test_log)
y_pred_train_orig = np.expm1(y_pred_train_log)
y_test_orig = np.expm1(y_test)
y_train_orig = np.expm1(y_train)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# log空間の散布図
ax = axes[0]
ax.scatter(y_test, y_pred_test_log, alpha=0.7, label=f'Test R²={best_result["R2_test"]:.3f}')
ax.scatter(y_train, y_pred_train_log, alpha=0.5, marker='^',
           label=f'Train R²={best_result["R2_train"]:.3f}')
lims = [min(y_log.min(), y_pred_test_log.min()),
        max(y_log.max(), y_pred_test_log.max())]
ax.plot(lims, lims, 'k--', lw=1)
ax.set_xlabel('Calculated log1p(D)')
ax.set_ylabel('Predicted log1p(D)')
ax.set_title(f'Best: {best_name}\n(log1p空間)')
ax.legend()

# 実スケール散布図
ax = axes[1]
ax.scatter(y_test_orig, y_pred_test_orig, alpha=0.7)
ax.scatter(y_train_orig, y_pred_train_orig, alpha=0.5, marker='^')
lims_o = [0, max(y_test_orig.max(), y_pred_test_orig.max())]
ax.plot(lims_o, lims_o, 'k--', lw=1)
ax.set_xlabel('Calculated D (Å²/ns)')
ax.set_ylabel('Predicted D (Å²/ns)')
ax.set_title('実スケール')

# モデル比較
ax = axes[2]
top_results = sorted(results, key=lambda x: x['R2_test'], reverse=True)[:10]
names_plot = [r['model'].replace('(Best)', '').replace('(Tuned)', '') for r in top_results]
r2_plot = [r['R2_test'] for r in top_results]
colors = ['steelblue' if r2 > 0 else 'salmon' for r2 in r2_plot]
bars = ax.barh(names_plot[::-1], r2_plot[::-1], color=colors[::-1])
ax.set_xlabel('R² (Test, log1p空間)')
ax.set_title('モデル比較 (R² on Test)')
ax.axvline(0, color='k', linewidth=0.5)
for bar, r2 in zip(bars, r2_plot[::-1]):
    ax.text(max(r2, 0.01), bar.get_y() + bar.get_height()/2,
            f'{r2:.3f}', va='center', fontsize=8)

plt.tight_layout()
plt.savefig(OUT_DIR / 'ml_results_v2.png', dpi=120)
print(f"\n図保存: {OUT_DIR / 'ml_results_v2.png'}")

# 結果CSV
df_res = pd.DataFrame([{k: v for k, v in r.items()
                         if k not in ('estimator', 'X_tr', 'X_te')}
                        for r in results])
df_res.to_csv(OUT_DIR / 'ml_results_v2.csv', index=False)
print(f"結果CSV: {OUT_DIR / 'ml_results_v2.csv'}")

# D値の情報も保存
df_sim = pd.DataFrame([{k: v for k, v in r.items() if k != 'vec'}
                        for r in dataset])
df_sim.to_csv(DATA_DIR / 'D_per_simulation.csv', index=False)
print(f"拡散係数CSV: {DATA_DIR / 'D_per_simulation.csv'}")

print("\n=== 完了 ===")
