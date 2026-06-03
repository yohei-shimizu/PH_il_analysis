"""
イオン液体の拡散係数再計算 + 機械学習モデル構築

データ構造:
- LAMMPS MSD: 各シミュレーション100ns (10ファイル×10ns, dt=1fs)
- pd_vector: p000-p499.dat (500ファイル、各シミュレーションの5時間ポイントの構造スナップショット)
- il_data_gra.csv: 500行の拡散係数データ (元の計算値)

pd_vectorの順序 (0-indexed):
  p_idx = comp_idx*50 + temp_idx*10 + ef_idx*5 + snap_idx
  comp_idx: 0-4 (No.01-No.05)
  temp_idx: 0-4 (250K, 300K, 350K, 400K, 450K)
  ef_idx: 0-1 (Efield=0, Efield=1)
  snap_idx: 0-4 (10ns=100ns, 6ns=60ns, 7ns=70ns, 8ns=80ns, 9ns=90ns)

再計算方針:
  全100nsのMSDデータをロードし、後半40-100nsの線形フィットでDを計算
  D = slope / 6 (total MSD使用, 元計算と同じ)
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
SNAP_LABELS = ['10ns', '6ns', '7ns', '8ns', '9ns']  # 100ns, 60ns, 70ns, 80ns, 90ns

# ============================================================
# 1. MSDファイルから拡散係数を計算する関数
# ============================================================

def load_all_msd(folder, prefix, temp):
    """全10セグメントのNa+MSDを読み込み連結"""
    times, msds = [], []
    for seg in range(1, 11):
        fpath = os.path.join(folder, f'{prefix}_{temp}K.{seg:02d}.msd')
        if not os.path.exists(fpath):
            return None, None
        try:
            data = np.loadtxt(fpath)
            times.extend(data[:, 0] / 1e6)   # ns単位
            msds.extend(data[:, 4])            # total MSD (Å²)
        except Exception:
            return None, None
    return np.array(times), np.array(msds)


def calc_diffusion(times, msds, fit_start_ns=40.0):
    """
    線形フィットでDを計算
    fit_start_ns以降のデータを使用
    D = slope / 6 [Å²/ns]
    物理的に負の値が出た場合は全区間でフィット
    """
    if times is None or len(times) == 0:
        return np.nan

    for t_start in [fit_start_ns, 20.0, 0.0]:
        mask = times >= t_start
        if mask.sum() < 10:
            continue
        t_fit = times[mask]
        msd_fit = msds[mask]
        slope, intercept, r, p, se = stats.linregress(t_fit, msd_fit)
        if slope > 0:
            return slope / 6.0  # D [Å²/ns]

    # どの区間でも正にならない場合はゼロ近傍とする
    mask = times >= 0
    slope, _, _, _, _ = stats.linregress(times[mask], msds[mask])
    return max(slope / 6.0, 0.0)


# ============================================================
# 2. BF4_gra と TFSI_gra の拡散係数を計算
# ============================================================

def calc_all_diffusion():
    results = {}

    # BF4_gra: No.01-No.05 × 250-450K × ef=0,1
    for comp_idx in range(5):
        no = comp_idx + 1
        for temp_idx, temp in enumerate(TEMPS):
            for ef_idx in range(2):
                ef_str = f'_{temp}K_ef' if ef_idx == 1 else f'_{temp}K'
                folder_name = f'Na_EMI_BF4_gra{ef_str}'
                folder = os.path.join(BASE, 'BF4_gra', folder_name,
                                      f'No.{no:02d}{ef_str}')
                times, msds = load_all_msd(folder, 'Na', temp)
                key = ('BF4', comp_idx, temp_idx, ef_idx)
                results[key] = calc_diffusion(times, msds)

    # TFSI_gra: No.01-No.05 × 250-450K × ef=0,1
    for comp_idx in range(5):
        no = comp_idx + 1
        for temp_idx, temp in enumerate(TEMPS):
            for ef_idx in range(2):
                ef_str = f'_{temp}K_ef' if ef_idx == 1 else f'_{temp}K'
                folder_name = f'Na_EMI_TFSI_gra{ef_str}'
                folder = os.path.join(BASE, 'TFSI_gra', folder_name,
                                      f'No.{no:02d}{ef_str}')
                times, msds = load_all_msd(folder, 'Na', temp)
                key = ('TFSI', comp_idx, temp_idx, ef_idx)
                results[key] = calc_diffusion(times, msds)

    return results


print("=== 拡散係数計算中 ===")
D_results = calc_all_diffusion()

# 計算できたケース数確認
valid = sum(1 for v in D_results.values() if not np.isnan(v))
print(f"有効な計算: {valid} / {len(D_results)}")

# D値の統計
D_values = [v for v in D_results.values() if not np.isnan(v)]
print(f"D範囲: {min(D_values):.2f} ~ {max(D_values):.2f} Å²/ns")
print(f"D中央値: {np.median(D_values):.2f} Å²/ns")
print(f"負の値: {sum(1 for v in D_values if v < 0)} 件")

# ============================================================
# 3. pd_vectorを読み込み、D値と正しく対応付ける
# ============================================================

print("\n=== pd_vector読み込み中 ===")
pd_vecs = []
for i in range(500):
    fpath = f'{BASE}/pd_vector/p{i:03d}.dat'
    vec = np.loadtxt(fpath)
    pd_vecs.append(vec)
pd_vecs = np.array(pd_vecs)
print(f"pd_vector形状: {pd_vecs.shape}")

# pd_vector→D値のマッピング
# p_idx = comp_idx*50 + temp_idx*10 + ef_idx*5 + snap_idx
# BF4: comp_idx 0-4, TFSI: 250+comp_idx*50+...

D_aligned = np.zeros(500)
for p_idx in range(500):
    if p_idx < 250:
        il_type = 'BF4'
        local = p_idx
    else:
        il_type = 'TFSI'
        local = p_idx - 250

    comp_idx = local // 50
    temp_idx = (local % 50) // 10
    ef_idx = (local % 10) // 5
    # snap_idx不要: 同じシミュレーションの5スナップショットは同じD値を使用

    key = (il_type, comp_idx, temp_idx, ef_idx)
    D_aligned[p_idx] = D_results.get(key, np.nan)

nan_count = np.sum(np.isnan(D_aligned))
print(f"NaN (未計算): {nan_count}")

# NaNを除外
valid_mask = ~np.isnan(D_aligned)
X_all = pd_vecs[valid_mask]
y_all = D_aligned[valid_mask]

print(f"有効サンプル数: {X_all.shape[0]}")
print(f"D範囲: {y_all.min():.2f} ~ {y_all.max():.2f} Å²/ns")

# 結果をCSVに保存
sim_labels = []
for p_idx in range(500):
    if p_idx < 250:
        il_type = 'BF4'
        local = p_idx
    else:
        il_type = 'TFSI'
        local = p_idx - 250
    comp_idx = local // 50
    temp_idx = (local % 50) // 10
    ef_idx = (local % 10) // 5
    snap_idx = local % 5
    sim_labels.append({
        'p_idx': p_idx,
        'il_type': il_type,
        'comp_idx': comp_idx + 1,
        'temp': TEMPS[temp_idx],
        'efield': ef_idx,
        'snap': SNAP_LABELS[snap_idx],
        'D_recalc': D_aligned[p_idx]
    })

df_D = pd.DataFrame(sim_labels)
df_D.to_csv(DATA_DIR / 'D_recalculated.csv', index=False)
print(f"D値CSV保存: {DATA_DIR / 'D_recalculated.csv'}")

# ============================================================
# 4. 機械学習モデルの構築
# ============================================================

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

print("\n=== 機械学習モデル構築 ===")

# 外れ値処理: 99パーセンタイルでクリッピング
y_max = np.percentile(y_all, 99)
y_clipped = np.clip(y_all, 0, y_max)
print(f"99パーセンタイルでクリッピング: D <= {y_max:.1f} Å²/ns")
n_clipped = np.sum(y_all > y_max)
print(f"クリッピング対象: {n_clipped} 件")

X = X_all.copy()
y = y_clipped

# PCA前処理
scaler_pre = StandardScaler()
X_scaled = scaler_pre.fit_transform(X)

pca = PCA(n_components=100)
X_pca = pca.fit_transform(X_scaled)
cum_var = np.cumsum(pca.explained_variance_ratio_)
n_90 = np.argmax(cum_var >= 0.90) + 1
n_95 = np.argmax(cum_var >= 0.95) + 1
print(f"PCA: 90%寄与率 → {n_90}次元, 95%寄与率 → {n_95}次元")

# Train/Test分割 (70/30)
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.3, random_state=42
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

print(f"学習データ: {X_train.shape[0]}件, テストデータ: {X_test.shape[0]}件")

kf = KFold(n_splits=5, shuffle=True, random_state=42)

def evaluate_model(name, model, X_tr, y_tr, X_te, y_te):
    """モデルを学習・評価してR2とRMSEを返す"""
    model.fit(X_tr, y_tr)
    y_pred_te = model.predict(X_te)
    y_pred_tr = model.predict(X_tr)
    r2_te = r2_score(y_te, y_pred_te)
    r2_tr = r2_score(y_tr, y_pred_tr)
    rmse_te = np.sqrt(mean_squared_error(y_te, y_pred_te))
    rmse_tr = np.sqrt(mean_squared_error(y_tr, y_pred_tr))
    cv_r2 = cross_val_score(model, X_tr, y_tr, cv=kf, scoring='r2').mean()
    print(f"\n[{name}]")
    print(f"  Test  R²={r2_te:.4f}, RMSE={rmse_te:.2f}")
    print(f"  Train R²={r2_tr:.4f}, RMSE={rmse_tr:.2f}")
    print(f"  CV R² (5-fold): {cv_r2:.4f}")
    return {'model': name, 'R2_test': r2_te, 'RMSE_test': rmse_te,
            'R2_train': r2_tr, 'RMSE_train': rmse_tr, 'CV_R2': cv_r2,
            'estimator': model}

results = []

# --- ElasticNet ---
en = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000)
results.append(evaluate_model("ElasticNet", en, X_train_sc, y_train, X_test_sc, y_test))

# --- Ridge ---
ridge = Ridge(alpha=1.0)
results.append(evaluate_model("Ridge", ridge, X_train_sc, y_train, X_test_sc, y_test))

# --- SVR (RBF) ---
svr = SVR(kernel='rbf', C=100, gamma='scale', epsilon=10)
results.append(evaluate_model("SVR(RBF)", svr, X_train_sc, y_train, X_test_sc, y_test))

# --- Random Forest (Tuned) ---
print("\n[Random Forest - ハイパーパラメータ探索中...]")
rf_params = {
    'n_estimators': randint(100, 500),
    'max_depth': randint(3, 30),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 'log2', 0.3, 0.5],
}
rf = RandomizedSearchCV(
    RandomForestRegressor(random_state=42),
    rf_params, n_iter=30, cv=kf, scoring='r2',
    random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)
best_rf = rf.best_estimator_
results.append(evaluate_model("RandomForest(Tuned)", best_rf, X_train, y_train, X_test, y_test))
print(f"  最適パラメータ: {rf.best_params_}")

# --- Gradient Boosting (Tuned) ---
print("\n[Gradient Boosting - ハイパーパラメータ探索中...]")
gb_params = {
    'n_estimators': randint(100, 500),
    'max_depth': randint(2, 8),
    'learning_rate': uniform(0.01, 0.2),
    'min_samples_split': randint(2, 20),
    'subsample': uniform(0.6, 0.4),
}
gb = RandomizedSearchCV(
    GradientBoostingRegressor(random_state=42),
    gb_params, n_iter=30, cv=kf, scoring='r2',
    random_state=42, n_jobs=-1
)
gb.fit(X_train, y_train)
best_gb = gb.best_estimator_
results.append(evaluate_model("GradientBoosting(Tuned)", best_gb, X_train, y_train, X_test, y_test))
print(f"  最適パラメータ: {gb.best_params_}")

# --- MLP Neural Network ---
print("\n[MLP Neural Network - ハイパーパラメータ探索中...]")
mlp_params = {
    'hidden_layer_sizes': [(200, 200), (300, 300), (200, 100, 50),
                           (300, 200, 100), (500, 300, 100)],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate_init': [0.001, 0.0005, 0.0001],
    'max_iter': [2000],
}
mlp = RandomizedSearchCV(
    MLPRegressor(random_state=42),
    mlp_params, n_iter=20, cv=kf, scoring='r2',
    random_state=42, n_jobs=-1
)
mlp.fit(X_train_sc, y_train)
best_mlp = mlp.best_estimator_
results.append(evaluate_model("MLP(Tuned)", best_mlp, X_train_sc, y_train, X_test_sc, y_test))
print(f"  最適パラメータ: {mlp.best_params_}")

# ============================================================
# 5. 結果まとめ
# ============================================================

print("\n" + "="*60)
print("=== モデル比較まとめ ===")
print("="*60)
print(f"{'モデル':<30} {'R²(Test)':>10} {'RMSE(Test)':>12} {'CV R²':>10}")
print("-"*62)
for r in sorted(results, key=lambda x: x['R2_test'], reverse=True):
    print(f"{r['model']:<30} {r['R2_test']:>10.4f} {r['RMSE_test']:>12.2f} {r['CV_R2']:>10.4f}")

best_result = max(results, key=lambda x: x['R2_test'])
print(f"\n最良モデル: {best_result['model']}")
print(f"  R²  (Test) = {best_result['R2_test']:.4f}")
print(f"  RMSE(Test) = {best_result['RMSE_test']:.2f} Å²/ns")
print(f"  R²  (Train)= {best_result['R2_train']:.4f}")
print(f"  RMSE(Train)= {best_result['RMSE_train']:.2f} Å²/ns")
print(f"  CV R²      = {best_result['CV_R2']:.4f}")

# 結果をCSVに保存
df_results = pd.DataFrame([{k: v for k, v in r.items() if k != 'estimator'}
                            for r in results])
df_results.to_csv(OUT_DIR / 'ml_results.csv', index=False)
print(f"\n機械学習結果CSV: {OUT_DIR / 'ml_results.csv'}")

# ============================================================
# 6. 可視化
# ============================================================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

best_model_name = best_result['model']
best_est = best_result['estimator']

# Train/Testに対応するスケーリング済みデータを判定
if 'SVR' in best_model_name or 'ElasticNet' in best_model_name or \
   'Ridge' in best_model_name or 'MLP' in best_model_name:
    X_tr_use, X_te_use = X_train_sc, X_test_sc
else:
    X_tr_use, X_te_use = X_train, X_test

y_pred_test = best_est.predict(X_te_use)
y_pred_train = best_est.predict(X_tr_use)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 散布図
ax = axes[0]
ax.scatter(y_test, y_pred_test, alpha=0.6, label=f'Test (R²={best_result["R2_test"]:.3f})')
ax.scatter(y_train, y_pred_train, alpha=0.4, label=f'Train (R²={best_result["R2_train"]:.3f})', marker='^')
lims = [min(y.min(), y_pred_test.min()), max(y.max(), y_pred_test.max())]
ax.plot(lims, lims, 'k--', lw=1)
ax.set_xlabel('Calculated D (Å²/ns)')
ax.set_ylabel('Predicted D (Å²/ns)')
ax.set_title(f'Best Model: {best_model_name}')
ax.legend()

# D値の分布
ax = axes[1]
ax.hist(y, bins=50, alpha=0.7, label='All (clipped)')
ax.set_xlabel('D (Å²/ns)')
ax.set_ylabel('Count')
ax.set_title('Distribution of Diffusion Coefficients')
ax.legend()

plt.tight_layout()
plt.savefig(OUT_DIR / 'ml_results.png', dpi=120)
print(f"図保存: {OUT_DIR / 'ml_results.png'}")

# 各モデルの比較グラフ
fig2, ax2 = plt.subplots(figsize=(10, 5))
names = [r['model'] for r in results]
r2s = [r['R2_test'] for r in results]
bars = ax2.barh(names, r2s)
ax2.set_xlabel('R² (Test)')
ax2.set_title('Model Comparison (R² on Test Set)')
ax2.axvline(0, color='k', linewidth=0.5)
for bar, r2 in zip(bars, r2s):
    ax2.text(max(r2, 0) + 0.01, bar.get_y() + bar.get_height()/2,
             f'{r2:.3f}', va='center')
plt.tight_layout()
plt.savefig(OUT_DIR / 'model_comparison.png', dpi=120)
print(f"モデル比較図保存: {OUT_DIR / 'model_comparison.png'}")

print("\n完了!")
