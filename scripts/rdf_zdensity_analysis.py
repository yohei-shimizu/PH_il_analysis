"""
RDF・z密度プロファイル解析 vs pd_vector比較
1. グラフェンからみたNa+ z密度プロファイル (グラフェン相関係数)
2. RDF (Na-all g(r)) の抽出
3. pd_vectorとの予測精度比較 (ML)
4. 構造的相関解析 (Pearson/Spearman相関係数)
"""

import numpy as np
import pandas as pd
import subprocess
import io
import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold, train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# 設定
# ============================================================
REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = str(REPO_ROOT / 'cache')
OUT_DIR   = REPO_ROOT / 'results'
DATA_DIR  = REPO_ROOT / 'data'
# Raw LAMMPS trajectory tree (Windows path, only needed for regeneration).
RAW_WIN_BASE = r'D:\path\to\raw\trajectories'
os.makedirs(CACHE_DIR, exist_ok=True)

TEMPS    = [250, 300, 350, 400, 450]
SNAP_NS  = [100, 60, 70, 80, 90]
SNAP_SEG = [10,   6,  7,  8,  9]
N_RDF    = 600   # RDFビン数
N_ZDENSITY = 50  # z密度プロファイルビン数
N_PD     = 2080  # pd_vector次元数

# ============================================================
# ファイルパス生成
# ============================================================
def build_sim_list():
    """全100シミュレーション × 5スナップショット = 500エントリのリストを生成"""
    rows = []
    sim_id = 0
    for il_type in ['BF4', 'TFSI']:
        system = 'BF4_gra' if il_type == 'BF4' else 'TFSI_gra'
        anion  = 'BF4' if il_type == 'BF4' else 'TFSI'
        p_base = 0 if il_type == 'BF4' else 250

        for comp_idx in range(5):
            no = comp_idx + 1
            for temp_idx, temp in enumerate(TEMPS):
                for ef_idx in range(2):
                    ef_str   = f'_{temp}K_ef' if ef_idx == 1 else f'_{temp}K'
                    subfolder = f'Na_EMI_{anion}_gra{ef_str}'
                    inner    = f'No.{no:02d}{ef_str}'
                    # Windows path
                    win_folder = f'{RAW_WIN_BASE}\\{system}\\{subfolder}\\{inner}'

                    for snap_idx in range(5):
                        seg   = SNAP_SEG[snap_idx]
                        t_ns  = SNAP_NS[snap_idx]
                        p_idx = p_base + comp_idx*50 + temp_idx*10 + ef_idx*5 + snap_idx

                        rows.append({
                            'sim_id':     sim_id,
                            'il_type':    il_type,
                            'comp_idx':   comp_idx,
                            'temp':       temp,
                            'efield':     ef_idx,
                            'snap_idx':   snap_idx,
                            'snap_ns':    t_ns,
                            'p_idx':      p_idx,
                            'seg':        seg,
                            'win_folder': win_folder,
                        })
                    sim_id += 1
    return rows

# ============================================================
# PowerShellバッチ読み込み
# ============================================================
def run_ps_script(ps_script_path, timeout=900):
    """WSLのps1ファイルをWindowsのPowerShellで実行"""
    # Translate WSL path → UNC path so Windows PowerShell can find the script.
    home = str(Path.home())
    distro = os.environ.get('WSL_DISTRO_NAME', 'Ubuntu')
    unc_path = ps_script_path.replace(home, fr'\\wsl.localhost\{distro}{home}')
    result = subprocess.run(
        ['powershell.exe', '-ExecutionPolicy', 'Bypass', '-File', unc_path],
        capture_output=True, text=True, timeout=timeout
    )
    return result.stdout, result.stderr

def build_and_run_batch(sim_list):
    """全RDF・z密度ファイルを一括読み込みするPowerShellスクリプトを生成・実行"""
    ps_path = os.path.join(CACHE_DIR, 'read_all.ps1')

    lines = [
        '$ErrorActionPreference = "SilentlyContinue"',
        '',
    ]
    for s in sim_list:
        rdf_path = f'{s["win_folder"]}\\rdf.{s["seg"]:02d}.out'
        zdensity_path = f'{s["win_folder"]}\\Na_density_number_{s["temp"]}K_z.{s["seg"]:02d}.dat'

        lines.append(f'Write-Output "==RDF=={s["p_idx"]}"')
        lines.append(f'Get-Content -Path "{rdf_path}" -Tail {N_RDF} -ErrorAction SilentlyContinue')
        lines.append(f'Write-Output "==ZDENSITY=={s["p_idx"]}"')
        lines.append(f'Get-Content -Path "{zdensity_path}" -Tail {N_ZDENSITY} -ErrorAction SilentlyContinue')

    with open(ps_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"PowerShellスクリプト実行中... (約3-5分)")
    stdout, stderr = run_ps_script(ps_path, timeout=600)
    if stderr.strip():
        print(f"警告: {stderr[:200]}")
    return stdout

def parse_batch_output(stdout, sim_list):
    """バッチ出力をp_idx→ベクトルの辞書に変換"""
    rdf_dict = {}
    zdensity_dict = {}

    current_mode = None
    current_pidx = None
    buffer = []

    for line in stdout.splitlines():
        if line.startswith('==RDF=='):
            # 前のバッファを保存
            if current_mode == 'ZDENSITY' and current_pidx is not None and buffer:
                try:
                    data = np.loadtxt(io.StringIO('\n'.join(buffer)))
                    zdensity_dict[current_pidx] = data[:, 3] if data.ndim==2 and data.shape[1]>=4 else np.full(N_ZDENSITY, np.nan)
                except:
                    zdensity_dict[current_pidx] = np.full(N_ZDENSITY, np.nan)
            current_mode = 'RDF'
            current_pidx = int(line.split('==')[2])
            buffer = []
        elif line.startswith('==ZDENSITY=='):
            # RDFバッファを保存
            if current_mode == 'RDF' and current_pidx is not None and buffer:
                try:
                    data = np.loadtxt(io.StringIO('\n'.join(buffer)))
                    rdf_dict[current_pidx] = data[:, 2] if data.ndim==2 and data.shape[1]>=3 else np.full(N_RDF, np.nan)
                except:
                    rdf_dict[current_pidx] = np.full(N_RDF, np.nan)
            current_mode = 'ZDENSITY'
            current_pidx = int(line.split('==')[2])
            buffer = []
        else:
            if current_mode is not None:
                buffer.append(line)

    # 最後のバッファ
    if current_mode == 'ZDENSITY' and current_pidx is not None and buffer:
        try:
            data = np.loadtxt(io.StringIO('\n'.join(buffer)))
            zdensity_dict[current_pidx] = data[:, 3] if data.ndim==2 and data.shape[1]>=4 else np.full(N_ZDENSITY, np.nan)
        except:
            zdensity_dict[current_pidx] = np.full(N_ZDENSITY, np.nan)

    return rdf_dict, zdensity_dict

def read_pd_vectors_batch():
    """pd_vectorを全500ファイル一括コピー→読み込み"""
    pd_cache = os.path.join(CACHE_DIR, 'pd_vectors.npy')
    if os.path.exists(pd_cache):
        print("pd_vectorキャッシュを使用")
        return np.load(pd_cache)

    print(f"pd_vectorを {RAW_WIN_BASE} からコピー中...")
    pd_local = os.path.join(CACHE_DIR, 'pd_vector')
    os.makedirs(pd_local, exist_ok=True)
    distro = os.environ.get('WSL_DISTRO_NAME', 'Ubuntu')
    unc_dst = pd_local.replace(str(Path.home()), fr'\\wsl.localhost\{distro}{str(Path.home())}')

    ps_cmd = f'Copy-Item -Path "{RAW_WIN_BASE}\\pd_vector\\*" -Destination "{unc_dst}"'
    subprocess.run(['powershell.exe', '-Command', ps_cmd], timeout=120)

    pd_vecs = np.full((500, N_PD), np.nan)
    for i in range(500):
        fpath = os.path.join(pd_local, f'p{i:03d}.dat')
        if os.path.exists(fpath):
            try:
                pd_vecs[i] = np.loadtxt(fpath)
            except:
                pass
    np.save(pd_cache, pd_vecs)
    print(f"pd_vector保存完了: {pd_cache}")
    return pd_vecs

# ============================================================
# データキャッシュ管理
# ============================================================
def load_or_build_cache():
    rdf_cache  = os.path.join(CACHE_DIR, 'rdf_matrix.npy')
    zdensity_cache = os.path.join(CACHE_DIR, 'zdensity_matrix.npy')

    if os.path.exists(rdf_cache) and os.path.exists(zdensity_cache):
        print("RDF・z密度キャッシュを使用")
        rdf_matrix = np.load(rdf_cache)
        zdensity_matrix = np.load(zdensity_cache)
    else:
        sim_list = build_sim_list()
        stdout = build_and_run_batch(sim_list)
        rdf_dict, zdensity_dict = parse_batch_output(stdout, sim_list)

        # p_idxの順番に並べる
        rdf_matrix = np.full((500, N_RDF), np.nan)
        zdensity_matrix = np.full((500, N_ZDENSITY), np.nan)
        for s in sim_list:
            pid = s['p_idx']
            if pid in rdf_dict:
                rdf_matrix[pid] = rdf_dict[pid]
            if pid in zdensity_dict:
                zdensity_matrix[pid] = zdensity_dict[pid]

        np.save(rdf_cache, rdf_matrix)
        np.save(zdensity_cache, zdensity_matrix)
        print(f"RDF形状: {rdf_matrix.shape}, NaN: {np.isnan(rdf_matrix).sum()}")
        print(f"z密度形状: {zdensity_matrix.shape}, NaN: {np.isnan(zdensity_matrix).sum()}")

    return rdf_matrix, zdensity_matrix

# ============================================================
# 特徴量エンジニアリング
# ============================================================
def make_pca_features(X_raw, n_components=20, name=''):
    """VarianceThreshold → StandardScaler → PCA"""
    vt = VarianceThreshold(threshold=1e-10)
    X_vt = vt.fit_transform(X_raw)
    sc = StandardScaler()
    X_sc = sc.fit_transform(X_vt)
    n_comp = min(n_components, X_sc.shape[1], X_sc.shape[0]-1)
    pca = PCA(n_components=n_comp, random_state=42)
    X_pca = pca.fit_transform(X_sc)
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    print(f"  [{name}] {X_raw.shape[1]}→{X_vt.shape[1]}→PCA{n_comp}成分, 累積寄与率={cum_var[-1]:.3f}")
    return X_pca, pca, sc, vt

def make_phys_features(df):
    """物理パラメータ特徴量 (5次元)"""
    il_enc   = (df['il_type'] == 'TFSI').astype(float).values.reshape(-1, 1)
    temp_n   = (df['temp'].values / 450.0).reshape(-1, 1)
    comp_n   = (df['comp_idx'].values / 4.0).reshape(-1, 1)
    ef_feat  = df['efield'].values.reshape(-1, 1)
    snap_n   = (df['snap_ns'].values / 100.0).reshape(-1, 1)
    return np.hstack([il_enc, temp_n, comp_n, ef_feat, snap_n])

# ============================================================
# MLモデル評価
# ============================================================
def eval_model(X_tr, y_tr, X_te, y_te, g_tr, name, n_estimators=500, max_depth=5):
    model = ExtraTreesRegressor(
        n_estimators=n_estimators, max_depth=max_depth,
        max_features=0.5, min_samples_leaf=2, random_state=42, n_jobs=-1
    )
    gkf = GroupKFold(n_splits=5)
    cv_scores = cross_val_score(model, X_tr, y_tr, groups=g_tr,
                                cv=gkf, scoring='r2')
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    r2_te = r2_score(y_te, y_pred)
    r2_tr = r2_score(y_tr, model.predict(X_tr))
    rmse_log = np.sqrt(mean_squared_error(y_te, y_pred))
    # 実スケールRMSE
    rmse_real = np.sqrt(mean_squared_error(np.expm1(y_te), np.expm1(y_pred)))
    print(f"[{name}]")
    print(f"  Test R²={r2_te:.4f}, Train R²={r2_tr:.4f}, RMSE(log)={rmse_log:.4f}")
    print(f"  RMSE(実スケール)={rmse_real:.2f} Å²/ns, CV R²={cv_scores.mean():.4f}±{cv_scores.std():.4f}")
    return {'name': name, 'R2_test': r2_te, 'R2_train': r2_tr,
            'RMSE_log': rmse_log, 'RMSE_real': rmse_real,
            'CV_R2': cv_scores.mean(), 'CV_std': cv_scores.std()}

# ============================================================
# 相関解析
# ============================================================
def compute_feature_correlation(X1, X2, names1, names2, top_n=10):
    """X1とX2の各特徴量間のPearson相関係数を計算"""
    corr_mat = np.zeros((X1.shape[1], X2.shape[1]))
    for i in range(X1.shape[1]):
        for j in range(X2.shape[1]):
            mask = ~(np.isnan(X1[:, i]) | np.isnan(X2[:, j]))
            if mask.sum() > 10:
                r, _ = pearsonr(X1[mask, i], X2[mask, j])
                corr_mat[i, j] = r
    return corr_mat

def zdensity_vs_D_correlation(zdensity_matrix, y_raw, sim_list):
    """z密度プロファイル各ビンとD_cumulのPearson相関係数 (グラフェン相関係数)"""
    N = len(sim_list)
    corr_arr = np.zeros(N_ZDENSITY)
    pval_arr = np.ones(N_ZDENSITY)
    z_coords = np.linspace(0.01, 0.99, N_ZDENSITY)

    # p_idx順に並べてあるので、sim_listのp_idxに合わせる
    pidxs = [s['p_idx'] for s in sim_list]
    Z = zdensity_matrix[pidxs]  # (500, 50)
    y = y_raw  # (500,)

    for j in range(N_ZDENSITY):
        z_col = Z[:, j]
        mask = ~np.isnan(z_col) & ~np.isnan(y)
        if mask.sum() > 10:
            r, p = pearsonr(z_col[mask], y[mask])
            corr_arr[j] = r
            pval_arr[j] = p

    return z_coords, corr_arr, pval_arr

def rdf_vs_D_correlation(rdf_matrix, y_raw, sim_list):
    """RDF各ビンとD_cumulのPearson相関係数"""
    pidxs = [s['p_idx'] for s in sim_list]
    R = rdf_matrix[pidxs]
    y = y_raw
    r_vals = np.arange(1, N_RDF+1) * 0.02 - 0.01  # Å

    corr_arr = np.zeros(N_RDF)
    pval_arr = np.ones(N_RDF)
    for j in range(N_RDF):
        col = R[:, j]
        mask = ~np.isnan(col) & ~np.isnan(y)
        if mask.sum() > 10:
            r, p = pearsonr(col[mask], y[mask])
            corr_arr[j] = r
            pval_arr[j] = p

    return r_vals, corr_arr, pval_arr

# ============================================================
# メイン解析
# ============================================================
def main():
    print("=" * 70)
    print("RDF・z密度プロファイル vs pd_vector 比較解析")
    print("=" * 70)

    # 1. D_cumulデータを読み込み
    dc = pd.read_csv(DATA_DIR / 'D_cumulative.csv')
    print(f"\nD_cumulative: {len(dc)}サンプル, D範囲: {dc['D_cumul'].min():.2f} ~ {dc['D_cumul'].max():.2f}")

    # 2. 構造データのキャッシュ読み込み
    rdf_matrix, zdensity_matrix = load_or_build_cache()

    # 3. pd_vectorを読み込み
    pd_vecs = read_pd_vectors_batch()
    print(f"\npd_vector形状: {pd_vecs.shape}")

    # 4. シミュレーションリスト構築 (p_idxの並び順)
    sim_list = build_sim_list()
    pidxs = np.array([s['p_idx'] for s in sim_list])
    groups = np.array([s['sim_id'] for s in sim_list])

    # D_cumulをp_idx順に並べる
    dc_indexed = dc.set_index('p_idx')
    y_raw = np.array([dc_indexed.loc[p, 'D_cumul'] if p in dc_indexed.index else np.nan
                      for p in pidxs])
    y_log = np.log1p(y_raw)

    # NaN除去
    valid = ~np.isnan(y_raw)
    print(f"有効サンプル: {valid.sum()}/500")
    sim_list_valid = [s for s, v in zip(sim_list, valid) if v]

    # DataFrameも作成 (物理パラメータ用)
    df_sim = pd.DataFrame(sim_list)[valid.tolist()]

    # 5. 特徴量を準備
    print("\n=== 特徴量準備 ===")
    X_pd_raw  = pd_vecs[pidxs[valid]]
    X_rdf_raw = rdf_matrix[pidxs[valid]]
    X_zd_raw  = zdensity_matrix[pidxs[valid]]
    y = y_log[valid]
    g = groups[valid]

    # NaN補完 (まれにNaNがある場合)
    X_rdf_raw = np.nan_to_num(X_rdf_raw, nan=0.0)
    X_zd_raw  = np.nan_to_num(X_zd_raw,  nan=0.0)

    # PCA特徴量
    X_pd_pca,  *_  = make_pca_features(X_pd_raw,  20, 'pd_vector')
    X_rdf_pca, *_  = make_pca_features(X_rdf_raw, 20, 'RDF')
    X_zd_pca,  *_  = make_pca_features(X_zd_raw,  20, 'z-density')

    # 物理パラメータ
    X_phys = make_phys_features(df_sim)
    scaler_phys = StandardScaler()
    X_phys_sc = scaler_phys.fit_transform(X_phys)

    # 6. 訓練/テスト分割 (シミュレーション単位)
    unique_sims = np.unique(g)
    train_sims, test_sims = train_test_split(unique_sims, test_size=0.3, random_state=42)
    train_mask = np.isin(g, train_sims)
    test_mask  = np.isin(g, test_sims)

    def split(X):
        return X[train_mask], X[test_mask]

    y_tr, y_te = y[train_mask], y[test_mask]
    g_tr = g[train_mask]

    print(f"\nTrain: {train_mask.sum()} (シミュ: {len(train_sims)})")
    print(f"Test:  {test_mask.sum()} (シミュ: {len(test_sims)})")

    # 7. MLモデル比較
    print("\n" + "=" * 70)
    print("=== MLモデル比較 ===")
    print("=" * 70)
    results = []

    # RDF only
    X_rdf_tr, X_rdf_te = split(X_rdf_pca)
    results.append(eval_model(X_rdf_tr, y_tr, X_rdf_te, y_te, g_tr, 'RDF_only'))

    # z-density only
    X_zd_tr, X_zd_te = split(X_zd_pca)
    results.append(eval_model(X_zd_tr, y_tr, X_zd_te, y_te, g_tr, 'z-density_only'))

    # RDF + z-density
    X_rz = np.hstack([X_rdf_pca, X_zd_pca])
    X_rz_tr, X_rz_te = split(X_rz)
    results.append(eval_model(X_rz_tr, y_tr, X_rz_te, y_te, g_tr, 'RDF+zdensity'))

    # pd_vector only (参照)
    X_pd_tr, X_pd_te = split(X_pd_pca)
    results.append(eval_model(X_pd_tr, y_tr, X_pd_te, y_te, g_tr, 'pd_vector_only'))

    # RDF + z-density + phys
    X_rzp = np.hstack([X_rdf_pca, X_zd_pca, X_phys_sc])
    X_rzp_tr, X_rzp_te = split(X_rzp)
    results.append(eval_model(X_rzp_tr, y_tr, X_rzp_te, y_te, g_tr, 'RDF+zdensity+phys'))

    # pd_vector + phys (参照: v3 ExtraTrees)
    X_pdp = np.hstack([X_pd_pca, X_phys_sc])
    X_pdp_tr, X_pdp_te = split(X_pdp)
    results.append(eval_model(X_pdp_tr, y_tr, X_pdp_te, y_te, g_tr, 'pd+phys(v3参照)'))

    # RDF + zdensity + pd + phys (全特徴量)
    X_all = np.hstack([X_rdf_pca, X_zd_pca, X_pd_pca, X_phys_sc])
    X_all_tr, X_all_te = split(X_all)
    results.append(eval_model(X_all_tr, y_tr, X_all_te, y_te, g_tr, 'RDF+zdensity+pd+phys'))

    df_results = pd.DataFrame(results).sort_values('CV_R2', ascending=False)
    print("\n=== まとめ ===")
    print(df_results[['name', 'R2_test', 'R2_train', 'RMSE_real', 'CV_R2']].to_string(index=False))
    df_results.to_csv(OUT_DIR / 'rdf_ml_results.csv', index=False)

    # 8. グラフェン相関係数計算 (z密度 vs D_cumul)
    print("\n" + "=" * 70)
    print("=== グラフェンからみた相関係数 (Na+ z密度 vs D_cumul) ===")
    print("=" * 70)
    z_coords, z_corr, z_pval = zdensity_vs_D_correlation(zdensity_matrix, y_raw, sim_list)

    print("\nz座標(換算) | Pearson r | p値")
    print("-" * 45)
    for zc, rc, pv in zip(z_coords, z_corr, z_pval):
        if abs(rc) > 0.1:
            sig = '***' if pv < 0.001 else ('**' if pv < 0.01 else ('*' if pv < 0.05 else ''))
            print(f"  z={zc:.2f}     |  r={rc:+.3f}  | p={pv:.4f}  {sig}")

    # RDF vs D_cumul
    print("\n=== RDF各ビン vs D_cumul 相関 (r>2Å, |corr|>0.15) ===")
    r_vals, rdf_corr, rdf_pval = rdf_vs_D_correlation(rdf_matrix, y_raw, sim_list)
    for rv, rc, pv in zip(r_vals, rdf_corr, rdf_pval):
        if rv >= 2.0 and abs(rc) > 0.15 and pv < 0.05:
            print(f"  r={rv:.2f} Å: Pearson r={rc:+.3f}, p={pv:.4f}")

    # 9. pd_vector PCA vs RDF/z密度 の相関マトリクス
    print("\n=== pd_vector PC vs 構造特徴量 最大相関係数 ===")
    corr_pd_rdf = compute_feature_correlation(X_pd_pca, X_rdf_pca,
                                              [f'PC{i+1}' for i in range(X_pd_pca.shape[1])],
                                              [f'RDF_PC{i+1}' for i in range(X_rdf_pca.shape[1])])
    corr_pd_zd  = compute_feature_correlation(X_pd_pca, X_zd_pca,
                                              [f'PC{i+1}' for i in range(X_pd_pca.shape[1])],
                                              [f'ZD_PC{i+1}' for i in range(X_zd_pca.shape[1])])
    corr_rdf_zd = compute_feature_correlation(X_rdf_pca, X_zd_pca,
                                              [f'RDF_PC{i+1}' for i in range(X_rdf_pca.shape[1])],
                                              [f'ZD_PC{i+1}' for i in range(X_zd_pca.shape[1])])

    print(f"pd_vector PC vs RDF PC: 最大|r|={np.abs(corr_pd_rdf).max():.3f}")
    print(f"pd_vector PC vs z密度 PC: 最大|r|={np.abs(corr_pd_zd).max():.3f}")
    print(f"RDF PC vs z密度 PC: 最大|r|={np.abs(corr_rdf_zd).max():.3f}")

    # ============================================================
    # 10. プロット
    # ============================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # (a) ML比較棒グラフ
    ax = axes[0, 0]
    labels = df_results['name']
    cv_vals = df_results['CV_R2'].values
    te_vals = df_results['R2_test'].values
    x = np.arange(len(labels))
    ax.bar(x - 0.2, cv_vals, 0.4, label='CV R²', color='steelblue')
    ax.bar(x + 0.2, te_vals, 0.4, label='Test R²', color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('R²')
    ax.set_title('ML予測精度比較 (ExtraTrees)')
    ax.legend()
    ax.axhline(0, color='k', lw=0.5)
    ax.set_ylim(-0.1, 0.9)

    # (b) z密度プロファイル vs D相関係数
    ax = axes[0, 1]
    ax.plot(z_coords, z_corr, 'b-o', markersize=3, label='Pearson r')
    ax.fill_between(z_coords, z_corr, alpha=0.3)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('z (換算座標: グラフェン=0, 1)')
    ax.set_ylabel('Pearson r (Na+ 密度 vs D_cumul)')
    ax.set_title('グラフェンからみた相関係数\n(Na+ z密度プロファイル vs 拡散係数)')
    ax.legend()

    # (c) RDF vs D相関係数
    ax = axes[0, 2]
    mask_r = r_vals >= 1.0
    ax.plot(r_vals[mask_r], rdf_corr[mask_r], 'g-', lw=0.8)
    ax.fill_between(r_vals[mask_r], rdf_corr[mask_r], alpha=0.3, color='green')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('r (Å)')
    ax.set_ylabel('Pearson r (g(r) vs D_cumul)')
    ax.set_title('RDF (Na-all g(r)) vs 拡散係数 相関係数')

    # (d) 代表的なz密度プロファイル (温度別)
    ax = axes[1, 0]
    for temp in TEMPS:
        idxs = [i for i, s in enumerate(sim_list)
                if s['temp'] == temp and s['il_type'] == 'BF4' and s['efield'] == 0]
        if idxs:
            pidx_list = [sim_list[i]['p_idx'] for i in idxs]
            zd_mean = np.nanmean(zdensity_matrix[pidx_list], axis=0)
            ax.plot(z_coords, zd_mean, label=f'{temp}K')
    ax.set_xlabel('z (換算座標)')
    ax.set_ylabel('Na+ 数密度')
    ax.set_title('BF4 (ef=0) Na+ z密度プロファイル (温度別平均)')
    ax.legend(fontsize=8)

    # (e) 代表的なRDF (温度別)
    ax = axes[1, 1]
    for temp in TEMPS:
        idxs = [i for i, s in enumerate(sim_list)
                if s['temp'] == temp and s['il_type'] == 'BF4' and s['efield'] == 0]
        if idxs:
            pidx_list = [sim_list[i]['p_idx'] for i in idxs]
            rdf_mean = np.nanmean(rdf_matrix[pidx_list], axis=0)
            ax.plot(r_vals[r_vals >= 1.5], rdf_mean[r_vals >= 1.5], label=f'{temp}K')
    ax.set_xlabel('r (Å)')
    ax.set_ylabel('g(r) [Na-all]')
    ax.set_title('BF4 (ef=0) Na-all RDF (温度別平均)')
    ax.set_xlim(1.5, 10)
    ax.legend(fontsize=8)

    # (f) pd_vector PC1 vs RDF/z密度 相関ヒートマップ
    ax = axes[1, 2]
    n_show = min(10, corr_pd_rdf.shape[0], corr_pd_rdf.shape[1])
    combined_corr = np.vstack([corr_pd_rdf[:n_show, :n_show],
                               corr_pd_zd[:n_show, :n_show]])
    im = ax.imshow(combined_corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.axhline(n_show - 0.5, color='k', lw=1.5)
    ax.set_xlabel('RDF PC / z密度 PC')
    ax.set_ylabel('pd_vector PC (上: vs RDF, 下: vs z密度)')
    ax.set_title('pd_vector PC vs RDF/z密度 PC 相関係数')
    plt.colorbar(im, ax=ax, label='Pearson r')

    plt.tight_layout()
    fig.savefig(OUT_DIR / 'rdf_zdensity_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n図保存: {OUT_DIR / 'rdf_zdensity_analysis.png'}")
    print("=== 完了 ===")


if __name__ == '__main__':
    main()
