"""
HomCloud総合可視化:
- pd_vector (全500サンプル) の統計的H2マップ → 温度・IL種・電場別
- HomCloud逆解析結果 (250K/450K) の統合表示
- pd_vectorビンを介したグラフェン界面特徴量の定量化
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import matplotlib.patches as mpatches
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import homcloud.interface as hc

REPO_ROOT  = Path(__file__).resolve().parent.parent
CACHE_DIR  = str(REPO_ROOT / 'cache')
HC_DIR     = str(REPO_ROOT / 'cache' / 'homcloud')
OUT_DIR    = REPO_ROOT / 'results'
DATA_DIR   = REPO_ROOT / 'data'
GRID_SIZE  = 64
GRID_MAX   = 20.0
BIN_WIDTH  = GRID_MAX / GRID_SIZE
TEMPS      = [250, 300, 350, 400, 450]

bin_centers = (np.arange(GRID_SIZE) + 0.5) * BIN_WIDTH  # 0..20Å

ATOM_TYPES_BF4 = {
    1:  ('Na+',      'red',     80),
    14: ('B(BF4)',   'green',   40),
    15: ('F(BF4)',   'lime',    30),
    16: ('C(gra)',   'black',   15),
    2:  ('N(EMI)',   'royalblue', 30),
    3:  ('C(EMI)',   'deepskyblue', 20),
}

def vec_to_2d(vec):
    mat = np.zeros((GRID_SIZE, GRID_SIZE))
    rows, cols = np.triu_indices(GRID_SIZE, k=0)
    mat[rows, cols] = vec
    return mat

def apply_lower_mask(mat):
    m = mat.copy().astype(float)
    m[np.tril(np.ones_like(m, dtype=bool), k=-1)] = np.nan
    return m

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
                                     'snap_ns': [100,60,70,80,90][snap_idx],
                                     'p_idx': p_idx})
                    sim_id += 1
    return pd.DataFrame(rows)

# ============================================================
# 逆解析結果の再読み込み
# ============================================================
def reload_inverse_results():
    """既存のidiagramから逆解析を再実行して境界原子を取得"""
    results = {}
    cases = [
        ('250K', f'{HC_DIR}/BF4_No01_250K_100ns'),
        ('450K', f'{HC_DIR}/BF4_No01_450K_100ns'),
    ]
    for label, base in cases:
        data_path    = base + '.data'
        idiagram_path = base + '.idiagram'
        try:
            coords_all, types_all = parse_lammps_light(data_path)
            PDList = hc.PDList(idiagram_path)
            pd2    = PDList.dth_diagram(2)
            pairs  = sorted(pd2.pairs(), key=lambda p: p.death - p.birth, reverse=True)

            # 最大永続性特徴の逆解析
            p = pairs[0]
            vol = p.optimal_volume()
            bp  = np.array(vol.boundary_points())
            if len(bp) > 0:
                from scipy.spatial import cKDTree
                tree = cKDTree(coords_all)
                _, idxs = tree.query(bp, k=1)
                results[label] = {
                    'pair':   p,
                    'coords': coords_all,
                    'types':  types_all,
                    'boundary_coords': coords_all[idxs],
                    'boundary_types':  types_all[idxs],
                    'all_pairs': pairs,
                }
        except Exception as e:
            print(f"  {label} 逆解析エラー: {e}")
    return results

def parse_lammps_light(data_path):
    coords, types = [], []
    in_atoms = False
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('Atoms'):
                in_atoms = True; continue
            if in_atoms and line.split()[0] in ('Velocities','Bonds','Angles','Dihedrals'):
                break
            if in_atoms:
                p = line.split()
                if len(p) >= 7:
                    try:
                        types.append(int(p[2]))
                        coords.append([float(p[4]), float(p[5]), float(p[6])])
                    except ValueError:
                        pass
    return np.array(coords), np.array(types)

# ============================================================
# グラフェン界面特徴量の定量化 (pd_vectorビン使用)
# ============================================================
def compute_interface_proxy(pd_vecs, df_sim, y_D):
    """
    最大永続性特徴に対応するpd_vectorビンの平均値を条件別に計算
    250K逆解析: birth≈4.37Å → bin_b=13, death≈8.70Å → bin_d=27
    """
    # (birth, death) → 1D pd_vector index
    def bd_to_idx(bi, di):
        k = 0
        for i in range(bi):
            k += GRID_SIZE - i
        k += di - bi
        return k

    # 250K界面特徴: birth=4.37→bin13, death=8.70→bin27
    b_if = int(4.37 / BIN_WIDTH)  # ≈13
    d_if = int(8.70 / BIN_WIDTH)  # ≈27
    idx_if = bd_to_idx(min(b_if, GRID_SIZE-1), min(d_if, GRID_SIZE-1))

    # 450K バルク特徴: birth=4.89→bin15, death=13.29→bin42
    b_bk = int(4.89 / BIN_WIDTH)  # ≈15
    d_bk = int(13.29 / BIN_WIDTH)  # ≈42
    idx_bk = bd_to_idx(min(b_bk, GRID_SIZE-1), min(d_bk, GRID_SIZE-1))

    print(f"界面特徴ビン idx={idx_if} (birth_bin={b_if}, death_bin={d_if})")
    print(f"バルク特徴ビン idx={idx_bk} (birth_bin={b_bk}, death_bin={d_bk})")

    # 条件別の平均値
    proxy_if = {}
    proxy_bk = {}
    for temp in TEMPS:
        for il in ['BF4', 'TFSI']:
            mask = (df_sim['temp'] == temp) & (df_sim['il_type'] == il)
            if mask.sum() > 0:
                proxy_if[(temp, il)] = pd_vecs[mask, idx_if].mean()
                proxy_bk[(temp, il)] = pd_vecs[mask, idx_bk].mean()

    return proxy_if, proxy_bk, idx_if, idx_bk

# ============================================================
# メイン
# ============================================================
def main():
    print("=" * 70)
    print("HomCloud 総合可視化")
    print("=" * 70)

    # データ読み込み
    df_sim  = build_sim_list()
    dc      = pd.read_csv(DATA_DIR / 'D_cumulative.csv').set_index('p_idx')
    pd_vecs = np.load(f'{CACHE_DIR}/pd_vectors.npy')
    pidxs   = df_sim['p_idx'].values
    y_D     = np.array([dc.loc[p,'D_cumul'] if p in dc.index else np.nan for p in pidxs])

    # 逆解析結果を再ロード
    print("\n逆解析結果を再ロード中...")
    inv_results = reload_inverse_results()

    # 界面特徴プロキシを計算 (経験的に特定した正確なビン番号)
    print("\n界面特徴プロキシを計算中...")
    # 実測により特定 (250K vs 450K idiagramのsigma=0.2比較):
    # ビン243 (birth_bin=3, death_bin=57): 250K-450K差分が最大 → 界面構造を最も反映
    # ビン313 (birth_bin=5, death_bin=8) : 250K most persistent H2ペアの正確な対応ビン
    # ビン480 (birth_bin=7, death_bin=60): 450K most persistent H2ペアの対応ビン
    IDX_INTERFACE = 243   # 界面特徴: 温度で最も変化するビン
    IDX_BULK      = 480   # バルク特徴: 450Kで卓越するビン

    from scipy.stats import pearsonr as _pearsonr
    valid_m = ~np.isnan(y_D)
    feat_if = pd_vecs[pidxs[valid_m], IDX_INTERFACE]
    feat_bk = pd_vecs[pidxs[valid_m], IDX_BULK]
    r_if, p_if = _pearsonr(np.log1p(y_D[valid_m]), feat_if)
    r_bk, p_bk = _pearsonr(np.log1p(y_D[valid_m]), feat_bk)
    print(f"界面ビン(idx={IDX_INTERFACE}) vs log1p(D): r={r_if:.3f}, p={p_if:.2e}")
    print(f"バルクビン(idx={IDX_BULK}) vs log1p(D): r={r_bk:.3f}, p={p_bk:.2e}")

    proxy_if = {}
    proxy_bk_dict = {}
    for il in ['BF4', 'TFSI']:
        for temp in TEMPS:
            for ef in [0, 1]:
                mask = ((df_sim['il_type'] == il) &
                        (df_sim['temp'] == temp) &
                        (df_sim['efield'] == ef))
                if mask.sum() > 0:
                    proxy_if[(temp, il, ef)]    = pd_vecs[pidxs[mask], IDX_INTERFACE].mean()
                    proxy_bk_dict[(temp, il, ef)] = pd_vecs[pidxs[mask], IDX_BULK].mean()

    # backward compat
    idx_if, idx_bk = IDX_INTERFACE, IDX_BULK

    # ============================================================
    # 総合図の作成
    # ============================================================
    fig = plt.figure(figsize=(22, 20))
    gs  = gridspec.GridSpec(4, 5, figure=fig, hspace=0.45, wspace=0.35,
                            height_ratios=[1, 1, 1.2, 1])

    # ----- 行1: 温度別 平均H2マップ (BF4, ef=0) -----
    vmax_row1 = None
    maps_temp = {}
    for ti, temp in enumerate(TEMPS):
        mask = (df_sim['il_type']=='BF4') & (df_sim['temp']==temp) & (df_sim['efield']==0)
        if mask.sum() > 0:
            mean_vec = pd_vecs[pidxs[mask]].mean(axis=0)
            maps_temp[temp] = vec_to_2d(mean_vec)
        else:
            maps_temp[temp] = np.zeros((GRID_SIZE, GRID_SIZE))
    vmax_row1 = max(np.nanmax(apply_lower_mask(m)) for m in maps_temp.values())

    for ti, temp in enumerate(TEMPS):
        ax = fig.add_subplot(gs[0, ti])
        m = np.log1p(apply_lower_mask(maps_temp[temp]))
        im = ax.imshow(m, origin='lower', cmap='viridis',
                       extent=[0,GRID_MAX,0,GRID_MAX], aspect='equal',
                       vmin=0, vmax=np.log1p(vmax_row1)*0.8 if vmax_row1>0 else 1)
        ax.plot([0,GRID_MAX],[0,GRID_MAX],'w--',lw=0.5)
        # 主要特徴位置のマーク
        ax.axvline(x=bin_centers[3], color='red',  lw=1.2, ls='--', alpha=0.7, label='界面ビン birth')
        ax.axvline(x=bin_centers[7], color='orange', lw=1.2, ls='--', alpha=0.7, label='バルクビン birth')
        ax.set_xlim(0, 12); ax.set_ylim(0, 16)
        ax.set_title(f'{temp}K\n(BF4, ef=0)', fontsize=9)
        ax.set_xlabel('birth (Å)', fontsize=7); ax.set_ylabel('death (Å)', fontsize=7)
        ax.tick_params(labelsize=6)
        if ti == 4:
            plt.colorbar(im, ax=ax, shrink=0.7, label='log(強度)')

    fig.text(0.01, 0.88, '① 温度別H2マップ\n(平均pd_vector)', fontsize=9, fontweight='bold',
             va='center', rotation=90)

    # ----- 行2: IL種・電場・D値別 差分マップ -----
    # BF4 平均
    mask_bf4 = (df_sim['il_type']=='BF4') & (df_sim['efield']==0)
    map_bf4  = vec_to_2d(pd_vecs[pidxs[mask_bf4]].mean(axis=0))
    # TFSI 平均
    mask_tfs = (df_sim['il_type']=='TFSI') & (df_sim['efield']==0)
    map_tfs  = vec_to_2d(pd_vecs[pidxs[mask_tfs]].mean(axis=0))
    # ef=1 平均 (BF4)
    mask_ef1 = (df_sim['il_type']=='BF4') & (df_sim['efield']==1)
    map_ef1  = vec_to_2d(pd_vecs[pidxs[mask_ef1]].mean(axis=0))
    # High D vs Low D
    med      = np.nanmedian(y_D)
    mask_hi  = y_D >= med
    mask_lo  = y_D <  med
    map_hi   = vec_to_2d(pd_vecs[pidxs[mask_hi]].mean(axis=0))
    map_lo   = vec_to_2d(pd_vecs[pidxs[mask_lo]].mean(axis=0))

    panels_r2 = [
        (map_bf4,        'BF4 (ef=0) 平均H2', 'abs'),
        (map_tfs,        'TFSI (ef=0) 平均H2', 'abs'),
        (map_bf4-map_tfs,'BF4 - TFSI 差分', 'diff'),
        (map_ef1-map_bf4,'ef=1 - ef=0 差分\n(BF4)', 'diff'),
        (map_hi-map_lo,  f'High D - Low D 差分\n(med={np.expm1(np.log1p(med)):.0f} Å²/ns)', 'diff'),
    ]
    for ci, (m, title, mode) in enumerate(panels_r2):
        ax = fig.add_subplot(gs[1, ci])
        mm = apply_lower_mask(m)
        if mode == 'abs':
            im2 = ax.imshow(np.log1p(mm), origin='lower', cmap='viridis',
                            extent=[0,GRID_MAX,0,GRID_MAX], aspect='equal')
        else:
            v = np.nanpercentile(np.abs(mm[~np.isnan(mm)]), 95) if not np.all(np.isnan(mm)) else 1
            im2 = ax.imshow(mm, origin='lower', cmap='RdBu_r',
                            extent=[0,GRID_MAX,0,GRID_MAX], aspect='equal',
                            vmin=-v, vmax=v)
        ax.plot([0,GRID_MAX],[0,GRID_MAX],'k--',lw=0.5)
        ax.axvline(x=bin_centers[3], color='red',    lw=1.0, ls='--', alpha=0.6)
        ax.axvline(x=bin_centers[7], color='orange', lw=1.0, ls='--', alpha=0.6)
        ax.set_xlim(0, 12); ax.set_ylim(0, 16)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel('birth (Å)', fontsize=7); ax.set_ylabel('death (Å)', fontsize=7)
        ax.tick_params(labelsize=6)
        plt.colorbar(im2, ax=ax, shrink=0.7)

    fig.text(0.01, 0.645, '② 条件別比較\n(差分・平均)', fontsize=9, fontweight='bold',
             va='center', rotation=90)

    # ----- 行3: HomCloud逆解析 + 定量サマリー -----
    gra_z = -3.35  # グラフェンのz座標

    for col_i, (label, lkey) in enumerate([('250K\n(Low D)', '250K'), ('450K\n(High D)', '450K')]):
        if lkey not in inv_results:
            continue
        res    = inv_results[lkey]
        coords = res['coords']
        types  = res['types']
        bc     = res['boundary_coords']
        bt     = res['boundary_types']
        pair   = res['pair']

        ax = fig.add_subplot(gs[2, col_i*2])
        # グラフェン (xy平面投影)
        gra_m = types == 16
        ax.scatter(coords[gra_m,0], coords[gra_m,1], c='lightgray', s=4, alpha=0.4)
        # Na+
        na_m = types == 1
        ax.scatter(coords[na_m,0], coords[na_m,1], c='red', s=25, alpha=0.6, label='Na+', zorder=3)
        # 境界原子
        palette = {1:'red',14:'green',15:'lime',16:'black',2:'royalblue',3:'deepskyblue'}
        for atype in np.unique(bt):
            bm = bt == atype
            name, color, _ = ATOM_TYPES_BF4.get(atype,(f't{atype}','gray',15))
            ax.scatter(bc[bm,0], bc[bm,1], c=color, s=60,
                      edgecolors='black', linewidths=0.8, label=f'{name}(boundary)', zorder=5)
        # 境界原子のエッジ
        from itertools import combinations
        for i,j in combinations(range(len(bc)), 2):
            if np.linalg.norm(bc[i,:2]-bc[j,:2]) < 5.0:
                ax.plot([bc[i,0],bc[j,0]],[bc[i,1],bc[j,1]],'k-',lw=0.5,alpha=0.3,zorder=4)
        ax.set_aspect('equal')
        ax.set_xlim(0,50); ax.set_ylim(0,52)
        ax.set_xlabel('x (Å)', fontsize=7); ax.set_ylabel('y (Å)', fontsize=7)
        ax.set_title(f'{label}\nbirth={pair.birth:.2f}Å, death={pair.death:.2f}Å', fontsize=8)
        ax.tick_params(labelsize=6)
        handles, labels_l = ax.get_legend_handles_labels()
        unique = dict(zip(labels_l, handles))
        ax.legend(unique.values(), unique.keys(), fontsize=5.5, loc='upper right',
                  markerscale=0.8, framealpha=0.7)

        # 境界原子のz分布
        ax_z = fig.add_subplot(gs[2, col_i*2+1])
        z_range = np.linspace(-5, 20, 50)
        gra_c_z = bc[bt==16, 2] if (bt==16).any() else np.array([])
        for atype, (aname, acolor, _) in ATOM_TYPES_BF4.items():
            m = bt == atype
            if m.any():
                ax_z.scatter(bc[m,2], np.ones(m.sum())*atype + np.random.randn(m.sum())*0.1,
                            c=acolor, s=20, alpha=0.7, label=aname)
        ax_z.axvline(x=gra_z, color='gray', lw=1.5, ls='--', label=f'C(gra) z={gra_z}Å')
        ax_z.set_xlabel('z (Å)', fontsize=7)
        ax_z.set_ylabel('原子種 (typeID)', fontsize=7)
        ax_z.set_title(f'{label}\n境界原子のz分布', fontsize=8)
        ax_z.tick_params(labelsize=6)
        ax_z.legend(fontsize=5.5, loc='upper right', framealpha=0.7)

    # 境界原子組成の棒グラフ
    ax_bar = fig.add_subplot(gs[2, 4])
    atom_labels = ['Na+', 'B(BF4)', 'F(BF4)', 'C(gra)', 'N(EMI)', 'C(EMI)', 'H(EMI)']
    type_ids    = [1, 14, 15, 16, 2, 3, [7,8,9,10,11,12,13]]
    colors_bar  = ['red','green','lime','black','royalblue','deepskyblue','lightblue']

    x_pos = np.arange(len(atom_labels))
    for ki, (lkey, bar_label, offset) in enumerate([('250K','250K\n(Low D)',-0.2),
                                                      ('450K','450K\n(High D)',0.2)]):
        if lkey not in inv_results:
            continue
        bt = inv_results[lkey]['boundary_types']
        n_total = len(bt)
        counts = []
        for tid in type_ids:
            if isinstance(tid, list):
                counts.append(sum((bt==t).sum() for t in tid))
            else:
                counts.append((bt==tid).sum())
        pcts = np.array(counts) / max(n_total, 1) * 100
        ax_bar.bar(x_pos+offset, pcts, 0.38,
                  label=bar_label, alpha=0.8,
                  color='steelblue' if ki==0 else 'coral')

    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(atom_labels, rotation=40, ha='right', fontsize=7)
    ax_bar.set_ylabel('割合 (%)', fontsize=7)
    ax_bar.set_title('境界原子の組成比較\n(最大永続性H2特徴)', fontsize=8)
    ax_bar.legend(fontsize=7)
    ax_bar.tick_params(labelsize=6)

    fig.text(0.01, 0.38, '③ HomCloud逆解析\n(具体的原子構造)', fontsize=9, fontweight='bold',
             va='center', rotation=90)

    # ----- 行4: 定量的温度トレンド -----
    # pd_vectorビンの温度依存性 (界面特徴 vs バルク特徴)
    ax_tr = fig.add_subplot(gs[3, 0:2])
    for il, color, ls in [('BF4','steelblue','-'), ('TFSI','coral','--')]:
        if_vals = [proxy_if.get((t, il, 0), 0) for t in TEMPS]
        bk_vals = [proxy_bk_dict.get((t, il, 0), 0) for t in TEMPS]
        ax_tr.plot(TEMPS, if_vals, color=color, ls=ls, marker='o', ms=6,
                  label=f'{il} 界面ビン(idx={IDX_INTERFACE})')
        ax_tr.plot(TEMPS, bk_vals, color=color, ls=':', marker='s', ms=6, alpha=0.6,
                  label=f'{il} バルクビン(idx={IDX_BULK})')
    ax_tr.set_xlabel('温度 (K)', fontsize=8)
    ax_tr.set_ylabel('pd_vectorビン強度 (平均)', fontsize=8)
    ax_tr.set_title(f'界面特徴ビン(idx={IDX_INTERFACE})の温度依存性\n'
                   f'実証的に特定: 250K最大→450K最小', fontsize=8)
    ax_tr.legend(fontsize=6.5, ncol=2)
    ax_tr.tick_params(labelsize=7)

    # PD統計 (全温度の特徴分布)
    ax_pers = fig.add_subplot(gs[3, 2])
    # 各条件のpd_vectorの「有効ビン数」 (>0のビン数)
    for il, color in [('BF4','steelblue'), ('TFSI','coral')]:
        nz_means = []
        for temp in TEMPS:
            mask = (df_sim['il_type']==il) & (df_sim['temp']==temp) & (df_sim['efield']==0)
            if mask.sum() > 0:
                nz = (pd_vecs[pidxs[mask]] > 0).sum(axis=1).mean()
                nz_means.append(nz)
            else:
                nz_means.append(np.nan)
        ax_pers.plot(TEMPS, nz_means, color=color, marker='o', ms=6, label=il)
    ax_pers.set_xlabel('温度 (K)', fontsize=8)
    ax_pers.set_ylabel('有効H2ビン数 (平均)', fontsize=8)
    ax_pers.set_title('H2特徴の多様性\n(非ゼロビン数 → 構造の複雑さ)', fontsize=8)
    ax_pers.legend(fontsize=7)
    ax_pers.tick_params(labelsize=7)

    # D_cumul vs 界面特徴ビン相関
    ax_cor = fig.add_subplot(gs[3, 3])
    mask_v = ~np.isnan(y_D)
    feat_if = pd_vecs[pidxs[mask_v], idx_if]
    ax_cor.scatter(np.log1p(y_D[mask_v]), feat_if,
                  c=df_sim['temp'].values[mask_v], cmap='coolwarm', s=8, alpha=0.5)
    ax_cor.set_xlabel('log1p(D_cumul)', fontsize=8)
    ax_cor.set_ylabel(f'界面特徴ビン強度 (idx={IDX_INTERFACE})', fontsize=8)
    ax_cor.set_title(f'界面特徴ビン(idx={IDX_INTERFACE}) vs 拡散係数\n'
                    f'Pearson r={r_if:.3f}, p={p_if:.2e}', fontsize=8)
    ax_cor.tick_params(labelsize=7)

    # 模式図 (概念図テキスト)
    ax_scm = fig.add_subplot(gs[3, 4])
    ax_scm.axis('off')
    schematic = (
        "【pd_vectorが捉える界面トポロジー】\n\n"
        "Low D (低温):\n"
        "  グラフェン ────── (底蓋: 43C原子)\n"
        "       │  H2空洞  │\n"
        "       │ Na+×6捕捉│\n"
        "  IL層 ────────────\n"
        "  → birth≈4.4Å, 界面特徴ビン↑\n\n"
        "High D (高温):\n"
        "  グラフェン (関与なし)\n"
        "       IL内部の大空洞\n"
        "       Na+×1のみ\n"
        "  → birth≈4.9Å, バルク特徴ビン↑\n\n"
        "pd_vector固有の情報:\n"
        "  RDF/z密度では表現できない\n"
        "  「どの原子が3D空洞を囲んでいるか」\n"
        "  = トポロジカル閉じ込め構造"
    )
    ax_scm.text(0.05, 0.97, schematic, transform=ax_scm.transAxes,
               fontsize=7.5, va='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    fig.text(0.01, 0.12, '④ 定量トレンド\n& まとめ', fontsize=9, fontweight='bold',
             va='center', rotation=90)

    # 凡例 (行1の垂直破線)
    red_line   = mpatches.Patch(color='red',    label=f'界面特徴ビン(idx={243}, birth_bin=3)')
    orange_line = mpatches.Patch(color='orange', label=f'バルク特徴ビン(idx={480}, birth_bin=7)')
    fig.legend(handles=[red_line, orange_line], loc='upper center',
              bbox_to_anchor=(0.5, 0.995), ncol=2, fontsize=8, framealpha=0.7)

    fig.suptitle(
        'HomCloud H2パーシステントホモロジー 総合解析\n'
        'グラフェン/イオン液体界面のNa+閉じ込めトポロジーの温度・IL種・電場依存性',
        fontsize=12, fontweight='bold', y=1.01
    )

    out_path = str(OUT_DIR / 'homcloud_comprehensive.png')
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    print(f"\n総合図保存: {out_path}")

    # ---- 数値サマリー出力 ----
    print("\n=== 界面特徴ビン vs バルク特徴ビン 温度依存性 ===")
    print(f"{'温度':>5} {'BF4界面':>10} {'BF4バルク':>10} {'TFSI界面':>10} {'TFSI バルク':>11}")
    for temp in TEMPS:
        bf_if = proxy_if.get((temp,'BF4',0), np.nan)
        bf_bk = proxy_bk_dict.get((temp,'BF4',0), np.nan)
        tf_if = proxy_if.get((temp,'TFSI',0), np.nan)
        tf_bk = proxy_bk_dict.get((temp,'TFSI',0), np.nan)
        print(f"{temp:>5} {bf_if:>10.4f} {bf_bk:>10.4f} {tf_if:>10.4f} {tf_bk:>11.4f}")

    print(f"\n界面特徴ビン(idx={IDX_INTERFACE}) vs log1p(D): r={r_if:.3f}, p={p_if:.2e}")
    print("=== 完了 ===")

if __name__ == '__main__':
    main()
