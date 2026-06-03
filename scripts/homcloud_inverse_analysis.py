"""
HomCloud 逆解析:
1. LAMMPSデータファイルから全原子座標+原子種を読み込み
2. アルファ複体でH2パーシステントホモロジー計算 (全原子: グラフェン+IL)
3. 重要なH2特徴量の逆解析 → 境界原子群を特定・可視化
対象: Low D (250K) vs High D (450K) の比較
"""

import numpy as np
import subprocess, io, os, sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import cKDTree
import warnings
warnings.filterwarnings('ignore')

import homcloud.interface as hc

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = str(REPO_ROOT / 'cache' / 'homcloud')
OUT_DIR   = REPO_ROOT / 'results'
# Raw LAMMPS data tree on the Windows host (only needed for regeneration).
RAW_WIN_BASE = r'D:\path\to\raw\trajectories'
os.makedirs(CACHE_DIR, exist_ok=True)

# 原子種マッピング (BF4系)
ATOM_TYPES_BF4 = {
    1:  ('Na',       'red',     80),
    2:  ('N(EMI)',   'blue',    30),
    3:  ('C(EMI)',   'cyan',    20),
    4:  ('N(EMI)',   'blue',    30),
    5:  ('C(EMI)',   'cyan',    20),
    6:  ('C(EMI)',   'cyan',    20),
    7:  ('H(EMI)',   'lightblue', 10),
    8:  ('C(EMI)',   'cyan',    20),
    9:  ('H(EMI)',   'lightblue', 10),
    10: ('C(EMI)',   'cyan',    20),
    11: ('H(EMI)',   'lightblue', 10),
    12: ('H(EMI)',   'lightblue', 10),
    13: ('H(EMI)',   'lightblue', 10),
    14: ('B(BF4)',   'green',   40),
    15: ('F(BF4)',   'lime',    30),
    16: ('C(gra)',   'black',   15),
}

# ============================================================
# LAMMPSデータファイルの読み込み
# ============================================================
def copy_lammps_data(win_folder, temp, snap_seg, local_name):
    """D:からLAMMPSデータファイルをローカルにコピー"""
    local_path = os.path.join(CACHE_DIR, f'{local_name}.data')
    if os.path.exists(local_path):
        return local_path
    win_path = f'{win_folder}\\data.eq4.{snap_seg}0000000.data'
    home = str(Path.home())
    distro = os.environ.get('WSL_DISTRO_NAME', 'Ubuntu')
    unc_dst = local_path.replace(home, fr'\\wsl.localhost\{distro}{home}')
    ps_cmd   = f'Copy-Item -Path "{win_path}" -Destination "{unc_dst}"'
    subprocess.run(['powershell.exe', '-Command', ps_cmd], timeout=120)
    return local_path

def parse_lammps_data(data_path):
    """LAMMPSデータファイルから原子座標と原子種を読み込む (full形式)
    戻り値: coords (N,3), types (N,), box (3,2)
    """
    coords = []
    types  = []
    box    = np.zeros((3, 2))
    in_atoms = False

    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if 'xlo xhi' in line:
                v = line.split(); box[0] = [float(v[0]), float(v[1])]
            elif 'ylo yhi' in line:
                v = line.split(); box[1] = [float(v[0]), float(v[1])]
            elif 'zlo zhi' in line:
                v = line.split(); box[2] = [float(v[0]), float(v[1])]
            elif line.startswith('Atoms'):
                in_atoms = True; continue
            elif in_atoms and line in ('Velocities', 'Bonds', 'Angles', 'Dihedrals'):
                break
            elif in_atoms:
                parts = line.split()
                if len(parts) >= 7:
                    try:
                        # full形式: atom_id mol_id type charge x y z [ix iy iz]
                        atom_type = int(parts[2])
                        x, y, z   = float(parts[4]), float(parts[5]), float(parts[6])
                        types.append(atom_type)
                        coords.append([x, y, z])
                    except ValueError:
                        pass

    return np.array(coords), np.array(types), box

# ============================================================
# HomCloud 計算
# ============================================================
def compute_pd(coords, idiagram_path):
    """アルファ複体でH2パーシステントホモロジー計算"""
    PDList = hc.PDList.from_alpha_filtration(
        coords,
        save_to=idiagram_path,
        save_boundary_map=True
    )
    return PDList

def get_h2_pairs_sorted(PDList):
    """H2ペアを永続度でソートして返す"""
    pd2   = PDList.dth_diagram(2)
    pairs = pd2.pairs()
    # persistenceでソート (lifetimeはメソッドなのでdeath-birthで計算)
    pairs_sorted = sorted(pairs, key=lambda p: p.death - p.birth, reverse=True)
    return pairs_sorted

# ============================================================
# 逆解析: 境界原子の特定
# ============================================================
def inverse_analysis(pair, coords, types, atom_types_dict):
    """H2特徴量の逆解析: 境界原子の3D座標と原子種を返す"""
    vol = pair.optimal_volume()
    bp  = vol.boundary_points()  # 境界点の3D座標リスト

    if not bp:
        return None, None

    bp_arr = np.array(bp)

    # KDTreeで最近傍原子を同定
    tree     = cKDTree(coords)
    dists, idxs = tree.query(bp_arr, k=1)

    boundary_atoms = {
        'coords':     coords[idxs],
        'types':      types[idxs],
        'names':      [atom_types_dict.get(t, (f'type{t}', 'gray', 15))[0] for t in types[idxs]],
        'colors':     [atom_types_dict.get(t, (f'type{t}', 'gray', 15))[1] for t in types[idxs]],
        'nn_dists':   dists,
        'bp_original': bp_arr,
    }
    return boundary_atoms, vol

# ============================================================
# 可視化
# ============================================================
def plot_system_with_feature(coords, types, boundary_atoms, pair_info,
                              atom_types_dict, title, ax3d, z_range=None):
    """3D散布図: 系全体 + H2特徴量の境界原子を強調"""
    # 表示する原子を絞る (z範囲指定)
    if z_range is not None:
        mask = (coords[:, 2] >= z_range[0]) & (coords[:, 2] <= z_range[1])
    else:
        mask = np.ones(len(coords), dtype=bool)

    # 背景: 全原子 (小さく薄く)
    for atype, (aname, acolor, asize) in atom_types_dict.items():
        tmask = (types == atype) & mask
        if tmask.sum() > 0:
            alpha = 0.05 if atype not in (1, 14, 15, 16) else 0.15
            ax3d.scatter(coords[tmask, 0], coords[tmask, 1], coords[tmask, 2],
                        c=acolor, s=asize*0.3, alpha=alpha, linewidths=0)

    # 境界原子: 大きく強調
    if boundary_atoms is not None:
        bc    = boundary_atoms['coords']
        btypes = boundary_atoms['types']
        for atype in np.unique(btypes):
            bm    = btypes == atype
            aname, acolor, asize = atom_types_dict.get(atype, (f't{atype}', 'gray', 15))
            ax3d.scatter(bc[bm, 0], bc[bm, 1], bc[bm, 2],
                        c=acolor, s=asize*8, alpha=0.9, edgecolors='black',
                        linewidths=0.8, label=aname, zorder=5)
        # 境界原子を線でつなぐ
        from itertools import combinations
        for i, j in combinations(range(len(bc)), 2):
            d = np.linalg.norm(bc[i] - bc[j])
            if d < 5.0:  # 5Å以内の原子を接続
                ax3d.plot([bc[i,0], bc[j,0]], [bc[i,1], bc[j,1]],
                          [bc[i,2], bc[j,2]], 'k-', lw=0.5, alpha=0.4)

    ax3d.set_xlabel('x (Å)', fontsize=7)
    ax3d.set_ylabel('y (Å)', fontsize=7)
    ax3d.set_zlabel('z (Å)', fontsize=7)
    ax3d.set_title(title, fontsize=8)
    ax3d.tick_params(labelsize=6)
    handles, labels = ax3d.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    if unique:
        ax3d.legend(unique.values(), unique.keys(), fontsize=6, loc='upper right')

# ============================================================
# メイン
# ============================================================
def analyze_one_system(label, win_folder, temp, snap_seg, system_name):
    """1シミュレーションのHomCloud逆解析を実行"""
    print(f"\n{'='*60}")
    print(f"解析: {label}")
    print(f"{'='*60}")

    # 1. LAMMPSデータファイルをコピー
    data_path    = copy_lammps_data(win_folder, temp, snap_seg, system_name)
    idiagram_path = os.path.join(CACHE_DIR, f'{system_name}.idiagram')

    if not os.path.exists(data_path):
        print(f"ERROR: データファイルが見つかりません: {data_path}")
        return None

    # 2. 原子座標を読み込み
    coords, types, box = parse_lammps_data(data_path)
    print(f"原子数: {len(coords)}")
    print(f"原子種: { {t: (types==t).sum() for t in np.unique(types)} }")
    print(f"Boxサイズ: x={box[0]}, y={box[1]}, z={box[2]}")

    # グラフェンのz座標を確認
    gra_z = coords[types == 16, 2]
    if len(gra_z) > 0:
        print(f"グラフェンC (type16) z範囲: {gra_z.min():.2f} ~ {gra_z.max():.2f} Å")

    Na_z = coords[types == 1, 2]
    if len(Na_z) > 0:
        print(f"Na+ (type1) z範囲: {Na_z.min():.2f} ~ {Na_z.max():.2f} Å")

    # 3. HomCloud計算
    print("HomCloudアルファ複体計算中...")
    PDList = compute_pd(coords, idiagram_path)
    pairs  = get_h2_pairs_sorted(PDList)
    print(f"H2ペア総数: {len(pairs)}")

    # 4. パーシステンスダイアグラムの概要
    births  = np.array([p.birth for p in pairs])
    deaths  = np.array([p.death for p in pairs])
    persist = deaths - births

    print(f"\nH2 birth範囲: {births.min():.3f} ~ {births.max():.3f}")
    print(f"H2 death範囲: {deaths.min():.3f} ~ {deaths.max():.3f}")
    print(f"最大persistence: {persist.max():.3f}")

    # 5. 対象ビンに近いペアを選択して逆解析
    # 実際のbirth範囲でターゲットを決定
    b_min = births.min()
    b_max = births.max()
    b_q25 = np.percentile(births, 25)
    b_q75 = np.percentile(births, 75)

    targets = [
        # (ラベル, birth目標値, mode)
        # mode: 'nearest_birth', 'max_persist', 'min_birth', 'nearest_birth_high_persist'
        (f'最小birth (b≈{b_min:.2f}Å)', b_min, 'min_birth'),
        (f'中間birth界面特徴 (b≈{b_q25:.2f}Å)', b_q25, 'nearest_birth_high_persist'),
        ('最大永続性', None, 'max_persist'),
    ]

    results = []
    for target_label, b_target, mode in targets:
        if mode == 'max_persist':
            idx = np.argmax(persist)
            sel = pairs[idx]
        elif mode == 'min_birth':
            idx = np.argmin(births)
            sel = pairs[idx]
        elif mode == 'nearest_birth':
            idx = np.argmin(np.abs(births - b_target))
            sel = pairs[idx]
        elif mode == 'nearest_birth_high_persist':
            # b_target付近 (±30%) かつ persistenceが高いもの
            in_range = np.abs(births - b_target) < b_target * 0.5
            if in_range.any():
                sub_persist = np.where(in_range, persist, -1)
                idx = np.argmax(sub_persist)
            else:
                idx = np.argmin(np.abs(births - b_target))
            sel = pairs[idx]
        else:
            idx = np.argmin(np.abs(births - b_target))
            sel = pairs[idx]

        print(f"\n[{target_label}]")
        print(f"  選択ペア: birth={sel.birth:.4f}, death={sel.death:.4f}, "
              f"persistence={sel.death-sel.birth:.4f}")

        try:
            boundary, vol = inverse_analysis(sel, coords, types, ATOM_TYPES_BF4)
            if boundary is not None:
                type_counts = {ATOM_TYPES_BF4.get(t, (f't{t}',))[0]:
                               (boundary['types']==t).sum()
                               for t in np.unique(boundary['types'])}
                print(f"  境界原子種: {type_counts}")
                print(f"  境界原子数: {len(boundary['coords'])}")
                z_boundary = boundary['coords'][:, 2]
                print(f"  境界原子z範囲: {z_boundary.min():.2f} ~ {z_boundary.max():.2f} Å")
                results.append({
                    'label':    target_label,
                    'pair':     sel,
                    'boundary': boundary,
                    'vol':      vol,
                })
        except Exception as e:
            print(f"  逆解析エラー: {e}")
            results.append({'label': target_label, 'pair': sel, 'boundary': None})

    return {
        'label':   label,
        'coords':  coords,
        'types':   types,
        'box':     box,
        'pairs':   pairs,
        'births':  births,
        'deaths':  deaths,
        'persist': persist,
        'results': results,
    }


def main():
    print("=" * 70)
    print("HomCloud 逆解析: グラフェン-イオン液体界面トポロジー")
    print("=" * 70)

    # 解析対象: BF4, No.01, 100ns スナップショット
    # Low D: 250K, High D: 450K
    cases = [
        {
            'label': 'BF4_gra No.01 250K (Low D)',
            'win_folder': fr'{RAW_WIN_BASE}\BF4_gra\Na_EMI_BF4_gra_250K\No.01_250K',
            'temp': 250, 'snap_seg': 10,
            'system_name': 'BF4_No01_250K_100ns',
        },
        {
            'label': 'BF4_gra No.01 450K (High D)',
            'win_folder': fr'{RAW_WIN_BASE}\BF4_gra\Na_EMI_BF4_gra_450K\No.01_450K',
            'temp': 450, 'snap_seg': 10,
            'system_name': 'BF4_No01_450K_100ns',
        },
    ]

    all_data = []
    for case in cases:
        data = analyze_one_system(
            case['label'], case['win_folder'],
            case['temp'], case['snap_seg'], case['system_name']
        )
        if data:
            all_data.append(data)

    # ============================================================
    # 可視化
    # ============================================================
    print("\n図を作成中...")

    # 図1: パーシステンスダイアグラム比較 + 高相関ビンのマーク
    fig1, axes = plt.subplots(1, len(all_data), figsize=(7*len(all_data), 6))
    if len(all_data) == 1:
        axes = [axes]

    for ax, data in zip(axes, all_data):
        b = data['births']; d = data['deaths']
        p = data['persist']
        sc = ax.scatter(b, d, c=np.log1p(p), cmap='viridis', s=8, alpha=0.6)
        plt.colorbar(sc, ax=ax, label='log(persistence)')
        ax.plot([0, 20], [0, 20], 'k--', lw=0.5)

        # 高相関ビンのマーク
        ax.axvline(x=0.47, color='red', lw=1.0, ls='--', label='birth=0.47Å (高相関)')
        ax.axvline(x=1.41, color='orange', lw=1.0, ls='--', label='birth=1.41Å (C-C)')

        # 逆解析した特徴量をマーク
        for res in data['results']:
            pair = res['pair']
            ax.scatter([pair.birth], [pair.death], s=150,
                      c='red' if '0.47' in res['label'] else ('orange' if 'C-C' in res['label'] else 'purple'),
                      marker='*', zorder=5, edgecolors='black', linewidths=0.5)

        ax.set_xlim(0, 8); ax.set_ylim(0, 20)
        ax.set_xlabel('birth (Å)')
        ax.set_ylabel('death (Å)')
        ax.set_title(f'H2 Persistence Diagram\n{data["label"]}', fontsize=9)
        ax.legend(fontsize=7)

    fig1.tight_layout()
    fig1.savefig(OUT_DIR / 'homcloud_pd_comparison.png', dpi=150, bbox_inches='tight')
    print("  homcloud_pd_comparison.png 保存")

    # 図2: 逆解析結果の3D可視化 (系ごと × 特徴量ごと)
    n_targets = max(len(d['results']) for d in all_data)
    fig2 = plt.figure(figsize=(7*len(all_data), 6*n_targets))

    plot_idx = 1
    for data in all_data:
        coords = data['coords']; types = data['types']
        # グラフェンz付近 ± 20Å を表示
        gra_z_mean = coords[types == 16, 2].mean() if (types == 16).any() else 0.0
        z_range = (gra_z_mean - 20, gra_z_mean + 30)

        for res in data['results']:
            ax3d = fig2.add_subplot(n_targets, len(all_data), plot_idx,
                                    projection='3d')
            title = f"{data['label']}\n{res['label']}\nbirth={res['pair'].birth:.3f}, death={res['pair'].death:.3f}"
            plot_system_with_feature(
                coords, types, res['boundary'], res['pair'],
                ATOM_TYPES_BF4, title, ax3d, z_range=z_range
            )
            plot_idx += 1

    fig2.tight_layout()
    fig2.savefig(OUT_DIR / 'homcloud_inverse_3d.png', dpi=120, bbox_inches='tight')
    print("  homcloud_inverse_3d.png 保存")

    # 図3: 境界原子のxy平面投影 (グラフェン視点)
    if len(all_data) > 0:
        fig3, axes = plt.subplots(len(all_data), n_targets,
                                   figsize=(5*n_targets, 5*len(all_data)))
        if len(all_data) == 1:
            axes = [axes]

        for row, data in enumerate(all_data):
            coords = data['coords']; types = data['types']
            # グラフェンのxy平面
            gra_mask = types == 16
            row_axes = axes[row] if n_targets > 1 else [axes[row]]

            for col, res in enumerate(data['results']):
                ax = row_axes[col] if hasattr(row_axes, '__len__') else row_axes

                # グラフェン原子を背景に
                if gra_mask.any():
                    ax.scatter(coords[gra_mask, 0], coords[gra_mask, 1],
                              c='lightgray', s=8, alpha=0.5, label='C(gra)')

                # Na+
                na_mask = types == 1
                if na_mask.any():
                    ax.scatter(coords[na_mask, 0], coords[na_mask, 1],
                              c='red', s=40, alpha=0.7, label='Na+', zorder=3)

                # 境界原子
                if res['boundary'] is not None:
                    bc = res['boundary']['coords']
                    for atype in np.unique(res['boundary']['types']):
                        bm = res['boundary']['types'] == atype
                        aname, acolor, asize = ATOM_TYPES_BF4.get(atype, (f't{atype}', 'gray', 15))
                        ax.scatter(bc[bm, 0], bc[bm, 1], c=acolor, s=asize*5,
                                  alpha=1.0, edgecolors='black', linewidths=1.0,
                                  label=f'{aname} (boundary)', zorder=5)
                    # 境界原子を線でつなぐ
                    from itertools import combinations
                    for i, j in combinations(range(len(bc)), 2):
                        d_ij = np.linalg.norm(bc[i,:2] - bc[j,:2])
                        if d_ij < 5.0:
                            ax.plot([bc[i,0], bc[j,0]], [bc[i,1], bc[j,1]],
                                   'k-', lw=1.0, alpha=0.5, zorder=4)

                ax.set_xlabel('x (Å)'); ax.set_ylabel('y (Å)')
                ax.set_title(f'{data["label"][:20]}\n{res["label"]}\n'
                            f'birth={res["pair"].birth:.3f}Å', fontsize=8)
                ax.set_aspect('equal')
                handles, labels = ax.get_legend_handles_labels()
                unique = dict(zip(labels, handles))
                ax.legend(unique.values(), unique.keys(), fontsize=6)

        fig3.tight_layout()
        fig3.savefig(OUT_DIR / 'homcloud_xy_projection.png', dpi=150, bbox_inches='tight')
        print("  homcloud_xy_projection.png 保存")

    # ============================================================
    # 数値サマリー
    # ============================================================
    print("\n" + "=" * 70)
    print("=== 逆解析サマリー ===")
    print("=" * 70)
    for data in all_data:
        print(f"\n{data['label']}")
        print(f"  H2ペア総数: {len(data['pairs'])}")
        print(f"  最大persistence: {data['persist'].max():.3f} Å")
        for res in data['results']:
            print(f"\n  [{res['label']}]")
            print(f"    birth={res['pair'].birth:.4f} Å, death={res['pair'].death:.4f} Å")
            if res['boundary'] is not None:
                tc = {}
                for t in res['boundary']['types']:
                    name = ATOM_TYPES_BF4.get(t, (f't{t}',))[0]
                    tc[name] = tc.get(name, 0) + 1
                print(f"    境界原子組成: {tc}")
                z_b = res['boundary']['coords'][:, 2]
                gra_z = data['coords'][data['types']==16, 2]
                if len(gra_z) > 0:
                    dist_to_gra = np.abs(z_b - gra_z.mean())
                    print(f"    グラフェンからの距離: {dist_to_gra.min():.2f} ~ {dist_to_gra.max():.2f} Å")

    print("\n=== 完了 ===")

if __name__ == '__main__':
    main()
