import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.ndimage import (
    convolve,
    gaussian_filter,
    distance_transform_edt,
    generate_binary_structure,
    binary_erosion,
    binary_dilation
)
import json
import io
import os

st.set_page_config(page_title="OPC Simulation Tool", layout="wide")
st.title("Optical Proximity Correction (OPC) Tool")
st.markdown("**7nm FinFET Standard Cell Metal1 Layout Simulation & Verification Platform**")
st.caption("Standard Design: 1 Target Grid = 3.5 nm × 3.5 nm | OPC Mask Sub-grid: 1 Sub-pixel = 0.875 nm (3.5nm / 4, 4x Reticle: 3.5 nm E-beam Unit)")

target_unit_nm = 3.5           # 웨이퍼 설계 기준 그리드 (nm)
subgrid_ratio = 4              # 4분할 서브그리드 (1 grid = 4 sub-pixels)
mask_unit_nm = target_unit_nm / subgrid_ratio  # 0.875 nm
reticle_mag = 4.0              # 4:1 축소 노광계
reticle_feature_nm = mask_unit_nm * reticle_mag # 3.5 nm 실물 마스크 E-beam 피처
pad_grids = 3                  # 외곽 패딩

# 1. 사이드바 설정
st.sidebar.header("Simulation Settings")
layout_mode = st.sidebar.radio("Layout Input Mode", ["Preset Standard Cells", "Interactive Grid Drawing", "CSV File Upload"])
process_mode = st.sidebar.radio("Process Mode", ["Standard 7nm Process Preset", "Custom Tuning Mode"])

# 2. 레이아웃 데이터 함수
@st.cache_data
def get_inv_x1_raw():
    mask = np.zeros((54, 18), dtype=float)
    mask[0:3, 0:18] = 1.0
    mask[51:54, 0:18] = 1.0
    mask[3:21, 3:6] = 1.0
    mask[33:51, 3:6] = 1.0
    mask[18:36, 12:15] = 1.0
    return mask

@st.cache_data
def get_nand2_x4_raw():
    mask = np.zeros((54, 45), dtype=float)
    mask[0:3, 0:45] = 1.0
    mask[51:54, 0:45] = 1.0
    mask[3:12, 3:6] = 1.0
    mask[3:12, 21:24] = 1.0
    mask[3:12, 39:42] = 1.0
    mask[6:24, 12:15] = 1.0
    mask[6:24, 30:33] = 1.0
    mask[21:24, 12:33] = 1.0
    mask[24:48, 21:24] = 1.0
    mask[30:51, 3:6] = 1.0
    mask[30:51, 39:42] = 1.0
    mask[30:48, 12:15] = 1.0
    mask[30:48, 30:33] = 1.0
    return mask

# 양방향 통신 컴포넌트 생성
def build_editor_component():
    current_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
    comp_dir = os.path.join(current_dir, "pixel_grid_component")
    os.makedirs(comp_dir, exist_ok=True)
    html_file = os.path.join(comp_dir, "index.html")

    html_code = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    body {
      user-select: none;
      -webkit-user-select: none;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
      background: #ffffff;
      margin: 0;
      padding: 10px;
    }
    .toolbar {
      display: flex;
      gap: 8px;
      align-items: center;
      margin-bottom: 12px;
      flex-wrap: wrap;
    }
    .btn {
      padding: 7px 14px;
      border: 1px solid #d0d7de;
      background: #f6f8fa;
      color: #24292f;
      font-weight: 600;
      font-size: 12px;
      cursor: pointer;
      border-radius: 6px;
      transition: all 0.15s ease;
      box-shadow: 0 1px 0 rgba(27,31,36,0.04);
    }
    .btn:hover {
      background: #f3f4f6;
      border-color: #1b1f2426;
    }
    .btn.active {
      background: #0969da;
      color: #ffffff;
      border-color: #0969da;
    }
    .btn-run {
      background: #2da44e;
      color: #ffffff;
      border-color: #2da44e;
    }
    .btn-run:hover {
      background: #2c974b;
    }
    #grid-box {
      display: grid;
      gap: 1px;
      background-color: #d0d7de;
      padding: 6px;
      border: 1px solid #d0d7de;
      border-radius: 8px;
      width: fit-content;
      max-height: 480px;
      overflow: auto;
      background: #f6f8fa;
    }
    .cell {
      width: 16px;
      height: 16px;
      background-color: #ffffff;
      border-radius: 2px;
      box-sizing: border-box;
      cursor: crosshair;
      transition: background-color 0.05s ease;
    }
    .cell:hover {
      background-color: #ddf4ff;
    }
    .cell.filled {
      background-color: #0969da !important;
    }
  </style>
</head>
<body>
  <div class="toolbar">
    <button id="drawBtn" class="btn active" onclick="setMode(1)">✏️ 배선 그리기</button>
    <button id="eraseBtn" class="btn" onclick="setMode(0)">🧹 지우개</button>
    <button id="toggleBtn" class="btn" onclick="setMode(2)">🔄 토글 모드</button>
    <button class="btn" onclick="clearGrid()">🗑️ 전체 초기화</button>
    <button class="btn btn-run" onclick="submitToSimulation()">⚡ 시뮬레이션 연산 적용</button>
    <span id="stat-text" style="font-size:12px; font-weight:600; color:#57606a; margin-left:8px;"></span>
  </div>
  <div id="grid-box"></div>

  <script>
    let isDrawing = false;
    let drawMode = 1;
    let currentToggleTarget = 1;
    let rows = 20;
    let cols = 20;
    let grid = [];
    let initialized = false;

    const container = document.getElementById('grid-box');

    function sendMsg(type, data) {
      window.parent.postMessage({
        isStreamlitMessage: true,
        type: type,
        ...data
      }, "*");
    }

    function renderGridUI() {
      container.innerHTML = '';
      container.style.gridTemplateColumns = `repeat(${cols}, 16px)`;
      container.style.gridTemplateRows = `repeat(${rows}, 16px)`;
      document.getElementById('stat-text').innerText = `${cols}×${rows} 칸 (${(cols*3.5).toFixed(1)}×${(rows*3.5).toFixed(1)} nm)`;

      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const cell = document.createElement('div');
          cell.className = 'cell';
          cell.id = `c_${r}_${c}`;
          if (grid[r] && grid[r][c] === 1) {
            cell.classList.add('filled');
          }

          cell.addEventListener('mousedown', (e) => {
            e.preventDefault();
            isDrawing = true;
            if (drawMode === 2) {
              currentToggleTarget = grid[r][c] === 1 ? 0 : 1;
              fillCell(cell, r, c, currentToggleTarget);
            } else {
              fillCell(cell, r, c, drawMode);
            }
          });

          cell.addEventListener('mouseenter', () => {
            if (isDrawing) {
              fillCell(cell, r, c, drawMode === 2 ? currentToggleTarget : drawMode);
            }
          });

          container.appendChild(cell);
        }
      }
      sendMsg("streamlit:setFrameHeight", { height: Math.min(540, rows * 18 + 75) });
    }

    function fillCell(cell, r, c, val) {
      grid[r][c] = val;
      if (val === 1) {
        cell.classList.add('filled');
      } else {
        cell.classList.remove('filled');
      }
    }

    window.addEventListener('mouseup', () => {
      isDrawing = false;
    });

    function setMode(mode) {
      drawMode = mode;
      document.getElementById('drawBtn').classList.toggle('active', mode === 1);
      document.getElementById('eraseBtn').classList.toggle('active', mode === 0);
      document.getElementById('toggleBtn').classList.toggle('active', mode === 2);
    }

    function clearGrid() {
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          grid[r][c] = 0;
        }
      }
      document.querySelectorAll('.cell').forEach(el => el.classList.remove('filled'));
      sendMsg("streamlit:setComponentValue", { value: { action: "clear", matrix: grid } });
    }

    function submitToSimulation() {
      sendMsg("streamlit:setComponentValue", { value: { action: "apply", matrix: grid } });
    }

    window.addEventListener("message", (event) => {
      if (event.data.type === "streamlit:render") {
        const args = event.data.args;
        if (args) {
          rows = args.rows;
          cols = args.cols;
          if (!initialized || args.force_reset) {
            grid = JSON.parse(JSON.stringify(args.initial_matrix));
            initialized = true;
          }
          renderGridUI();
        }
      }
    });

    sendMsg("streamlit:componentReady", { apiVersion: 1 });
  </script>
</body>
</html>
"""
    with open(html_file, "w", encoding="utf-8") as f:
        f.write(html_code)

    return components.declare_component("pixel_grid_editor", path=comp_dir)

pixel_grid_component = build_editor_component()

# 3. 레이아웃 모드별 타겟 처리
if layout_mode == "Preset Standard Cells":
    preset_choice = st.sidebar.selectbox("Select Target Cell", ["INV_X1", "NAND2_X4"])
    raw_target = get_inv_x1_raw() if preset_choice == "INV_X1" else get_nand2_x4_raw()
    layout_name = preset_choice

elif layout_mode == "Interactive Grid Drawing":
    st.sidebar.subheader("Canvas Grid Size")
    grid_w = st.sidebar.number_input("Grid Width (X, columns)", min_value=5, max_value=40, value=20, step=1)
    grid_h = st.sidebar.number_input("Grid Height (Y, rows)", min_value=5, max_value=60, value=20, step=1)
    st.sidebar.caption("📏 **최대 허용 크기:** Max X: 40 cols, Max Y: 60 rows (단위: 1 grid = 3.5 nm)")

    st.subheader("Minesweeper-Style Pixel Grid Editor")
    st.markdown("💡 **격자판에서 배선을 그리고 [⚡ 시뮬레이션 연산 적용]을 누르면 연산이 시작됩니다.**")

    if "editor_version" not in st.session_state:
        st.session_state.editor_version = 0

    matrix_key = f"drawn_data_{grid_w}_{grid_h}"
    if matrix_key not in st.session_state:
        st.session_state[matrix_key] = [[0 for _ in range(grid_w)] for _ in range(grid_h)]

    if len(st.session_state[matrix_key]) != grid_h or len(st.session_state[matrix_key][0]) != grid_w:
        st.session_state[matrix_key] = [[0 for _ in range(grid_w)] for _ in range(grid_h)]

    comp_result = pixel_grid_component(
        rows=int(grid_h),
        cols=int(grid_w),
        initial_matrix=st.session_state[matrix_key],
        force_reset=(st.session_state.get("need_force_reset", False)),
        key=f"editor_v{st.session_state.editor_version}_{grid_w}_{grid_h}"
    )

    st.session_state.need_force_reset = False

    if comp_result is not None and isinstance(comp_result, dict):
        action = comp_result.get("action")
        mat = comp_result.get("matrix")
        
        last_applied_key = f"last_applied_{grid_w}_{grid_h}"
        mat_str = json.dumps(mat) if mat is not None else ""
        
        if mat is not None and mat_str != st.session_state.get(last_applied_key, ""):
            st.session_state[matrix_key] = mat
            st.session_state[last_applied_key] = mat_str
            if action == "clear":
                st.session_state.editor_version += 1
                st.session_state.need_force_reset = True
                st.session_state.current_mask_sub = None
                st.rerun()
            elif action == "apply":
                st.session_state.current_mask_sub = None
                st.rerun()

    raw_target = np.array(st.session_state[matrix_key], dtype=float)
    layout_name = f"Custom_{grid_w}x{grid_h}"

else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV (0 and 1 matrix)", type=["csv"])
    if uploaded_file is not None:
        raw_target = np.genfromtxt(uploaded_file, delimiter=",").astype(float)
        raw_target = np.nan_to_num(raw_target, nan=0.0)
    else:
        raw_target = np.zeros((30, 30), dtype=float)
        raw_target[10:20, 10:20] = 1.0
    layout_name = "Uploaded_CSV"

# 4. 정수 4배 분할 서브그리드 확장
padded_target = np.pad(raw_target, pad_grids, mode='constant', constant_values=0.0)
h_grids, w_grids = padded_target.shape
total_w_nm = w_grids * target_unit_nm
total_h_nm = h_grids * target_unit_nm

# 각 3.5nm 그리드를 4x4 서브픽셀(0.875nm)로 정수 복제
target_sub = np.repeat(np.repeat(padded_target, subgrid_ratio, axis=0), subgrid_ratio, axis=1)

extent_nm = [0, total_w_nm, total_h_nm, 0]
x0_core = pad_grids * target_unit_nm
y0_core = pad_grids * target_unit_nm
w_core_nm = raw_target.shape[1] * target_unit_nm
h_core_nm = raw_target.shape[0] * target_unit_nm

# 5. 세션 마스크 관리
if "current_mask_sub" not in st.session_state or st.session_state.current_mask_sub is None or st.session_state.get("last_layout") != layout_name:
    st.session_state.current_mask_sub = target_sub.copy()
    st.session_state.last_layout = layout_name

if st.session_state.current_mask_sub.shape != target_sub.shape:
    st.session_state.current_mask_sub = target_sub.copy()

# 6. 광학 파라미터
if process_mode == "Standard 7nm Process Preset":
    st.sidebar.markdown("✅ **표준 7nm 공정 (정수 4분할 고속 서브그리드 모드)**")
    sigma_nm = 6.5
    threshold = 0.50
else:
    st.sidebar.subheader("Optical Lithography Parameters")
    sigma_nm = st.sidebar.slider("Optical Blur Radius (sigma, nm)", 3.0, 12.0, 6.5, 0.5)
    threshold = st.sidebar.slider("Resist Threshold (I_th)", 0.20, 0.80, 0.50, 0.01)

st.sidebar.subheader("Display Options")
show_grid = st.sidebar.checkbox("Display Unit Grid (3.5 nm Design Pitch)", value=True)
show_dimension = st.sidebar.checkbox("Display CAD Dimension Arrows", value=True)
show_outline = st.sidebar.checkbox("Display Target Outline (White Dashed)", value=True)

# 0.875 nm 서브픽셀 단위 표준편차
sigma_px = sigma_nm / mask_unit_nm

# 7. 고속 DFM 실효 수율 평가 함수
def evaluate_context_aware_yield(target_mat, printed_mat):
    inter = np.logical_and(target_mat == 1, printed_mat == 1).sum()
    union = np.logical_or(target_mat == 1, printed_mat == 1).sum()
    strict_iou = (inter / union) * 100.0 if union > 0 else 0.0

    space_to_wire_nm = distance_transform_edt(target_mat == 0) * mask_unit_nm

    struct_sq = generate_binary_structure(2, 1)
    core = binary_erosion(target_mat, structure=struct_sq, iterations=int(round(2.5 / mask_unit_nm))).astype(float)

    under_mask = (target_mat == 1) & (printed_mat == 0)
    over_mask = (target_mat == 0) & (printed_mat == 1)

    crit_open = np.logical_and(under_mask, core == 1).sum()
    marginal_shrink = under_mask.sum() - crit_open

    crit_short = np.logical_and(over_mask, space_to_wire_nm <= 7.0).sum()
    allowable_serif = over_mask.sum() - crit_short

    total_target_px = float(np.sum(target_mat == 1))

    weighted_defect_density = (
        1.8 * (crit_open / total_target_px) +
        1.8 * (crit_short / total_target_px) +
        0.5 * (marginal_shrink / total_target_px) +
        0.05 * (allowable_serif / total_target_px)
    )
    dfm_yield = np.exp(-weighted_defect_density) * 100.0

    stats = {
        "strict_iou": strict_iou,
        "dfm_yield": dfm_yield,
        "total_error": int(union - inter),
        "crit_open": int(crit_open),
        "crit_short": int(crit_short),
        "marginal_shrink": int(marginal_shrink),
        "allowable_serif": int(allowable_serif)
    }
    return stats

# 8. 고속 광학 결상 연산 (1D Separable Gaussian Filter 적용)
def simulate_aerial_image(mask_mat, sigma):
    exp = gaussian_filter(mask_mat, sigma=sigma, mode='constant', cval=0.0)
    max_e = np.max(exp) if np.max(exp) > 0 else 1.0
    return exp / max_e

norm_exp_before = simulate_aerial_image(target_sub, sigma_px)
printed_before = (norm_exp_before >= threshold).astype(float)
stats_b = evaluate_context_aware_yield(target_sub, printed_before)

norm_exp_after = simulate_aerial_image(st.session_state.current_mask_sub, sigma_px)
printed_after = (norm_exp_after >= threshold).astype(float)
stats_a = evaluate_context_aware_yield(target_sub, printed_after)

iou_delta = stats_a["strict_iou"] - stats_b["strict_iou"]
dfm_delta = stats_a["dfm_yield"] - stats_b["dfm_yield"]
open_saved = stats_b["crit_open"] - stats_a["crit_open"]
short_added = stats_a["crit_short"] - stats_b["crit_short"]

# 9. 메트릭 대시보드
st.subheader("Quantitative Evaluation (Dual Primary Metrics: Geometric IoU & DFM Yield)")
m_row1_col1, m_row1_col2 = st.columns(2)

with m_row1_col1:
    st.markdown("#### 📐 Metric 1: Geometric Fidelity (기하학적 일치도)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Target Physical Size", f"{w_core_nm:.1f} × {h_core_nm:.1f} nm", f"{raw_target.shape[1]}×{raw_target.shape[0]} Grids")
    c2.metric("Before OPC (Strict IoU)", f"{stats_b['strict_iou']:.1f}%", f"Error: {stats_b['total_error']} px", delta_color="off")
    c3.metric("After OPC (Strict IoU)", f"{stats_a['strict_iou']:.1f}%", f"{iou_delta:+.1f}% p 개선", delta_color="normal")

with m_row1_col2:
    st.markdown("#### ⚡ Metric 2: Semiconductor DFM Yield (공정 신뢰성 실효 수율)")
    d1, d2, d3 = st.columns(3)
    d1.metric("치명적 단선(Open) 결함", f"{stats_a['crit_open']} px", f"{open_saved:+} px 구출 (위험 제로)", delta_color="normal")
    d2.metric("Before OPC (DFM 수율)", f"{stats_b['dfm_yield']:.1f}%", f"쇼트위험: {stats_b['crit_short']} px", delta_color="off")
    d3.metric("After OPC (DFM 수율)", f"{stats_a['dfm_yield']:.1f}%", f"{dfm_delta:+.1f}% p (쇼트 +{short_added}px)", delta_color="inverse")

# 10. CAD 치수 화살표
def draw_dimension_arrows(ax):
    if not show_dimension:
        return
    arrow_props = dict(arrowstyle='<->', color='#1a4b8c', lw=1.2)
    if layout_name == "INV_X1":
        ax.annotate('', xy=(x0_core, y0_core - 4), xytext=(x0_core + 63, y0_core - 4), arrowprops=arrow_props, annotation_clip=False)
        ax.text(x0_core + 31.5, y0_core - 6, '63 nm (18λ)', color='#1a4b8c', fontsize=8.5, fontweight='bold', ha='center', va='bottom', clip_on=False)
        ax.annotate('', xy=(x0_core + 67, y0_core), xytext=(x0_core + 67, y0_core + 189), arrowprops=arrow_props, annotation_clip=False)
        ax.text(x0_core + 70, y0_core + 94.5, '189 nm (54λ)', color='#1a4b8c', fontsize=8.5, fontweight='bold', ha='left', va='center', rotation=-90, clip_on=False)
    elif layout_name == "NAND2_X4":
        ax.annotate('', xy=(x0_core, y0_core - 4), xytext=(x0_core + 157.5, y0_core - 4), arrowprops=arrow_props, annotation_clip=False)
        ax.text(x0_core + 78.75, y0_core - 6, '157.5 nm (45λ)', color='#1a4b8c', fontsize=8.5, fontweight='bold', ha='center', va='bottom', clip_on=False)
        ax.annotate('', xy=(x0_core + 162, y0_core), xytext=(x0_core + 162, y0_core + 189), arrowprops=arrow_props, annotation_clip=False)
        ax.text(x0_core + 165, y0_core + 94.5, '189 nm (54λ)', color='#1a4b8c', fontsize=8.5, fontweight='bold', ha='left', va='center', rotation=-90, clip_on=False)
    else:
        ax.annotate('', xy=(x0_core, y0_core - 4), xytext=(x0_core + w_core_nm, y0_core - 4), arrowprops=arrow_props, annotation_clip=False)
        ax.text(x0_core + w_core_nm / 2.0, y0_core - 6, f'{w_core_nm:.1f} nm', color='#1a4b8c', fontsize=8.5, fontweight='bold', ha='center', va='bottom', clip_on=False)
        ax.annotate('', xy=(x0_core + w_core_nm + 4, y0_core), xytext=(x0_core + w_core_nm + 4, y0_core + h_core_nm), arrowprops=arrow_props, annotation_clip=False)
        ax.text(x0_core + w_core_nm + 7, y0_core + h_core_nm / 2.0, f'{h_core_nm:.1f} nm', color='#1a4b8c', fontsize=8.5, fontweight='bold', ha='left', va='center', rotation=-90, clip_on=False)

# 11. 외곽선 오버레이
def overlay_target_outline(ax):
    if not show_outline:
        return
    ax.contour(target_sub, levels=[0.5], colors=['#ffffff'], linestyles=['--'], linewidths=[1.2],
               extent=[0, total_w_nm, total_h_nm, 0], origin='upper')

# 12. 최적화된 종합 비교 플롯
def generate_comprehensive_plot():
    fig = plt.figure(figsize=(19.0, 12.0), facecolor='#ffffff')
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.0], hspace=0.28, wspace=0.22, left=0.06, right=0.95, top=0.93, bottom=0.05)

    major_x = 21.0 if total_w_nm > 50 else 10.5
    major_y = 31.5 if total_h_nm > 70 else 10.5

    # ROW 0: BEFORE OPC
    ax_top = [fig.add_subplot(gs[0, c]) for c in range(4)]
    fig.text(0.06, 0.955, "SECTION 1: BEFORE OPC (Conventional Mask on 0.875nm Sub-grid)", fontsize=12, fontweight='heavy', color='#b71c1c', ha='left')

    ax_top[0].imshow(target_sub, cmap='Blues', origin='upper', extent=extent_nm, interpolation='nearest', vmin=0, vmax=1)
    ax_top[0].set_title("(A) Target Pattern\n(Design: 3.5nm Grid)", fontsize=10, fontweight='bold', pad=8)
    draw_dimension_arrows(ax_top[0])

    ax_top[1].imshow(target_sub, cmap='Purples', origin='upper', extent=extent_nm, interpolation='nearest', vmin=0, vmax=1)
    ax_top[1].set_title("(B-1) Before Mask (No OPC)\n(0.875nm Raw Mask)", fontsize=10, fontweight='bold', pad=8)

    im_c0 = ax_top[2].imshow(norm_exp_before, cmap='inferno', origin='upper', extent=extent_nm, interpolation='bilinear', vmin=0.0, vmax=1.0)
    ax_top[2].set_title("(C-1) Aerial Image (Before)\n(Intensity Profile: σ=6.5nm)", fontsize=10, fontweight='bold', pad=8)
    overlay_target_outline(ax_top[2])
    cbar0 = fig.colorbar(im_c0, ax=ax_top[2], fraction=0.046, pad=0.04)
    cbar0.ax.tick_params(labelsize=8)

    diff_before = np.zeros((*target_sub.shape, 3))
    diff_before[(target_sub == 1) & (printed_before == 1)] = [0.1, 0.85, 0.1]
    diff_before[(target_sub == 1) & (printed_before == 0)] = [0.95, 0.1, 0.1]
    diff_before[(target_sub == 0) & (printed_before == 1)] = [0.1, 0.4, 0.95]
    ax_top[3].imshow(diff_before, origin='upper', extent=extent_nm, interpolation='nearest')
    ax_top[3].set_title(f"(D-1) Superposition (Before)\nStrict IoU: {stats_b['strict_iou']:.1f}% | DFM: {stats_b['dfm_yield']:.1f}%", fontsize=10, fontweight='bold', pad=8)
    overlay_target_outline(ax_top[3])

    # MIDDLE SUMMARY
    fig.text(0.06, 0.485, "SECTION 2: AFTER OPC (0.875nm Fine-Tuned Sub-grid OPC Optimization)", fontsize=12, fontweight='heavy', color='#1b5e20', ha='left', va='center')
    summary_text = (
        f"[SUB-GRID] Target: {target_unit_nm}nm | Mask: {mask_unit_nm:.3f}nm | 4x Reticle: {reticle_feature_nm:.1f}nm | PSF: σ={sigma_nm:.1f}nm, I_th={threshold:.2f}\n"
        f"[PRIMARY 1: GEOMETRIC IoU] Strict IoU: {stats_b['strict_iou']:.1f}% ──▶ {stats_a['strict_iou']:.1f}% ({iou_delta:+.1f}%p Improved)\n"
        f"[PRIMARY 2: DFM RELIABILITY] DFM Yield: {stats_b['dfm_yield']:.1f}% ──▶ {stats_a['dfm_yield']:.1f}% ({dfm_delta:+.1f}%p) | Opens: {stats_a['crit_open']}px | Short Hazards: +{short_added}px"
    )
    fig.text(0.95, 0.485, summary_text, fontsize=8, family='monospace', fontweight='bold', color='#0f172a', ha='right', va='center',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8fafc', edgecolor='#2563eb', linewidth=1.2, alpha=0.98))

    # ROW 1: AFTER OPC
    ax_bot = [fig.add_subplot(gs[1, c]) for c in range(4)]
    ax_bot[0].imshow(target_sub, cmap='Blues', origin='upper', extent=extent_nm, interpolation='nearest', vmin=0, vmax=1)
    ax_bot[0].set_title("(A) Target Reference\n(Golden Target Layout)", fontsize=10, fontweight='bold', pad=8)

    ax_bot[1].imshow(st.session_state.current_mask_sub, cmap='Purples', origin='upper', extent=extent_nm, interpolation='nearest', vmin=0, vmax=1)
    ax_bot[1].set_title("(B-2) After Mask (0.875nm Sub-grid)\n(4x Reticle: 3.5nm E-beam Serifs)", fontsize=10, fontweight='bold', pad=8)

    im_c1 = ax_bot[2].imshow(norm_exp_after, cmap='inferno', origin='upper', extent=extent_nm, interpolation='bilinear', vmin=0.0, vmax=1.0)
    ax_bot[2].set_title("(C-2) Aerial Image (After)\n(Corrected Intensity Profile)", fontsize=10, fontweight='bold', pad=8)
    overlay_target_outline(ax_bot[2])
    cbar1 = fig.colorbar(im_c1, ax=ax_bot[2], fraction=0.046, pad=0.04)
    cbar1.ax.tick_params(labelsize=8)

    diff_after = np.zeros((*target_sub.shape, 3))
    diff_after[(target_sub == 1) & (printed_after == 1)] = [0.1, 0.85, 0.1]
    diff_after[(target_sub == 1) & (printed_after == 0)] = [0.95, 0.1, 0.1]
    diff_after[(target_sub == 0) & (printed_after == 1)] = [0.1, 0.4, 0.95]
    ax_bot[3].imshow(diff_after, origin='upper', extent=extent_nm, interpolation='nearest')
    ax_bot[3].set_title(f"(D-2) Superposition (After)\nStrict IoU: {stats_a['strict_iou']:.1f}% | DFM: {stats_a['dfm_yield']:.1f}%", fontsize=10, fontweight='bold', pad=8)
    overlay_target_outline(ax_bot[3])

    for ax in ax_top + ax_bot:
        ax.set_xlabel("X Position (nm)", fontsize=8.5, fontweight='bold')
        ax.set_ylabel("Y Position (nm)", fontsize=8.5, fontweight='bold')
        ax.xaxis.set_major_locator(MultipleLocator(major_x))
        ax.yaxis.set_major_locator(MultipleLocator(major_y))
        ax.tick_params(axis='both', which='major', labelsize=8)
        if show_grid:
            ax.xaxis.set_minor_locator(MultipleLocator(target_unit_nm))
            ax.yaxis.set_minor_locator(MultipleLocator(target_unit_nm))
            ax.grid(which='minor', color='gray', linestyle=':', linewidth=0.4, alpha=0.4)
            ax.grid(which='major', color='black', linestyle='--', linewidth=0.6, alpha=0.5)

    return fig

st.subheader("Lithography & OPC Comprehensive Verification")
fig_result = generate_comprehensive_plot()
st.pyplot(fig_result)
plt.close(fig_result)  # 메모리 누수 방지

# 13. [초고속 완전 벡터화] 서브그리드 자동 OPC 엔진
def run_subgrid_opc_fast(target_mat_sub, sigma_px_val, opt_th, iterations=8):
    base = target_mat_sub.copy()
    is_symmetric = np.allclose(base, np.fliplr(base))
    mask = base.copy()

    # 거리 맵
    space_to_wire_nm = distance_transform_edt(base == 0) * mask_unit_nm
    wire_depth_nm = distance_transform_edt(base == 1) * mask_unit_nm

    # 1. 고속 코너 판정 (7x7 박스 필터 컨볼루션)
    box_kernel_7 = np.ones((7, 7), dtype=float)
    win_sum = convolve(base, box_kernel_7, mode='constant', cval=0.0)

    # 이웃 4방향 0 검출용 십자 커널
    cross_kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
    zero_neighbors = convolve((base == 0).astype(float), cross_kernel, mode='constant', cval=0.0)

    # (1) 외부 볼록 코너 세리프 (+) 부착
    convex_mask = (base == 0) & (space_to_wire_nm > 5.0) & (win_sum >= 14) & (win_sum <= 32)
    mask[convex_mask] = 1.0

    # (2) 내부 오목 코너 홈(-) 파내기 (Anti-Serif Cut)
    concave_mask = (base == 1) & (win_sum >= 28) & (win_sum < 36) & (wire_depth_nm <= 2.625) & (zero_neighbors >= 1)
    mask[concave_mask] = 0.0

    # 2. 선 끝단(Line-End) 안전 외곽 연장 (해머헤드 효과)
    struct_cross = generate_binary_structure(2, 1)
    dilated = binary_dilation(mask > 0, structure=struct_cross, iterations=3)
    safe_growth = dilated & (space_to_wire_nm > 5.0)
    mask = np.maximum(mask, safe_growth.astype(float))

    if is_symmetric:
        mask = np.maximum(mask, np.fliplr(mask))

    # 3. 척추 중심 코어 보호 (절대 뚫리지 않아야 하는 메인 도선)
    struct_sq = generate_binary_structure(2, 1)
    spine_core = binary_erosion(base > 0, structure=struct_sq, iterations=int(round(2.625 / mask_unit_nm))).astype(float)

    # 4. 고속 ILT 루프 (gaussian_filter 기반 가속)
    cont_mask = mask.copy()
    diff_sigma = sigma_px_val * 0.55

    for _ in range(iterations):
        bin_m = (cont_mask >= 0.40).astype(float)
        bin_m = np.maximum(bin_m, spine_core)

        exp = gaussian_filter(bin_m, sigma=sigma_px_val, mode='constant', cval=0.0)
        max_e = np.max(exp) if np.max(exp) > 0 else 1.0
        printed = ((exp / max_e) >= opt_th).astype(float)

        error = base - printed
        grad = gaussian_filter(error, sigma=diff_sigma, mode='constant', cval=0.0)

        cont_mask += 0.50 * grad
        cont_mask = np.clip(cont_mask, 0.0, 1.0)
        cont_mask = np.maximum(cont_mask, spine_core)

        if is_symmetric:
            cont_mask = 0.5 * (cont_mask + np.fliplr(cont_mask))

    final_mask = (cont_mask >= 0.38).astype(float)
    final_mask = np.maximum(final_mask, spine_core)

    if is_symmetric:
        final_mask = np.maximum(final_mask, np.fliplr(final_mask))

    return final_mask

# 14. 마스크 제어 및 내보내기 영역
st.subheader("OPC Modification & Export")
ctrl_cols = st.columns(5 if layout_mode == "Interactive Grid Drawing" else 4)

with ctrl_cols[0]:
    if st.button("🔄 마스크 초기화 (Reset)"):
        st.session_state.current_mask_sub = target_sub.copy()
        st.rerun()

with ctrl_cols[1]:
    if st.button("⚡ 0.875nm 서브그리드 자동 OPC 실행 (ILT Loop)", type="primary"):
        with st.spinner("0.875nm 서브그리드 고속 역노광 연산 중..."):
            optimized_mask = run_subgrid_opc_fast(target_sub, sigma_px, threshold, iterations=8)
            st.session_state.current_mask_sub = optimized_mask
            st.rerun()

with ctrl_cols[2]:
    mask_csv_buf = io.StringIO()
    np.savetxt(mask_csv_buf, st.session_state.current_mask_sub.astype(int), delimiter=",", fmt="%d")
    st.download_button(
        label="📥 0.875nm 마스크 도면 다운로드 (CSV)",
        data=mask_csv_buf.getvalue(),
        file_name=f"{layout_name}_0p875nm_subgrid_mask.csv",
        mime="text/csv"
    )

if layout_mode == "Interactive Grid Drawing":
    with ctrl_cols[3]:
        drawn_csv_buf = io.StringIO()
        np.savetxt(drawn_csv_buf, raw_target.astype(int), delimiter=",", fmt="%d")
        st.download_button(
            label="📥 그린 타겟 다운로드 (CSV)",
            data=drawn_csv_buf.getvalue(),
            file_name=f"{layout_name}_target.csv",
            mime="text/csv"
        )
    btn_col_idx = 4
else:
    btn_col_idx = 3

with ctrl_cols[btn_col_idx]:
    img_buf = io.BytesIO()
    fig_result.savefig(img_buf, format='png', dpi=200, bbox_inches='tight')
    st.download_button(
        label="📊 전·후 종합 비교 분석 플롯 (PNG)",
        data=img_buf.getvalue(),
        file_name=f"{layout_name}_0p875nm_subgrid_analysis.png",
        mime="image/png"
    )
