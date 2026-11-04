import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.ndimage import convolve, distance_transform_edt, generate_binary_structure, binary_erosion
import json
import io
import os

st.set_page_config(page_title="OPC Simulation Tool", layout="wide")
st.title("Optical Proximity Correction (OPC) Tool")
st.markdown("**7nm FinFET Standard Cell Metal1 Layout Simulation & Verification Platform**")
st.caption("Standard Design: 1 Target Grid = 3.5 nm × 3.5 nm | OPC Mask Sub-grid: 1 Sub-pixel = 0.875 nm (3.5nm / 4, 4x Reticle: 3.5 nm E-beam Unit)")

target_unit_nm = 3.5           # 웨이퍼 설계 기준 그리드 (nm)
subgrid_ratio = 4              # 정확한 4배 분할 서브그리드 (1 grid = 4 sub-pixels)
mask_unit_nm = target_unit_nm / subgrid_ratio  # 0.875 nm
reticle_mag = 4.0              # 4:1 축소 노광계
reticle_feature_nm = mask_unit_nm * reticle_mag # 3.5 nm 실물 마스크 E-beam 피처
pad_grids = 3                  # 타겟 기준 상하좌우 패딩 (3 그리드 = 10.5 nm)

# 1. 사이드바 설정
st.sidebar.header("Simulation Settings")
layout_mode = st.sidebar.radio("Layout Input Mode", ["Preset Standard Cells", "Interactive Grid Drawing", "CSV File Upload"])
process_mode = st.sidebar.radio("Process Mode", ["Standard 7nm Process Preset", "Custom Tuning Mode"])

# 2. 레이아웃 데이터 함수 (3.5 nm 원본 그리드)
def get_inv_x1_raw():
    mask = np.zeros((54, 18), dtype=float)
    mask[0:3, 0:18] = 1.0     # 상단 바 (y: 0~10.5 nm)
    mask[51:54, 0:18] = 1.0   # 하단 바 (y: 178.5~189 nm)
    mask[3:21, 3:6] = 1.0    # 좌상단 세로선
    mask[33:51, 3:6] = 1.0   # 좌하단 세로선
    mask[18:36, 12:15] = 1.0 # 우측 중앙 세로선
    return mask

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
        if mat is not None:
            st.session_state[matrix_key] = mat
            if action == "clear":
                st.session_state.editor_version += 1
                st.session_state.need_force_reset = True
                st.session_state.current_mask_sub = None
                st.rerun()
            elif action == "apply":
                st.session_state.current_mask_sub = None

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

# 4. 정수 4배 분할 서브그리드 확장 (에일리어싱 완전 제거)
padded_target = np.pad(raw_target, pad_grids, mode='constant', constant_values=0.0)
h_grids, w_grids = padded_target.shape
total_w_nm = w_grids * target_unit_nm
total_h_nm = h_grids * target_unit_nm

# 각 3.5nm 그리드를 4x4의 0.875nm 서브픽셀로 정수 복제 (선폭 왜곡 제로)
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
    st.sidebar.markdown("✅ **표준 7nm 공정 (정수 4분할 서브그리드 모드)**")
    kernel_type = "2D Gaussian Function"
    sigma_nm = 6.5
    threshold = 0.50
else:
    st.sidebar.subheader("Optical Lithography Parameters")
    kernel_type = st.sidebar.radio("Optical Blur Kernel", ["2D Gaussian Function", "Standard 5x5 Matrix"])
    sigma_nm = st.sidebar.slider("Optical Blur Radius (sigma, nm)", 3.0, 12.0, 6.5, 0.5)
    threshold = st.sidebar.slider("Resist Threshold (I_th)", 0.20, 0.80, 0.50, 0.01)

st.sidebar.subheader("Display Options")
show_grid = st.sidebar.checkbox("Display Unit Grid (3.5 nm Design Pitch)", value=True)
show_dimension = st.sidebar.checkbox("Display CAD Dimension Arrows", value=True)
show_outline = st.sidebar.checkbox("Display Target Outline (White Dashed)", value=True)

# 0.875 nm 서브픽셀 단위 가우시안 커널
sigma_px = sigma_nm / mask_unit_nm
k_radius = int(round(sigma_px * 3.5)) | 1
ax_k = np.linspace(-(k_radius // 2), k_radius // 2, k_radius)
xx, yy = np.meshgrid(ax_k, ax_k)
opt_kernel_sub = np.exp(-(xx**2 + yy**2) / (2.0 * sigma_px**2))
opt_kernel_sub = opt_kernel_sub / np.sum(opt_kernel_sub)

# 7. DFM 실효 수율 평가 함수
def evaluate_context_aware_yield(target_mat, printed_mat):
    inter = np.logical_and(target_mat == 1, printed_mat == 1).sum()
    union = np.logical_or(target_mat == 1, printed_mat == 1).sum()
    strict_iou = (inter / union) * 100.0 if union > 0 else 0.0

    # 서브픽셀 단위 거리
    space_to_wire_nm = distance_transform_edt(target_mat == 0) * mask_unit_nm

    # 배선 척추 코어 (중심 5nm 선폭 보존)
    struct_sq = generate_binary_structure(2, 1)
    core = binary_erosion(target_mat, structure=struct_sq, iterations=int(round(2.5 / mask_unit_nm))).astype(float)

    under_mask = (target_mat == 1) & (printed_mat == 0)
    over_mask = (target_mat == 0) & (printed_mat == 1)

    crit_open = np.logical_and(under_mask, core == 1).sum()
    marginal_shrink = under_mask.sum() - crit_open

    # 인접 배선 7nm 이내 위험 구역 번짐
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

# 8. 전/후 연산
exp_before = convolve(target_sub, opt_kernel_sub, mode='constant', cval=0.0)
max_eb = np.max(exp_before) if np.max(exp_before) > 0 else 1.0
norm_exp_before = exp_before / max_eb
printed_before = (norm_exp_before >= threshold).astype(float)
stats_b = evaluate_context_aware_yield(target_sub, printed_before)

exp_after = convolve(st.session_state.current_mask_sub, opt_kernel_sub, mode='constant', cval=0.0)
max_ea = np.max(exp_after) if np.max(exp_after) > 0 else 1.0
norm_exp_after = exp_after / max_ea
printed_after = (norm_exp_after >= threshold).astype(float)
stats_a = evaluate_context_aware_yield(target_sub, printed_after)

dfm_delta = stats_a["dfm_yield"] - stats_b["dfm_yield"]
iou_delta = stats_a["strict_iou"] - stats_b["strict_iou"]
open_saved = stats_b["crit_open"] - stats_a["crit_open"]
short_saved = stats_b["crit_short"] - stats_a["crit_short"]

# 9. 메트릭 대시보드
st.subheader("Quantitative DFM Evaluation (0.875 nm Sub-grid OPC Performance)")
m_col1, m_col2, m_col3, m_col4 = st.columns(4)
m_col1.metric("Target Physical Size", f"{w_core_nm:.1f} × {h_core_nm:.1f} nm", f"{raw_target.shape[1]}×{raw_target.shape[0]} Grids (Sub-grid: 0.875nm)")
m_col2.metric("Before OPC (DFM 실효 수율)", f"{stats_b['dfm_yield']:.1f}%", f"IoU: {stats_b['strict_iou']:.1f}% (단선 {stats_b['crit_open']}px)", delta_color="off")
m_col3.metric("After OPC (DFM 실효 수율)", f"{stats_a['dfm_yield']:.1f}%", f"{dfm_delta:+.1f}% p 향상 (IoU {stats_a['strict_iou']:.1f}%)", delta_color="normal")
m_col4.metric("치명 결함(단선/쇼트) 구출", f"{open_saved + short_saved} px 제거", f"잔여 치명결함: {stats_a['crit_open'] + stats_a['crit_short']} px", delta_color="normal")

# 10. CAD 치수 화살표
def draw_dimension_arrows(ax):
    if not show_dimension:
        return

    arrow_props = dict(arrowstyle='<->', color='#1a4b8c', lw=1.3)

    if layout_name == "INV_X1":
        ax.annotate('', xy=(x0_core, y0_core - 4), xytext=(x0_core + 63, y0_core - 4), arrowprops=arrow_props, annotation_clip=False)
        ax.text(x0_core + 31.5, y0_core - 6, '63 nm (18λ)', color='#1a4b8c', fontsize=9, fontweight='bold', ha='center', va='bottom', clip_on=False)

        ax.annotate('', xy=(x0_core + 67, y0_core), xytext=(x0_core + 67, y0_core + 189), arrowprops=arrow_props, annotation_clip=False)
        ax.text(x0_core + 70, y0_core + 94.5, '189 nm (54λ)', color='#1a4b8c', fontsize=9, fontweight='bold', ha='left', va='center', rotation=-90, clip_on=False)

        ax.annotate('', xy=(x0_core + 55, y0_core + 63), xytext=(x0_core + 55, y0_core + 126), arrowprops=arrow_props, annotation_clip=False)
        ax.text(x0_core + 58, y0_core + 94.5, '63 nm*', color='#1a4b8c', fontsize=8, fontweight='bold', ha='left', va='center', rotation=-90, clip_on=False)

    elif layout_name == "NAND2_X4":
        ax.annotate('', xy=(x0_core, y0_core - 4), xytext=(x0_core + 157.5, y0_core - 4), arrowprops=arrow_props, annotation_clip=False)
        ax.text(x0_core + 78.75, y0_core - 6, '157.5 nm (45λ)', color='#1a4b8c', fontsize=9, fontweight='bold', ha='center', va='bottom', clip_on=False)

        ax.annotate('', xy=(x0_core + 162, y0_core), xytext=(x0_core + 162, y0_core + 189), arrowprops=arrow_props, annotation_clip=False)
        ax.text(x0_core + 165, y0_core + 94.5, '189 nm (54λ)', color='#1a4b8c', fontsize=9, fontweight='bold', ha='left', va='center', rotation=-90, clip_on=False)

    else:
        ax.annotate('', xy=(x0_core, y0_core - 4), xytext=(x0_core + w_core_nm, y0_core - 4), arrowprops=arrow_props, annotation_clip=False)
        ax.text(x0_core + w_core_nm / 2.0, y0_core - 6, f'{w_core_nm:.1f} nm', color='#1a4b8c', fontsize=9, fontweight='bold', ha='center', va='bottom', clip_on=False)

        ax.annotate('', xy=(x0_core + w_core_nm + 4, y0_core), xytext=(x0_core + w_core_nm + 4, y0_core + h_core_nm), arrowprops=arrow_props, annotation_clip=False)
        ax.text(x0_core + w_core_nm + 7, y0_core + h_core_nm / 2.0, f'{h_core_nm:.1f} nm', color='#1a4b8c', fontsize=9, fontweight='bold', ha='left', va='center', rotation=-90, clip_on=False)

# 11. 타겟 외곽선 오버레이
def overlay_target_outline(ax):
    if not show_outline:
        return
    ax.contour(target_sub, levels=[0.5], colors=['#ffffff'], linestyles=['--'], linewidths=[1.4],
               extent=[0, total_w_nm, total_h_nm, 0], origin='upper')

# 12. 전/후 종합 비교 분석 플롯
def generate_comprehensive_plot():
    fig = plt.figure(figsize=(22.5, 17.5), facecolor='#ffffff')

    gs = fig.add_gridspec(
        3, 4,
        height_ratios=[1.0, 0.18, 1.0],
        hspace=0.45,
        wspace=0.22,
        left=0.06, right=0.95, top=0.93, bottom=0.04
    )

    major_x = 21.0 if total_w_nm > 50 else 10.5
    major_y = 31.5 if total_h_nm > 70 else 10.5

    # ------------------ ROW 0: BEFORE OPC ------------------
    ax_top = [fig.add_subplot(gs[0, c]) for c in range(4)]
    fig.text(0.06, 0.965, "SECTION 1: BEFORE OPC (Conventional Mask on 0.875nm Sub-grid)",
             fontsize=14, fontweight='heavy', color='#b71c1c', ha='left')

    ax_top[0].imshow(target_sub, cmap='Blues', origin='upper', extent=extent_nm, interpolation='nearest', vmin=0, vmax=1)
    ax_top[0].set_title("(A) Target Pattern\n(Design Specification: 3.5nm Grid)", fontsize=11, fontweight='bold', pad=12)
    draw_dimension_arrows(ax_top[0])

    ax_top[1].imshow(target_sub, cmap='Purples', origin='upper', extent=extent_nm, interpolation='nearest', vmin=0, vmax=1)
    ax_top[1].set_title("(B-1) Before Mask (No OPC)\n(0.875nm Sub-pixel Raw Mask)", fontsize=11, fontweight='bold', pad=12)

    im_c0 = ax_top[2].imshow(norm_exp_before, cmap='inferno', origin='upper', extent=extent_nm, interpolation='bilinear', vmin=0.0, vmax=1.0)
    ax_top[2].set_title("(C-1) Aerial Image (Before)\n(Intensity Profile: σ=6.5nm)", fontsize=11, fontweight='bold', pad=12)
    overlay_target_outline(ax_top[2])
    cbar0 = fig.colorbar(im_c0, ax=ax_top[2], fraction=0.046, pad=0.04)
    cbar0.ax.tick_params(labelsize=9)

    diff_before = np.zeros((*target_sub.shape, 3))
    diff_before[(target_sub == 1) & (printed_before == 1)] = [0.1, 0.85, 0.1]
    diff_before[(target_sub == 1) & (printed_before == 0)] = [0.95, 0.1, 0.1]
    diff_before[(target_sub == 0) & (printed_before == 1)] = [0.1, 0.4, 0.95]
    ax_top[3].imshow(diff_before, origin='upper', extent=extent_nm, interpolation='nearest')
    ax_top[3].set_title(f"(D-1) Superposition (Before)\nDFM Yield: {stats_b['dfm_yield']:.1f}% (IoU {stats_b['strict_iou']:.1f}%)", fontsize=11, fontweight='bold', pad=12)
    overlay_target_outline(ax_top[3])

    # ------------------ ROW 1: MIDDLE PARAMETER BRIDGE (0.875nm & 4x Reticle) ------------------
    ax_mid = fig.add_subplot(gs[1, :])
    ax_mid.axis('off')

    summary_text = (
        f"  [SUB-GRID SPEC] Target Grid: {target_unit_nm} nm (λ)  |  Mask Sub-grid: {mask_unit_nm:.3f} nm/px (1/4 Grid)  |  4x Reticle: {reticle_feature_nm:.1f} nm E-beam Unit\n"
        f"  [LITHO CONFIG] Optical PSF Diffusion: σ = {sigma_nm:.1f} nm  |  Resist Threshold: I_th = {threshold:.2f}  |  Wire Pitch: 10.5 nm\n"
        f"  [DFM YIELD ADVANTAGE] Effective Yield: {stats_b['dfm_yield']:.1f}% ──▶ {stats_a['dfm_yield']:.1f}% ({dfm_delta:+.1f}%p)  |  "
        f"Strict IoU: {stats_b['strict_iou']:.1f}% ──▶ {stats_a['strict_iou']:.1f}%\n"
        f"  [DEFECT METRICS] Fatal Disconnections (Open): {stats_b['crit_open']} px ──▶ {stats_a['crit_open']} px ({open_saved:+} px Saved)  |  "
        f"Short Hazards: {stats_b['crit_short']} px ──▶ {stats_a['crit_short']} px"
    )

    ax_mid.text(
        0.5, 0.62, summary_text,
        fontsize=10.5, family='monospace', fontweight='bold', color='#0f172a',
        ha='center', va='center',
        bbox=dict(
            boxstyle='round,pad=0.8',
            facecolor='#f8fafc',
            edgecolor='#2563eb',
            linewidth=1.8,
            alpha=0.98
        )
    )

    # ------------------ ROW 2: AFTER OPC ------------------
    ax_bot = [fig.add_subplot(gs[2, c]) for c in range(4)]
    fig.text(0.06, 0.465, "SECTION 2: AFTER OPC (0.875nm Fine-Tuned Sub-grid OPC Optimization)",
             fontsize=14, fontweight='heavy', color='#1b5e20', ha='left')

    ax_bot[0].imshow(target_sub, cmap='Blues', origin='upper', extent=extent_nm, interpolation='nearest', vmin=0, vmax=1)
    ax_bot[0].set_title("(A) Target Reference\n(Golden Target Layout)", fontsize=11, fontweight='bold', pad=12)

    ax_bot[1].imshow(st.session_state.current_mask_sub, cmap='Purples', origin='upper', extent=extent_nm, interpolation='nearest', vmin=0, vmax=1)
    ax_bot[1].set_title("(B-2) After Mask (0.875nm Sub-grid)\n(4x Reticle: 3.5nm E-beam Serifs)", fontsize=11, fontweight='bold', pad=12)

    im_c1 = ax_bot[2].imshow(norm_exp_after, cmap='inferno', origin='upper', extent=extent_nm, interpolation='bilinear', vmin=0.0, vmax=1.0)
    ax_bot[2].set_title("(C-2) Aerial Image (After)\n(Sub-grid Corrected Intensity)", fontsize=11, fontweight='bold', pad=12)
    overlay_target_outline(ax_bot[2])
    cbar1 = fig.colorbar(im_c1, ax=ax_bot[2], fraction=0.046, pad=0.04)
    cbar1.ax.tick_params(labelsize=9)

    diff_after = np.zeros((*target_sub.shape, 3))
    diff_after[(target_sub == 1) & (printed_after == 1)] = [0.1, 0.85, 0.1]
    diff_after[(target_sub == 1) & (printed_after == 0)] = [0.95, 0.1, 0.1]
    diff_after[(target_sub == 0) & (printed_after == 1)] = [0.1, 0.4, 0.95]
    ax_bot[3].imshow(diff_after, origin='upper', extent=extent_nm, interpolation='nearest')
    ax_bot[3].set_title(f"(D-2) Superposition (After)\nDFM Yield: {stats_a['dfm_yield']:.1f}% (IoU {stats_a['strict_iou']:.1f}%)", fontsize=11, fontweight='bold', pad=12)
    overlay_target_outline(ax_bot[3])

    for ax in ax_top + ax_bot:
        ax.set_xlabel("X Position (nm)", fontsize=9.5, fontweight='bold')
        ax.set_ylabel("Y Position (nm)", fontsize=9.5, fontweight='bold')
        ax.xaxis.set_major_locator(MultipleLocator(major_x))
        ax.yaxis.set_major_locator(MultipleLocator(major_y))
        ax.tick_params(axis='both', which='major', labelsize=8.5)
        if show_grid:
            ax.xaxis.set_minor_locator(MultipleLocator(target_unit_nm))
            ax.yaxis.set_minor_locator(MultipleLocator(target_unit_nm))
            ax.grid(which='minor', color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
            ax.grid(which='major', color='black', linestyle='--', linewidth=0.7, alpha=0.6)

    return fig

st.subheader("Lithography & OPC Comprehensive Verification")
fig_result = generate_comprehensive_plot()
st.pyplot(fig_result)

# 13. 0.875 nm 서브그리드 전용 정밀 ILT & 대칭 세리프 최적화 엔진
def run_subgrid_opc(target_mat_sub, opt_kernel, opt_th, iterations=8):
    h, w = target_mat_sub.shape
    base = target_mat_sub.copy()

    is_symmetric = np.allclose(base, np.fliplr(base))
    mask = base.copy()

    space_to_wire_nm = distance_transform_edt(base == 0) * mask_unit_nm

    # 0.875 nm 서브픽셀 단위 미세 돌출 (코너 2~3 sub-pixels, 끝단 3~4 sub-pixels 연장)
    pad = np.pad(base, 3, mode='constant', constant_values=0)
    for r in range(h):
        for c in range(w):
            center = pad[r+3, c+3]
            if center == 0:
                # 쇼트 위험 없는 외곽 빈 공간(>7nm)에만 정밀 세리프 부착
                if space_to_wire_nm[r, c] > 7.0:
                    corner_window = pad[r:r+7, c:c+7]
                    if np.sum(corner_window) >= 24:
                        mask[r, c] = 1.0
            elif center == 1:
                # 상하 선 끝단(Line-End) 3~4 sub-pixels (약 2.6~3.5 nm) 정밀 연장
                up, down = pad[r+2, c+3], pad[r+4, c+3]
                if (up == 0 and down == 1) or (up == 1 and down == 0):
                    if r > 2 and pad[r+2, c+3] == 0:
                        mask[max(0, r-3):r+1, c] = 1.0
                    if r < h - 3 and pad[r+4, c+3] == 0:
                        mask[r:min(h, r+4), c] = 1.0

    if is_symmetric:
        mask = np.maximum(mask, np.fliplr(mask))

    # 오차 역확산 ILT 루프
    k_diff = int(round(sigma_px * 1.5)) | 1
    ax_d = np.linspace(-(k_diff // 2), k_diff // 2, k_diff)
    xd, yd = np.meshgrid(ax_d, ax_d)
    diff_kernel = np.exp(-(xd**2 + yd**2) / (2.0 * (sigma_px * 0.55)**2))
    diff_kernel = diff_kernel / np.sum(diff_kernel)

    cont_mask = mask.copy()
    for _ in range(iterations):
        bin_m = (cont_mask >= 0.48).astype(float)
        exp = convolve(bin_m, opt_kernel, mode='constant', cval=0.0)
        max_e = np.max(exp) if np.max(exp) > 0 else 1.0
        printed = ((exp / max_e) >= opt_th).astype(float)

        error = base - printed
        grad = convolve(error, diff_kernel, mode='constant', cval=0.0)

        cont_mask += 0.40 * grad
        cont_mask = np.clip(cont_mask, 0.0, 1.0)
        cont_mask = np.maximum(cont_mask, base)  # 원본 선폭 골격 절대 보존

        if is_symmetric:
            cont_mask = 0.5 * (cont_mask + np.fliplr(cont_mask))

    final_mask = (cont_mask >= 0.48).astype(float)
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
        with st.spinner("0.875nm 정밀 서브그리드 역노광 최적화 연산 중..."):
            optimized_mask = run_subgrid_opc(target_sub, opt_kernel_sub, threshold, iterations=8)
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
    fig_result.savefig(img_buf, format='png', dpi=300, bbox_inches='tight')
    st.download_button(
        label="📊 전·후 종합 비교 분석 플롯 (PNG)",
        data=img_buf.getvalue(),
        file_name=f"{layout_name}_0p875nm_subgrid_analysis.png",
        mime="image/png"
    )
