import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.ndimage import convolve, binary_erosion
import json
import io
import os

st.set_page_config(page_title="OPC Simulation Tool", layout="wide")
st.title("Optical Proximity Correction (OPC) Tool")
st.markdown("**7nm FinFET Standard Cell Metal1 Layout Simulation & Verification Platform**")
st.caption("Standard Unit: 1 Grid = 3.5 nm × 3.5 nm (λ = 3.5 nm, Metal1 Wire Width = 3 Grid = 10.5 nm)")

unit_scale = 3.5  # nm per pixel
pad_size = 3      # 마스크 확장 패딩 (상하좌우 3픽셀 = 10.5 nm 여유 공간)

# 1. 사이드바 설정
st.sidebar.header("Simulation Settings")
layout_mode = st.sidebar.radio("Layout Input Mode", ["Preset Standard Cells", "Interactive Grid Drawing", "CSV File Upload"])
process_mode = st.sidebar.radio("Process Mode", ["Standard 7nm Process Preset", "Custom Tuning Mode"])

# 2. 레이아웃 데이터 함수 (내부 원본)
def get_inv_x1_raw():
    mask = np.zeros((54, 18), dtype=float)
    mask[0:3, 0:18] = 1.0
    mask[51:54, 0:18] = 1.0
    mask[3:21, 3:6] = 1.0
    mask[33:51, 3:6] = 1.0
    mask[18:36, 12:15] = 1.0
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
                st.session_state.current_mask = None
                st.rerun()
            elif action == "apply":
                st.session_state.current_mask = None

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

# 핵심: 마스크가 패턴 바깥으로 뻗어나갈 수 있도록 상하좌우 패딩(Padding)을 적용한 확장 타겟 생성!
target = np.pad(raw_target, pad_size, mode='constant', constant_values=0.0)

# 4. 세션 마스크 상태 관리
if "current_mask" not in st.session_state or st.session_state.current_mask is None or st.session_state.get("last_layout") != layout_name:
    st.session_state.current_mask = target.copy()
    st.session_state.last_layout = layout_name

if st.session_state.current_mask.shape != target.shape:
    st.session_state.current_mask = target.copy()

# 5. 광학 공정 파라미터
if process_mode == "Standard 7nm Process Preset":
    st.sidebar.markdown("✅ **표준 7nm 공정 파라미터가 적용되었습니다.**")
    kernel_type = "2D Gaussian Function"
    sigma = 1.9
    threshold = 0.50
else:
    st.sidebar.subheader("Optical Lithography Parameters")
    kernel_type = st.sidebar.radio("Optical Blur Kernel", ["2D Gaussian Function", "Standard 5x5 Matrix"])
    sigma = st.sidebar.slider("Gaussian Diffusion Radius (sigma, px)", 1.0, 3.5, 1.9, 0.1)
    threshold = st.sidebar.slider("Resist Threshold (I_th)", 0.20, 0.80, 0.50, 0.01)

st.sidebar.subheader("Display Options")
show_grid = st.sidebar.checkbox("Display Unit Grid (3.5 nm)", value=True)
show_dimension = st.sidebar.checkbox("Display CAD Dimension Arrows", value=True)
show_outline = st.sidebar.checkbox("Display Target Outline (White Dashed)", value=True)

# 커널 생성
if kernel_type == "Standard 5x5 Matrix":
    kernel = np.array([
        [0.0, 0.5, 1.0, 0.5, 0.0],
        [0.5, 4.0, 5.0, 4.0, 0.5],
        [1.0, 5.0, 10.0, 5.0, 1.0],
        [0.5, 4.0, 5.0, 4.0, 0.5],
        [0.0, 0.5, 1.0, 0.5, 0.0]
    ])
    kernel = kernel / np.sum(kernel)
else:
    k_size = int(round(sigma * 6)) | 1
    ax_k = np.linspace(-(k_size // 2), k_size // 2, k_size)
    xx, yy = np.meshgrid(ax_k, ax_k)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    kernel = kernel / np.sum(kernel)

# 6. 전/후 시뮬레이션 연산 (패딩된 확장 캔버스에서 수행)
exp_before = convolve(target, kernel, mode='constant', cval=0.0)
max_eb = np.max(exp_before) if np.max(exp_before) > 0 else 1.0
norm_exp_before = exp_before / max_eb
printed_before = (norm_exp_before >= threshold).astype(float)

inter_b = np.logical_and(target == 1, printed_before == 1).sum()
union_b = np.logical_or(target == 1, printed_before == 1).sum()
iou_before = (inter_b / union_b) * 100.0 if union_b > 0 else 0.0
error_b = int(union_b - inter_b)

exp_after = convolve(st.session_state.current_mask, kernel, mode='constant', cval=0.0)
max_ea = np.max(exp_after) if np.max(exp_after) > 0 else 1.0
norm_exp_after = exp_after / max_ea
printed_after = (norm_exp_after >= threshold).astype(float)

inter_a = np.logical_and(target == 1, printed_after == 1).sum()
union_a = np.logical_or(target == 1, printed_after == 1).sum()
iou_after = (inter_a / union_a) * 100.0 if union_a > 0 else 0.0
error_a = int(union_a - inter_a)

iou_delta = iou_after - iou_before
error_delta = error_b - error_a

# 치수 계산 (패딩 포함 전체 캔버스)
h_tot, w_tot = target.shape
width_tot_nm = w_tot * unit_scale
height_tot_nm = h_tot * unit_scale
extent_nm = [0, width_tot_nm, height_tot_nm, 0]

# 원본 패턴 코어 영역 좌표 (nm)
x0_core = pad_size * unit_scale
y0_core = pad_size * unit_scale
w_core_nm = raw_target.shape[1] * unit_scale
h_core_nm = raw_target.shape[0] * unit_scale

# 7. 메트릭 대시보드
st.subheader("Quantitative Evaluation & Improvement")
m_col1, m_col2, m_col3, m_col4 = st.columns(4)
m_col1.metric("Pattern Dimension", f"{w_core_nm:.1f} × {h_core_nm:.1f} nm", f"{raw_target.shape[1]} × {raw_target.shape[0]} Grids")
m_col2.metric("Before OPC (IoU)", f"{iou_before:.2f}%", f"{error_b} px Error", delta_color="off")
m_col3.metric("After OPC (IoU)", f"{iou_after:.2f}%", f"{iou_delta:+.2f}% p", delta_color="normal")
m_col4.metric("Error Reduction", f"{error_delta} px Saved", f"Improvement: {(iou_delta/iou_before*100) if iou_before>0 else 0:+.1f}%", delta_color="normal")

# 8. CAD 치수 화살표 렌더링 함수 (글자 겹침 완벽 차단)
def draw_dimension_arrows(ax):
    if not show_dimension:
        return

    arrow_props = dict(arrowstyle='<->', color='#1a4b8c', lw=1.3)

    if layout_name == "INV_X1":
        # 상단 전체 가로폭: 63 nm (18λ)
        ax.annotate('', xy=(x0_core, y0_core - 4), xytext=(x0_core + 63, y0_core - 4), arrowprops=arrow_props, annotation_clip=False)
        ax.text(x0_core + 31.5, y0_core - 6, '63 nm (18λ)', color='#1a4b8c', fontsize=9, fontweight='bold', ha='center', va='bottom', clip_on=False)

        # 우측 전체 높이: 189 nm (54λ)
        ax.annotate('', xy=(x0_core + 67, y0_core), xytext=(x0_core + 67, y0_core + 189), arrowprops=arrow_props, annotation_clip=False)
        ax.text(x0_core + 70, y0_core + 94.5, '189 nm (54λ)', color='#1a4b8c', fontsize=9, fontweight='bold', ha='left', va='center', rotation=-90, clip_on=False)

        # 우측 중앙 세로선: 63 nm*
        ax.annotate('', xy=(x0_core + 55, y0_core + 63), xytext=(x0_core + 55, y0_core + 126), arrowprops=arrow_props, annotation_clip=False)
        ax.text(x0_core + 58, y0_core + 94.5, '63 nm*', color='#1a4b8c', fontsize=8, fontweight='bold', ha='left', va='center', rotation=-90, clip_on=False)

    elif layout_name == "NAND2_X4":
        # 상단 전체 가로폭: 157.5 nm (45λ)
        ax.annotate('', xy=(x0_core, y0_core - 4), xytext=(x0_core + 157.5, y0_core - 4), arrowprops=arrow_props, annotation_clip=False)
        ax.text(x0_core + 78.75, y0_core - 6, '157.5 nm (45λ)', color='#1a4b8c', fontsize=9, fontweight='bold', ha='center', va='bottom', clip_on=False)

        # 우측 전체 높이: 189 nm (54λ)
        ax.annotate('', xy=(x0_core + 162, y0_core), xytext=(x0_core + 162, y0_core + 189), arrowprops=arrow_props, annotation_clip=False)
        ax.text(x0_core + 165, y0_core + 94.5, '189 nm (54λ)', color='#1a4b8c', fontsize=9, fontweight='bold', ha='left', va='center', rotation=-90, clip_on=False)

    else:
        ax.annotate('', xy=(x0_core, y0_core - 4), xytext=(x0_core + w_core_nm, y0_core - 4), arrowprops=arrow_props, annotation_clip=False)
        ax.text(x0_core + w_core_nm / 2.0, y0_core - 6, f'{w_core_nm:.1f} nm', color='#1a4b8c', fontsize=9, fontweight='bold', ha='center', va='bottom', clip_on=False)

        ax.annotate('', xy=(x0_core + w_core_nm + 4, y0_core), xytext=(x0_core + w_core_nm + 4, y0_core + h_core_nm), arrowprops=arrow_props, annotation_clip=False)
        ax.text(x0_core + w_core_nm + 7, y0_core + h_core_nm / 2.0, f'{h_core_nm:.1f} nm', color='#1a4b8c', fontsize=9, fontweight='bold', ha='left', va='center', rotation=-90, clip_on=False)

# 9. 타겟 외곽선 오버레이
def overlay_target_outline(ax):
    if not show_outline:
        return
    ax.contour(target, levels=[0.5], colors=['#ffffff'], linestyles=['--'], linewidths=[1.4],
               extent=[0, width_tot_nm, height_tot_nm, 0], origin='upper')

# 10. 전/후 명확 분리 종합 비교 분석 플롯 (여백 및 글자 위치 최적화)
def generate_comprehensive_plot():
    fig = plt.figure(figsize=(22.5, 17.5), facecolor='#ffffff')

    # GridSpec: 상단(Before), 중앙 배너, 하단(After)
    gs = fig.add_gridspec(
        3, 4,
        height_ratios=[1.0, 0.18, 1.0],
        hspace=0.45,
        wspace=0.22,
        left=0.06, right=0.95, top=0.93, bottom=0.04
    )

    major_x = 21.0 if width_tot_nm > 50 else 10.5
    major_y = 31.5 if height_tot_nm > 70 else 10.5

    # ------------------ ROW 0: BEFORE OPC ------------------
    ax_top = [fig.add_subplot(gs[0, c]) for c in range(4)]
    # 섹션 제목을 위로 바짝 올려서 치수선과 닿지 않도록 설정
    fig.text(0.06, 0.965, "SECTION 1: BEFORE OPC (Conventional Mask Simulation)",
             fontsize=14, fontweight='heavy', color='#b71c1c', ha='left')

    ax_top[0].imshow(target, cmap='Blues', origin='upper', extent=extent_nm, interpolation='nearest', vmin=0, vmax=1)
    ax_top[0].set_title("(A) Target Pattern\n(Design Specification)", fontsize=11, fontweight='bold', pad=12)
    draw_dimension_arrows(ax_top[0])

    ax_top[1].imshow(target, cmap='Purples', origin='upper', extent=extent_nm, interpolation='nearest', vmin=0, vmax=1)
    ax_top[1].set_title("(B-1) Before Mask (No OPC)\n(Raw Layout Mask)", fontsize=11, fontweight='bold', pad=12)

    im_c0 = ax_top[2].imshow(norm_exp_before, cmap='inferno', origin='upper', extent=extent_nm, interpolation='bilinear', vmin=0.0, vmax=1.0)
    ax_top[2].set_title("(C-1) Aerial Image (Before)\n(Light Intensity Profile)", fontsize=11, fontweight='bold', pad=12)
    overlay_target_outline(ax_top[2])
    cbar0 = fig.colorbar(im_c0, ax=ax_top[2], fraction=0.046, pad=0.04)
    cbar0.ax.tick_params(labelsize=9)

    diff_before = np.zeros((*target.shape, 3))
    diff_before[(target == 1) & (printed_before == 1)] = [0.1, 0.85, 0.1]
    diff_before[(target == 1) & (printed_before == 0)] = [0.95, 0.1, 0.1]
    diff_before[(target == 0) & (printed_before == 1)] = [0.1, 0.4, 0.95]
    ax_top[3].imshow(diff_before, origin='upper', extent=extent_nm, interpolation='nearest')
    ax_top[3].set_title(f"(D-1) Superposition (Before)\nIoU: {iou_before:.2f}% ({error_b}px error)", fontsize=11, fontweight='bold', pad=12)
    overlay_target_outline(ax_top[3])

    # ------------------ ROW 1: MIDDLE PARAMETER BRIDGE (위치 상향 조정) ------------------
    ax_mid = fig.add_subplot(gs[1, :])
    ax_mid.axis('off')

    kernel_str = f"2D Gaussian (sigma={sigma:.2f} px)" if kernel_type == "2D Gaussian Function" else "Standard 5x5 Matrix"
    summary_text = (
        f"  [PROCESS SPECIFICATION] Layout: {layout_name} (Core: {raw_target.shape[1]}×{raw_target.shape[0]} Grids, {w_core_nm:.1f}×{h_core_nm:.1f} nm)  |  "
        f"Extended Mask Padding: +{pad_size} Grids\n"
        f"  [LITHO CONFIG] Optical Kernel: {kernel_str}  |  Resist Threshold: I_th = {threshold:.2f}  |  Wire Pitch: 10.5 nm (3λ)\n"
        f"  [OPC PERFORMANCE] Pattern Fidelity (IoU): {iou_before:.2f}% ──▶ {iou_after:.2f}% ({iou_delta:+.2f}%p)  |  "
        f"Pixel Error: {error_b} px ──▶ {error_a} px ({error_delta:+} px Saved)  |  Improvement: {(iou_delta/iou_before*100) if iou_before>0 else 0:+.1f}%"
    )

    # 살짝 위(0.62)로 올려서 하단 제목과의 충돌 방지
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
    # 하단 섹션 제목을 살짝 내려서 중앙 배너와 절대 겹치지 않게 배치
    fig.text(0.06, 0.465, "SECTION 2: AFTER OPC (Safe-Core Extended Mask & Serif Optimization)",
             fontsize=14, fontweight='heavy', color='#1b5e20', ha='left')

    ax_bot[0].imshow(target, cmap='Blues', origin='upper', extent=extent_nm, interpolation='nearest', vmin=0, vmax=1)
    ax_bot[0].set_title("(A) Target Reference\n(Golden Target Layout)", fontsize=11, fontweight='bold', pad=12)

    ax_bot[1].imshow(st.session_state.current_mask, cmap='Purples', origin='upper', extent=extent_nm, interpolation='nearest', vmin=0, vmax=1)
    ax_bot[1].set_title("(B-2) After Mask (With OPC)\n(Extended Outer Serifs)", fontsize=11, fontweight='bold', pad=12)

    im_c1 = ax_bot[2].imshow(norm_exp_after, cmap='inferno', origin='upper', extent=extent_nm, interpolation='bilinear', vmin=0.0, vmax=1.0)
    ax_bot[2].set_title("(C-2) Aerial Image (After)\n(Corrected Intensity Profile)", fontsize=11, fontweight='bold', pad=12)
    overlay_target_outline(ax_bot[2])
    cbar1 = fig.colorbar(im_c1, ax=ax_bot[2], fraction=0.046, pad=0.04)
    cbar1.ax.tick_params(labelsize=9)

    diff_after = np.zeros((*target.shape, 3))
    diff_after[(target == 1) & (printed_after == 1)] = [0.1, 0.85, 0.1]
    diff_after[(target == 1) & (printed_after == 0)] = [0.95, 0.1, 0.1]
    diff_after[(target == 0) & (printed_after == 1)] = [0.1, 0.4, 0.95]
    ax_bot[3].imshow(diff_after, origin='upper', extent=extent_nm, interpolation='nearest')
    ax_bot[3].set_title(f"(D-2) Superposition (After)\nIoU: {iou_after:.2f}% ({error_a}px error)", fontsize=11, fontweight='bold', pad=12)
    overlay_target_outline(ax_bot[3])

    # 축 설정 공통화
    for ax in ax_top + ax_bot:
        ax.set_xlabel("X Position (nm)", fontsize=9.5, fontweight='bold')
        ax.set_ylabel("Y Position (nm)", fontsize=9.5, fontweight='bold')
        ax.xaxis.set_major_locator(MultipleLocator(major_x))
        ax.yaxis.set_major_locator(MultipleLocator(major_y))
        ax.tick_params(axis='both', which='major', labelsize=8.5)
        if show_grid:
            ax.xaxis.set_minor_locator(MultipleLocator(unit_scale))
            ax.yaxis.set_minor_locator(MultipleLocator(unit_scale))
            ax.grid(which='minor', color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
            ax.grid(which='major', color='black', linestyle='--', linewidth=0.7, alpha=0.6)

    return fig

st.subheader("Lithography & OPC Comprehensive Verification")
fig_result = generate_comprehensive_plot()
st.pyplot(fig_result)

# 11. 확장 마스크 기반 강력한 수율 보장 하이브리드 OPC 엔진
def run_iterative_opc(target_pattern, opt_kernel, opt_th, iterations=7):
    h, w = target_pattern.shape
    base = target_pattern.copy()

    # 1. 뼈대 코어 보호
    core_skeleton = binary_erosion(base, structure=np.ones((2, 2))).astype(float)
    mask = base.copy()

    # 2. 확장된 여백 영역으로 외곽 뿔(Outer Serif) 및 선 끝단 해머헤드 과감하게 돌출
    pad = np.pad(base, 1, mode='constant', constant_values=0)
    for r in range(h):
        for c in range(w):
            center = pad[r+1, c+1]
            if center == 0:
                neighbors_diag = [
                    (pad[r, c+1], pad[r+1, c]),
                    (pad[r, c+1], pad[r+1, c+2]),
                    (pad[r+2, c+1], pad[r+1, c]),
                    (pad[r+2, c+1], pad[r+1, c+2])
                ]
                # 외곽 볼록 모서리 주변에 세리프 부착
                if any(v1 == 1 and v2 == 1 for v1, v2 in neighbors_diag):
                    mask[r, c] = 1.0
            elif center == 1:
                # 선 끝단 밖으로 연장 (상하 여백으로 뻗어나감)
                up, down = pad[r, c+1], pad[r+2, c+1]
                if (up == 0 and down == 1) or (up == 1 and down == 0):
                    if r > 0 and pad[r, c+1] == 0:
                        mask[r-1, c] = 1.0
                    if r < h - 1 and pad[r+2, c+1] == 0:
                        mask[r+1, c] = 1.0

    # 3. 오차 기반 역피드백 루프 (확장 영역에서 안정적으로 수렴)
    diff_kernel = np.array([
        [0.2, 0.5, 0.2],
        [0.5, 1.0, 0.5],
        [0.2, 0.5, 0.2]
    ])
    diff_kernel = diff_kernel / np.sum(diff_kernel)

    cont_mask = mask.copy()
    for _ in range(iterations):
        bin_m = (cont_mask >= 0.45).astype(float)
        exp = convolve(bin_m, opt_kernel, mode='constant', cval=0.0)
        max_e = np.max(exp) if np.max(exp) > 0 else 1.0
        printed = ((exp / max_e) >= opt_th).astype(float)

        error = base - printed
        # 결손(빨강, +)을 우선 보강하고 과도 번짐(파랑, -)은 억제
        error_weighted = np.where(error > 0, error * 1.1, error * 0.9)

        grad = convolve(error_weighted, diff_kernel, mode='constant', cval=0.0)
        cont_mask += 0.5 * grad
        cont_mask = np.clip(cont_mask, 0.0, 1.0)
        cont_mask = np.maximum(cont_mask, core_skeleton)

    return (cont_mask >= 0.45).astype(float)

# 12. 마스크 제어 및 내보내기 영역
st.subheader("OPC Modification & Export")
ctrl_cols = st.columns(5 if layout_mode == "Interactive Grid Drawing" else 4)

with ctrl_cols[0]:
    if st.button("🔄 마스크 초기화 (Reset)"):
        st.session_state.current_mask = target.copy()
        st.rerun()

with ctrl_cols[1]:
    if st.button("⚡ 역노광 자동 OPC 실행 (ILT Loop)", type="primary"):
        with st.spinner("OPC 역보정 최적화 연산 중..."):
            optimized_mask = run_iterative_opc(target, kernel, threshold, iterations=7)
            st.session_state.current_mask = optimized_mask
            st.rerun()

with ctrl_cols[2]:
    mask_csv_buf = io.StringIO()
    np.savetxt(mask_csv_buf, st.session_state.current_mask.astype(int), delimiter=",", fmt="%d")
    st.download_button(
        label="📥 보정 마스크 다운로드 (CSV)",
        data=mask_csv_buf.getvalue(),
        file_name=f"{layout_name}_opc_mask.csv",
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
        file_name=f"{layout_name}_comparison_analysis.png",
        mime="image/png"
    )
