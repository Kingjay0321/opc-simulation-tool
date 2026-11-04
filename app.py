import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.ndimage import convolve, binary_dilation
import json
import io
import os

st.set_page_config(page_title="OPC Simulation Tool", layout="wide")
st.title("Optical Proximity Correction (OPC) Tool")
st.markdown("**7nm FinFET Standard Cell Metal1 Layout Simulation & Correction Platform**")
st.caption("Standard Unit: 1 Grid = 3.5 nm × 3.5 nm (λ = 3.5 nm, Metal1 Wire Width = 3 Grid = 10.5 nm)")

unit_scale = 3.5  # nm per pixel

# 1. 사이드바 설정
st.sidebar.header("Simulation Settings")
layout_mode = st.sidebar.radio("Layout Input Mode", ["Preset Standard Cells", "Interactive Grid Drawing", "CSV File Upload"])

# 2. 프리셋 레이아웃 정의
def get_inv_x1():
    mask = np.zeros((54, 18), dtype=float)
    mask[0:3, 0:18] = 1.0
    mask[51:54, 0:18] = 1.0
    mask[3:21, 3:6] = 1.0
    mask[33:51, 3:6] = 1.0
    mask[18:36, 12:15] = 1.0
    return mask

def get_nand2_x4():
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

# 양방향 통신 컴포넌트 생성 함수
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
      font-family: sans-serif;
      background: #f8f9fa;
      margin: 0;
      padding: 8px;
    }
    .toolbar {
      display: flex;
      gap: 6px;
      align-items: center;
      margin-bottom: 10px;
      flex-wrap: wrap;
    }
    .btn {
      padding: 6px 12px;
      border: 2px outset #ffffff;
      background: #e0e0e0;
      font-weight: bold;
      font-size: 11px;
      cursor: pointer;
      border-radius: 3px;
    }
    .btn:active {
      border-style: inset;
    }
    .btn.active {
      background: #1a4b8c;
      color: white;
      border: 2px inset #0d2b56;
    }
    .btn-run {
      background: #2e7d32;
      color: white;
      border: 2px outset #1b5e20;
    }
    .btn-run:hover {
      background: #1b5e20;
    }
    #grid-box {
      display: grid;
      gap: 1px;
      background-color: #7b7b7b;
      padding: 4px;
      border: 3px inset #c0c0c0;
      width: fit-content;
      max-height: 480px;
      overflow: auto;
    }
    .cell {
      width: 16px;
      height: 16px;
      background-color: #dcdcdc;
      border-top: 1px solid #ffffff;
      border-left: 1px solid #ffffff;
      border-right: 1px solid #999;
      border-bottom: 1px solid #999;
      box-sizing: border-box;
      cursor: crosshair;
    }
    .cell.filled {
      background-color: #1a4b8c !important;
      border: 1px solid #0d2b56 !important;
    }
  </style>
</head>
<body>
  <div class="toolbar">
    <button id="drawBtn" class="btn active" onclick="setMode(1)">✏️ 그리기 (1)</button>
    <button id="eraseBtn" class="btn" onclick="setMode(0)">🧹 지우개 (0)</button>
    <button id="toggleBtn" class="btn" onclick="setMode(2)">🔄 토글 모드</button>
    <button class="btn" onclick="clearGrid()">🗑️ 전체 초기화</button>
    <button class="btn btn-run" onclick="submitToSimulation()">⚡ 시뮬레이션 연산 실행 (Apply)</button>
    <span id="stat-text" style="font-size:11px; font-weight:bold; color:#444; margin-left:6px;"></span>
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
      sendMsg("streamlit:setFrameHeight", { height: Math.min(540, rows * 18 + 70) });
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
    target = get_inv_x1() if preset_choice == "INV_X1" else get_nand2_x4()
    layout_name = preset_choice

elif layout_mode == "Interactive Grid Drawing":
    st.sidebar.subheader("Canvas Grid Size")
    grid_w = st.sidebar.number_input("Grid Width (X, columns)", min_value=5, max_value=40, value=20, step=1)
    grid_h = st.sidebar.number_input("Grid Height (Y, rows)", min_value=5, max_value=60, value=20, step=1)
    st.sidebar.caption("📏 **최대 허용 크기:** Max X: 40 cols, Max Y: 60 rows (단위: 1 grid = 3.5 nm)")

    st.subheader("Minesweeper-Style Pixel Grid Editor")
    st.markdown("💡 **격자판에서 마우스로 드래그하여 배선을 그리고, [⚡ 시뮬레이션 연산 실행]을 누르면 연산이 시작됩니다.**")

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
                st.session_state.current_mask = np.zeros((grid_h, grid_w), dtype=float)
                st.rerun()
            elif action == "apply":
                st.session_state.current_mask = np.array(mat, dtype=float)

    target = np.array(st.session_state[matrix_key], dtype=float)
    layout_name = f"Custom_{grid_w}x{grid_h}"

else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV (0 and 1 matrix)", type=["csv"])
    if uploaded_file is not None:
        target = np.genfromtxt(uploaded_file, delimiter=",").astype(float)
        target = np.nan_to_num(target, nan=0.0)
    else:
        target = np.zeros((30, 30), dtype=float)
        target[10:20, 10:20] = 1.0
    layout_name = "Uploaded_CSV"

# 4. 세션 마스크 상태 관리
if "current_mask" not in st.session_state or st.session_state.get("last_layout") != layout_name:
    st.session_state.current_mask = target.copy()
    st.session_state.last_layout = layout_name

if st.session_state.current_mask.shape != target.shape:
    st.session_state.current_mask = target.copy()

# 5. 광학 공정 파라미터 및 치수선 옵션
st.sidebar.subheader("Optical Lithography Parameters")
kernel_type = st.sidebar.radio("Optical Blur Kernel", ["Standard 5x5 Matrix", "2D Gaussian Function"])
sigma = st.sidebar.slider("Gaussian Diffusion Radius (sigma, px)", 1.0, 3.0, 1.8, 0.1)
threshold = st.sidebar.slider("Resist Threshold (I_th)", 0.20, 0.80, 0.55, 0.01)

st.sidebar.subheader("Display Options")
show_grid = st.sidebar.checkbox("Display Unit Grid (3.5 nm)", value=True)
show_dimension = st.sidebar.checkbox("Display CAD Dimension Arrows (치수선)", value=True)

# 커널 정의
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
    ax = np.linspace(-(k_size // 2), k_size // 2, k_size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    kernel = kernel / np.sum(kernel)

# 6. 광학 노광 및 현상 연산
exposure = convolve(st.session_state.current_mask, kernel, mode='constant', cval=0.0)
max_val = np.max(exposure) if np.max(exposure) > 0 else 1.0
norm_exposure = exposure / max_val
printed = (norm_exposure >= threshold).astype(float)

# 정량적 지표 (IoU)
intersection = np.logical_and(target == 1, printed == 1).sum()
union = np.logical_or(target == 1, printed == 1).sum()
iou = (intersection / union) * 100 if union > 0 else 0.0
error_pixels = int(union - intersection)

h_px, w_px = target.shape
width_nm = w_px * unit_scale
height_nm = h_px * unit_scale
extent_nm = [0, width_nm, height_nm, 0]
aspect_ratio = h_px / w_px

# 메트릭 표시
st.subheader("Quantitative Evaluation")
c_m1, c_m2, c_m3 = st.columns(3)
c_m1.metric("Grid Dimension", f"{w_px} × {h_px} px")
c_m2.metric("Physical Size", f"{width_nm:.1f} × {height_nm:.1f} nm")
c_m3.metric("Pattern Fidelity (IoU)", f"{iou:.2f}%", delta=f"{error_pixels} px Error", delta_color="inverse")

# 7. CAD 치수 화살표 렌더링 함수
def draw_dimension_arrows(ax):
    if not show_dimension:
        return

    arrow_props = dict(arrowstyle='<->', color='#1a4b8c', lw=1.2)

    if layout_name == "INV_X1":
        ax.annotate('', xy=(0, -8), xytext=(63, -8), arrowprops=arrow_props, annotation_clip=False)
        ax.text(31.5, -10, '63 nm (18λ)', color='#1a4b8c', fontsize=9, fontweight='bold', ha='center', va='bottom', clip_on=False)

        ax.annotate('', xy=(68, 0), xytext=(68, 189), arrowprops=arrow_props, annotation_clip=False)
        ax.text(71, 94.5, '189 nm (54λ)', color='#1a4b8c', fontsize=9, fontweight='bold', ha='left', va='center', rotation=-90, clip_on=False)

        ax.annotate('', xy=(56, 63), xytext=(56, 126), arrowprops=arrow_props, annotation_clip=False)
        ax.text(58, 94.5, '63 nm*', color='#1a4b8c', fontsize=8, fontweight='bold', ha='left', va='center', rotation=-90, clip_on=False)

        ax.annotate('', xy=(21, 136), xytext=(42, 136), arrowprops=arrow_props, annotation_clip=False)
        ax.text(31.5, 134, '21 nm*', color='#1a4b8c', fontsize=9, fontweight='bold', ha='center', va='bottom', clip_on=False)

    elif layout_name == "NAND2_X4":
        ax.annotate('', xy=(0, -8), xytext=(157.5, -8), arrowprops=arrow_props, annotation_clip=False)
        ax.text(78.75, -10, '157.5 nm (45λ)', color='#1a4b8c', fontsize=9, fontweight='bold', ha='center', va='bottom', clip_on=False)

        ax.annotate('', xy=(164, 0), xytext=(164, 189), arrowprops=arrow_props, annotation_clip=False)
        ax.text(167, 94.5, '189 nm (54λ)', color='#1a4b8c', fontsize=9, fontweight='bold', ha='left', va='center', rotation=-90, clip_on=False)

        ax.annotate('', xy=(10.5, 172), xytext=(42, 172), arrowprops=arrow_props, annotation_clip=False)
        ax.text(26.25, 170, '31.5 nm*', color='#1a4b8c', fontsize=9, fontweight='bold', ha='center', va='bottom', clip_on=False)

    else:
        offset_top = max(6.0, height_nm * 0.05)
        offset_right = max(6.0, width_nm * 0.05)

        ax.annotate('', xy=(0, -offset_top), xytext=(width_nm, -offset_top), arrowprops=arrow_props, annotation_clip=False)
        ax.text(width_nm / 2.0, -(offset_top + 2.0), f'{width_nm:.1f} nm ({w_px}λ)',
                color='#1a4b8c', fontsize=9, fontweight='bold', ha='center', va='bottom', clip_on=False)

        ax.annotate('', xy=(width_nm + offset_right, 0), xytext=(width_nm + offset_right, height_nm), arrowprops=arrow_props, annotation_clip=False)
        ax.text(width_nm + offset_right + 2.0, height_nm / 2.0, f'{height_nm:.1f} nm ({h_px}λ)',
                color='#1a4b8c', fontsize=9, fontweight='bold', ha='left', va='center', rotation=-90, clip_on=False)

# 8. 하단 4개 플롯 최적 동적 배치
def generate_analysis_plot():
    is_tall = aspect_ratio >= 1.4
    if is_tall:
        fig, axes_flat = plt.subplots(1, 4, figsize=(22, 7), constrained_layout=True)
    else:
        fig, axes = plt.subplots(2, 2, figsize=(13, 13), constrained_layout=True)
        axes_flat = axes.flatten()

    # (A) Target Pattern
    axes_flat[0].imshow(target, cmap='Blues', origin='upper', extent=extent_nm, interpolation='nearest', vmin=0, vmax=1)
    axes_flat[0].set_title("(A) Target Pattern", fontsize=12, fontweight='bold')
    draw_dimension_arrows(axes_flat[0])

    # (B) Current Mask
    axes_flat[1].imshow(st.session_state.current_mask, cmap='Purples', origin='upper', extent=extent_nm, interpolation='nearest', vmin=0, vmax=1)
    axes_flat[1].set_title("(B) Current Mask (Input)", fontsize=12, fontweight='bold')

    # (C) Aerial Image
    im_c = axes_flat[2].imshow(norm_exposure, cmap='inferno', origin='upper', extent=extent_nm, interpolation='bilinear', vmin=0.0, vmax=1.0)
    axes_flat[2].set_title("(C) Aerial Image Intensity", fontsize=12, fontweight='bold')
    plt.colorbar(im_c, ax=axes_flat[2], fraction=0.046, pad=0.04)

    # (D) Superposition Analysis
    diff_view = np.zeros((*target.shape, 3))
    diff_view[(target == 1) & (printed == 1)] = [0.1, 0.85, 0.1]
    diff_view[(target == 1) & (printed == 0)] = [0.95, 0.1, 0.1]
    diff_view[(target == 0) & (printed == 1)] = [0.1, 0.4, 0.95]

    axes_flat[3].imshow(diff_view, origin='upper', extent=extent_nm, interpolation='nearest')
    axes_flat[3].set_title("(D) Superposition (G: Match, R: Under, B: Over)", fontsize=11, fontweight='bold')

    major_x = 21.0 if width_nm > 40 else 10.5
    major_y = 31.5 if height_nm > 60 else 10.5

    for ax in axes_flat:
        ax.set_xlabel("X (nm)", fontsize=10)
        ax.set_ylabel("Y (nm)", fontsize=10)
        ax.xaxis.set_major_locator(MultipleLocator(major_x))
        ax.yaxis.set_major_locator(MultipleLocator(major_y))
        if show_grid:
            ax.xaxis.set_minor_locator(MultipleLocator(unit_scale))
            ax.yaxis.set_minor_locator(MultipleLocator(unit_scale))
            ax.grid(which='minor', color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
            ax.grid(which='major', color='black', linestyle='--', linewidth=0.7, alpha=0.6)

    return fig

st.subheader("Lithography & OPC Analysis Result")
fig_result = generate_analysis_plot()
st.pyplot(fig_result)

# 9. 실전 역노광(ILT) + 모서리 세리프 하이브리드 최적화 엔진
def run_iterative_opc(target_pattern, opt_kernel, opt_th, iterations=8):
    h, w = target_pattern.shape
    # 연속 마스크 변수 (0.0 ~ 1.0)
    cont_mask = target_pattern.copy()

    # 1단계: 선 끝단(Line-End) 및 볼록 코너(Outer Corner) 기하학적 초기 세리프 부착
    pad = np.pad(target_pattern, 1, mode='constant', constant_values=0)
    for r in range(h):
        for c in range(w):
            center = pad[r+1, c+1]
            if center == 0:
                # 상하좌우 및 대각 성분 검사로 볼록 코너에 뿔(Serif) 배치
                up = pad[r, c+1]
                down = pad[r+2, c+1]
                left = pad[r+1, c]
                right = pad[r+1, c+2]
                if (up == 1 and left == 1) or (up == 1 and right == 1) or \
                   (down == 1 and left == 1) or (down == 1 and right == 1):
                    cont_mask[r, c] += 0.6
            elif center == 1:
                # 선 끝단(Line-End)에 해머헤드(Hammerhead) 확장 유도
                up = pad[r, c+1]
                down = pad[r+2, c+1]
                if (up == 0 and down == 1) or (up == 1 and down == 0):
                    if c > 0:
                        cont_mask[r, c-1] += 0.5
                    if c < w - 1:
                        cont_mask[r, c+1] += 0.5

    # 2단계: 역노광(Inverse Lithography) 그래디언트 디센트 피드백 루프
    # 에러 E = Target - Printed 를 광학 커널 K 로 역확산(Adjoint Convolve)시켜 마스크 외곽으로 살을 찌움
    diff_kernel = np.array([
        [0.2, 0.5, 0.2],
        [0.5, 1.0, 0.5],
        [0.2, 0.5, 0.2]
    ])
    diff_kernel = diff_kernel / np.sum(diff_kernel)

    for it in range(iterations):
        # 현재 마스크 상태로 노광 시뮬레이션
        bin_mask = (cont_mask >= 0.40).astype(float)
        exp = convolve(bin_mask, opt_kernel, mode='constant', cval=0.0)
        max_e = np.max(exp) if np.max(exp) > 0 else 1.0
        printed_curr = ((exp / max_e) >= opt_th).astype(float)

        # 오차 계산: (빨간색 = +1, 파란색 = -1)
        error = target_pattern - printed_curr

        # 핵심: 오차 신호를 인접 픽셀로 확산시켜 마스크 경계 바깥/안쪽으로 그래디언트 전달!
        gradient = convolve(error, diff_kernel, mode='constant', cval=0.0)

        # 마스크 업데이트 (빨간 결손부는 밖으로 커지고, 파란 번짐부는 안쪽으로 깎임)
        cont_mask += 0.7 * gradient
        cont_mask = np.clip(cont_mask, 0.0, 1.2)

    # 3단계: 뚜렷한 외곽 형상을 갖춘 이진 마스크 생성
    final_opc_mask = (cont_mask >= 0.40).astype(float)
    return final_opc_mask

# 10. 마스크 제어 및 내보내기 파트
st.subheader("OPC Modification & Export")
ctrl_cols = st.columns(5 if layout_mode == "Interactive Grid Drawing" else 4)

with ctrl_cols[0]:
    if st.button("Reset Mask to Target"):
        st.session_state.current_mask = target.copy()
        st.rerun()

with ctrl_cols[1]:
    if st.button("⚡ 역노광 최적화 자동 OPC 실행 (ILT Loop)", type="primary"):
        with st.spinner("OPC 역보정 최적화 연산 중..."):
            optimized_mask = run_iterative_opc(target, kernel, threshold, iterations=8)
            st.session_state.current_mask = optimized_mask
            st.rerun()

with ctrl_cols[2]:
    mask_csv_buf = io.StringIO()
    np.savetxt(mask_csv_buf, st.session_state.current_mask.astype(int), delimiter=",", fmt="%d")
    st.download_button(
        label="Download Mask (CSV)",
        data=mask_csv_buf.getvalue(),
        file_name=f"{layout_name}_mask.csv",
        mime="text/csv"
    )

if layout_mode == "Interactive Grid Drawing":
    with ctrl_cols[3]:
        drawn_csv_buf = io.StringIO()
        np.savetxt(drawn_csv_buf, target.astype(int), delimiter=",", fmt="%d")
        st.download_button(
            label="Download Drawn Target (CSV)",
            data=drawn_csv_buf.getvalue(),
            file_name=f"{layout_name}_drawn_target.csv",
            mime="text/csv"
        )
    btn_col_idx = 4
else:
    btn_col_idx = 3

with ctrl_cols[btn_col_idx]:
    img_buf = io.BytesIO()
    fig_result.savefig(img_buf, format='png', dpi=300, bbox_inches='tight')
    st.download_button(
        label="Download Analysis Plot (PNG)",
        data=img_buf.getvalue(),
        file_name=f"{layout_name}_analysis_plot.png",
        mime="image/png"
    )
