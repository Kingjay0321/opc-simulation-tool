import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.ndimage import convolve
import json
import io
import os

st.set_page_config(page_title="OPC Simulation Tool", layout="wide")
st.title("Optical Proximity Correction (OPC) Tool")
st.markdown("**7nm FinFET Standard Cell Metal1 Layout Simulation & Verification Platform**")
st.caption("Standard Unit: 1 Grid = 3.5 nm × 3.5 nm (λ = 3.5 nm, Metal1 Wire Width = 3 Grid = 10.5 nm)")

unit_scale = 3.5  # nm per pixel

# 1. 사이드바 설정
st.sidebar.header("Simulation Settings")
layout_mode = st.sidebar.radio("Layout Input Mode", ["Preset Standard Cells", "Interactive Grid Drawing", "CSV File Upload"])

# 2. 프리셋 레이아웃 정의
def get_inv_x1():
    mask = np.zeros((54, 18), dtype=float)
    mask[0:3, 0:18] = 1.0     # 상단 바 (y: 0~10.5 nm)
    mask[51:54, 0:18] = 1.0   # 하단 바 (y: 178.5~189 nm)
    mask[3:21, 3:6] = 1.0    # 좌상단 세로선
    mask[33:51, 3:6] = 1.0   # 좌하단 세로선
    mask[18:36, 12:15] = 1.0 # 우측 중앙 세로선
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

# 양방향 통신 컴포넌트 생성 (모던 UI 적용)
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
    target = get_inv_x1() if preset_choice == "INV_X1" else get_nand2_x4()
    layout_name = preset_choice

elif layout_mode == "Interactive Grid Drawing":
    st.sidebar.subheader("Canvas Grid Size")
    grid_w = st.sidebar.number_input("Grid Width (X, columns)", min_value=5, max_value=40, value=20, step=1)
    grid_h = st.sidebar.number_input("Grid Height (Y, rows)", min_value=5, max_value=60, value=20, step=1)
    st.sidebar.caption("📏 **최대 허용 크기:** Max X: 40 cols, Max Y: 60 rows (단위: 1 grid = 3.5 nm)")

    st.subheader("Minesweeper-Style Pixel Grid Editor")
    st.markdown("💡 **격자판에서 마우스 드래그로 배선을 그리고, [⚡ 시뮬레이션 연산 적용]을 누르면 연산이 시작됩니다.**")

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
sigma = st.sidebar.slider("Gaussian Diffusion Radius (sigma, px)", 1.0, 3.5, 2.2, 0.1)
threshold = st.sidebar.slider("Resist Threshold (I_th)", 0.20, 0.80, 0.55, 0.01)

st.sidebar.subheader("Display Options")
show_grid = st.sidebar.checkbox("Display Unit Grid (3.5 nm)", value=True)
show_dimension = st.sidebar.checkbox("Display CAD Dimension Arrows (치수선)", value=True)
show_outline = st.sidebar.checkbox("Display Target White Dashed Outline (목표 외곽선)", value=True)

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

# 6. 정량적 전/후(Before vs After) 듀얼 시뮬레이션 연산
# (1) 보정 전(Before OPC): 타겟 패턴을 그대로 마스크로 사용
exp_before = convolve(target, kernel, mode='constant', cval=0.0)
max_eb = np.max(exp_before) if np.max(exp_before) > 0 else 1.0
norm_exp_before = exp_before / max_eb
printed_before = (norm_exp_before >= threshold).astype(float)

inter_b = np.logical_and(target == 1, printed_before == 1).sum()
union_b = np.logical_or(target == 1, printed_before == 1).sum()
iou_before = (inter_b / union_b) * 100.0 if union_b > 0 else 0.0
error_b = int(union_b - inter_b)

# (2) 보정 후(After OPC): 현재 세션의 마스크 사용
exp_after = convolve(st.session_state.current_mask, kernel, mode='constant', cval=0.0)
max_ea = np.max(exp_after) if np.max(exp_after) > 0 else 1.0
norm_exp_after = exp_after / max_ea
printed_after = (norm_exp_after >= threshold).astype(float)

inter_a = np.logical_and(target == 1, printed_after == 1).sum()
union_a = np.logical_or(target == 1, printed_after == 1).sum()
iou_after = (inter_a / union_a) * 100.0 if union_a > 0 else 0.0
error_a = int(union_a - inter_a)

# 개선율 지표 산출
iou_delta = iou_after - iou_before
error_delta = error_b - error_a  # 양수면 에러 감소

h_px, w_px = target.shape
width_nm = w_px * unit_scale
height_nm = h_px * unit_scale
extent_nm = [0, width_nm, height_nm, 0]
aspect_ratio = h_px / w_px

# 7. 메트릭 대시보드 (과제 마 항목 정량적 비교 충족)
st.subheader("Quantitative Evaluation & OPC Improvement (과제 결과 검증)")
m_col1, m_col2, m_col3, m_col4 = st.columns(4)
m_col1.metric("레이아웃 실제 치수", f"{width_nm:.1f} × {height_nm:.1f} nm", f"{w_px} × {h_px} Grid")
m_col2.metric("보정 전 일치도 (Before IoU)", f"{iou_before:.2f}%", f"{error_b} px 오차", delta_color="off")
m_col3.metric("보정 후 일치도 (After IoU)", f"{iou_after:.2f}%", f"{iou_delta:+.2f}% p 향상", delta_color="normal")
m_col4.metric("오차 픽셀 감소량", f"{error_delta} px 감소", f"개선도: {(iou_delta/iou_before*100) if iou_before>0 else 0:+.1f}%", delta_color="normal")

# 8. CAD 치수 화살표 렌더링 함수
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

# 9. 목표 패턴 흰색 점선 외곽선 오버레이 헬퍼
def overlay_target_outline(ax):
    if not show_outline:
        return
    # Matplotlib contour로 0.5 임계 경계선을 흰색 점선으로 추출
    ax.contour(target, levels=[0.5], colors=['#ffffff'], linestyles=['--'], linewidths=[1.3],
               extent=[0, width_nm, 0, height_nm], origin='upper')

# 10. 종합 전/후 비교 분석 플롯 생성 (과제 보고서용 종합 플롯)
def generate_comprehensive_plot():
    # 2개 행 구조: [Row 0: Before OPC 분석] / [Row 1: After OPC 분석]
    fig, axes = plt.subplots(2, 4, figsize=(22, 12), constrained_layout=True)

    major_x = 21.0 if width_nm > 40 else 10.5
    major_y = 31.5 if height_nm > 60 else 10.5

    # ------------------ Row 0: Before OPC ------------------
    # (0,0) Target Pattern
    axes[0, 0].imshow(target, cmap='Blues', origin='upper', extent=extent_nm, interpolation='nearest', vmin=0, vmax=1)
    axes[0, 0].set_title("(A) Target Pattern\n(목표 설계)", fontsize=11, fontweight='bold')
    draw_dimension_arrows(axes[0, 0])

    # (0,1) Before Mask (Raw Target)
    axes[0, 1].imshow(target, cmap='Purples', origin='upper', extent=extent_nm, interpolation='nearest', vmin=0, vmax=1)
    axes[0, 1].set_title("(B-1) Before Mask (No OPC)\n(보정 전 원본 마스크)", fontsize=11, fontweight='bold')

    # (0,2) Before Aerial Image
    im_c0 = axes[0, 2].imshow(norm_exp_before, cmap='inferno', origin='upper', extent=extent_nm, interpolation='bilinear', vmin=0.0, vmax=1.0)
    axes[0, 2].set_title("(C-1) Aerial Image (Before)\n(보정 전 노광 분포)", fontsize=11, fontweight='bold')
    overlay_target_outline(axes[0, 2])
    plt.colorbar(im_c0, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # (0,3) Before Superposition
    diff_before = np.zeros((*target.shape, 3))
    diff_before[(target == 1) & (printed_before == 1)] = [0.1, 0.85, 0.1]  # 일치 (초록)
    diff_before[(target == 1) & (printed_before == 0)] = [0.95, 0.1, 0.1]  # 결손 (빨강)
    diff_before[(target == 0) & (printed_before == 1)] = [0.1, 0.4, 0.95]  # 번짐 (파랑)
    axes[0, 3].imshow(diff_before, origin='upper', extent=extent_nm, interpolation='nearest')
    axes[0, 3].set_title(f"(D-1) Superposition (Before)\nIoU: {iou_before:.2f}% ({error_b}px error)", fontsize=11, fontweight='bold')
    overlay_target_outline(axes[0, 3])

    # ------------------ Row 1: After OPC ------------------
    # (1,0) Target Reference
    axes[1, 0].imshow(target, cmap='Blues', origin='upper', extent=extent_nm, interpolation='nearest', vmin=0, vmax=1)
    axes[1, 0].set_title("(A) Target Reference\n(비교 기준 도면)", fontsize=11, fontweight='bold')

    # (1,1) After Mask (Optimized)
    axes[1, 1].imshow(st.session_state.current_mask, cmap='Purples', origin='upper', extent=extent_nm, interpolation='nearest', vmin=0, vmax=1)
    axes[1, 1].set_title("(B-2) After Mask (OPC Applied)\n(세리프/해머헤드 보정 마스크)", fontsize=11, fontweight='bold')

    # (1,2) After Aerial Image
    im_c1 = axes[1, 2].imshow(norm_exp_after, cmap='inferno', origin='upper', extent=extent_nm, interpolation='bilinear', vmin=0.0, vmax=1.0)
    axes[1, 2].set_title("(C-2) Aerial Image (After)\n(보정 후 노광 분포)", fontsize=11, fontweight='bold')
    overlay_target_outline(axes[1, 2])
    plt.colorbar(im_c1, ax=axes[1, 2], fraction=0.046, pad=0.04)

    # (1,3) After Superposition
    diff_after = np.zeros((*target.shape, 3))
    diff_after[(target == 1) & (printed_after == 1)] = [0.1, 0.85, 0.1]  # 일치 (초록)
    diff_after[(target == 1) & (printed_after == 0)] = [0.95, 0.1, 0.1]  # 결손 (빨강)
    diff_after[(target == 0) & (printed_after == 1)] = [0.1, 0.4, 0.95]  # 번짐 (파랑)
    axes[1, 3].imshow(diff_after, origin='upper', extent=extent_nm, interpolation='nearest')
    axes[1, 3].set_title(f"(D-2) Superposition (After)\nIoU: {iou_after:.2f}% ({error_a}px error)", fontsize=11, fontweight='bold')
    overlay_target_outline(axes[1, 3])

    # 축 설정 공통 적용
    for ax in axes.flat:
        ax.set_xlabel("X (nm)", fontsize=9)
        ax.set_ylabel("Y (nm)", fontsize=9)
        ax.xaxis.set_major_locator(MultipleLocator(major_x))
        ax.yaxis.set_major_locator(MultipleLocator(major_y))
        if show_grid:
            ax.xaxis.set_minor_locator(MultipleLocator(unit_scale))
            ax.yaxis.set_minor_locator(MultipleLocator(unit_scale))
            ax.grid(which='minor', color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
            ax.grid(which='major', color='black', linestyle='--', linewidth=0.7, alpha=0.6)

    return fig

st.subheader("Lithography & OPC Comprehensive Verification (전·후 비교 검증)")
fig_result = generate_comprehensive_plot()
st.pyplot(fig_result)

# 11. 역노광(ILT) + 모서리 세리프 최적화 엔진
def run_iterative_opc(target_pattern, opt_kernel, opt_th, iterations=8):
    h, w = target_pattern.shape
    cont_mask = target_pattern.copy()

    # 기하학적 초기 세리프 및 선 끝단 해머헤드 유도
    pad = np.pad(target_pattern, 1, mode='constant', constant_values=0)
    for r in range(h):
        for c in range(w):
            center = pad[r+1, c+1]
            if center == 0:
                up, down, left, right = pad[r, c+1], pad[r+2, c+1], pad[r+1, c], pad[r+1, c+2]
                if (up == 1 and left == 1) or (up == 1 and right == 1) or \
                   (down == 1 and left == 1) or (down == 1 and right == 1):
                    cont_mask[r, c] += 0.6
            elif center == 1:
                up, down = pad[r, c+1], pad[r+2, c+1]
                if (up == 0 and down == 1) or (up == 1 and down == 0):
                    if c > 0:
                        cont_mask[r, c-1] += 0.5
                    if c < w - 1:
                        cont_mask[r, c+1] += 0.5

    diff_kernel = np.array([
        [0.2, 0.5, 0.2],
        [0.5, 1.0, 0.5],
        [0.2, 0.5, 0.2]
    ])
    diff_kernel = diff_kernel / np.sum(diff_kernel)

    for it in range(iterations):
        bin_mask = (cont_mask >= 0.40).astype(float)
        exp = convolve(bin_mask, opt_kernel, mode='constant', cval=0.0)
        max_e = np.max(exp) if np.max(exp) > 0 else 1.0
        printed_curr = ((exp / max_e) >= opt_th).astype(float)

        error = target_pattern - printed_curr
        gradient = convolve(error, diff_kernel, mode='constant', cval=0.0)
        cont_mask += 0.7 * gradient
        cont_mask = np.clip(cont_mask, 0.0, 1.2)

    return (cont_mask >= 0.40).astype(float)

# 12. 컨트롤 및 내보내기 버튼 영역 (모던 스타일 적용)
st.subheader("OPC Modification & Export")
ctrl_cols = st.columns(5 if layout_mode == "Interactive Grid Drawing" else 4)

with ctrl_cols[0]:
    if st.button("🔄 마스크 초기화 (Reset)"):
        st.session_state.current_mask = target.copy()
        st.rerun()

with ctrl_cols[1]:
    if st.button("⚡ 역노광 자동 OPC 실행 (ILT Loop)", type="primary"):
        with st.spinner("OPC 역보정 최적화 연산 중..."):
            optimized_mask = run_iterative_opc(target, kernel, threshold, iterations=8)
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
        np.savetxt(drawn_csv_buf, target.astype(int), delimiter=",", fmt="%d")
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
