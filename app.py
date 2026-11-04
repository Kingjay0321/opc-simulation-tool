import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.ndimage import convolve
import json
import io

st.set_page_config(page_title="OPC Simulation Tool", layout="wide")
st.title("Optical Proximity Correction (OPC) Tool")
st.markdown("**7nm FinFET Standard Cell Metal1 Layout Simulation & Correction Platform**")
st.caption("Standard Unit: 1 Grid = 3.5 nm × 3.5 nm (λ = 3.5 nm, Metal1 Wire Width = 3 Grid = 10.5 nm)")

unit_scale = 3.5  # nm/pixel

# 1. 사이드바 설정
st.sidebar.header("Simulation Settings")
layout_mode = st.sidebar.radio("Layout Input Mode", ["Preset Standard Cells", "Interactive Grid Drawing", "CSV File Upload"])

# 2. 레이아웃 데이터 함수
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

# 3. 레이아웃 로드 분기
if layout_mode == "Preset Standard Cells":
    preset_choice = st.sidebar.selectbox("Select Target Cell", ["INV_X1", "NAND2_X4"])
    target = get_inv_x1() if preset_choice == "INV_X1" else get_nand2_x4()
    layout_name = preset_choice

elif layout_mode == "Interactive Grid Drawing":
    st.sidebar.subheader("Canvas Grid Size")
    grid_w = st.sidebar.number_input("Grid Width (X, columns)", min_value=5, max_value=40, value=18, step=1)
    grid_h = st.sidebar.number_input("Grid Height (Y, rows)", min_value=5, max_value=60, value=30, step=1)

    st.subheader("Minesweeper-Style Pixel Grid Editor")
    st.markdown("💡 **마우스 왼쪽 버튼을 누른 채로 쓱 드래그하면 연속으로 빠르게 칠해집니다.**")

    # 세션 상태에 사용자 드로잉 행렬 보관
    canvas_session_key = f"drawn_matrix_{grid_w}_{grid_h}"
    if canvas_session_key not in st.session_state:
        st.session_state[canvas_session_key] = [[0 for _ in range(grid_w)] for _ in range(grid_h)]

    initial_matrix_json = json.dumps(st.session_state[canvas_session_key])

    # HTML5/JS 드래그 픽셀 그리드 컴포넌트
    custom_canvas_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <style>
        body {{
          user-select: none;
          font-family: monospace, sans-serif;
          background: #f0f2f6;
          margin: 0;
          padding: 8px;
        }}
        .controls {{
          margin-bottom: 10px;
          display: flex;
          gap: 10px;
          align-items: center;
        }}
        .btn {{
          padding: 6px 14px;
          border: 2px outset #ffffff;
          background: #d4d0c8;
          font-weight: bold;
          cursor: pointer;
          border-radius: 3px;
        }}
        .btn:active {{
          border-style: inset;
        }}
        .btn.active {{
          background: #0044aa;
          color: white;
          border-style: inset;
        }}
        #grid-container {{
          display: grid;
          grid-template-columns: repeat({grid_w}, 16px);
          grid-template-rows: repeat({grid_h}, 16px);
          gap: 1px;
          background-color: #808080;
          padding: 4px;
          border: 3px inset #c0c0c0;
          width: fit-content;
          max-height: 520px;
          overflow: auto;
        }}
        .cell {{
          width: 16px;
          height: 16px;
          background-color: #c0c0c0;
          border-top: 1px solid #ffffff;
          border-left: 1px solid #ffffff;
          border-right: 1px solid #808080;
          border-bottom: 1px solid #808080;
          box-sizing: border-box;
          cursor: crosshair;
        }}
        .cell.filled {{
          background-color: #1a4b8c !important;
          border: 1px solid #0d2b56 !important;
        }}
      </style>
    </head>
    <body>
      <div class="controls">
        <button id="drawBtn" class="btn active" onclick="setMode(1)">✏️ 배선 그리기 (1)</button>
        <button id="eraseBtn" class="btn" onclick="setMode(0)">🧹 지우개 (0)</button>
        <button class="btn" onclick="clearGrid()">🔄 전체 리셋</button>
        <span style="font-size:12px; color:#555;">크기: {grid_w} × {grid_h} 칸</span>
      </div>
      <div id="grid-container"></div>
      
      <script>
        let isDrawing = false;
        let drawMode = 1; // 1: draw, 0: erase
        const rows = {grid_h};
        const cols = {grid_w};
        let gridData = {initial_matrix_json};

        const container = document.getElementById('grid-container');

        function createGrid() {{
          container.innerHTML = '';
          for (let r = 0; r < rows; r++) {{
            for (let c = 0; c < cols; c++) {{
              const cell = document.createElement('div');
              cell.classList.add('cell');
              cell.dataset.r = r;
              cell.dataset.c = c;
              if (gridData[r] && gridData[r][c] === 1) {{
                cell.classList.add('filled');
              }}

              cell.addEventListener('mousedown', (e) => {{
                e.preventDefault();
                isDrawing = true;
                applyState(cell, r, c);
              }});

              cell.addEventListener('mouseenter', () => {{
                if (isDrawing) {{
                  applyState(cell, r, c);
                }}
              }});

              container.appendChild(cell);
            }}
          }}
        }}

        function applyState(cell, r, c) {{
          if (drawMode === 1) {{
            cell.classList.add('filled');
            gridData[r][c] = 1;
          }} else {{
            cell.classList.remove('filled');
            gridData[r][c] = 0;
          }}
          sendDataToStreamlit();
        }}

        window.addEventListener('mouseup', () => {{
          isDrawing = false;
        }});

        function setMode(mode) {{
          drawMode = mode;
          document.getElementById('drawBtn').classList.toggle('active', mode === 1);
          document.getElementById('eraseBtn').classList.toggle('active', mode === 0);
        }}

        function clearGrid() {{
          for (let r = 0; r < rows; r++) {{
            for (let c = 0; c < cols; c++) {{
              gridData[r][c] = 0;
            }}
          }}
          document.querySelectorAll('.cell').forEach(el => el.classList.remove('filled'));
          sendDataToStreamlit();
        }}

        function sendDataToStreamlit() {{
          const strData = JSON.stringify(gridData);
          window.parent.postMessage({{
            type: 'streamlit:setComponentValue',
            value: strData
          }}, '*');
        }}

        createGrid();
      </script>
    </body>
    </html>
    """

    col_canvas, col_info = st.columns([1.3, 1])
    with col_canvas:
        drawn_json_str = components.html(custom_canvas_html, height=min(600, grid_h * 18 + 100))

    # 사용자 편집 데이터 수신
    if drawn_json_str:
        try:
            parsed_matrix = json.loads(drawn_json_str)
            if len(parsed_matrix) == grid_h and len(parsed_matrix[0]) == grid_w:
                st.session_state[canvas_session_key] = parsed_matrix
        except Exception:
            pass

    target = np.array(st.session_state[canvas_session_key], dtype=float)
    layout_name = f"Custom_{grid_w}x{grid_h}"

    with col_info:
        st.markdown("**실시간 패턴 프리뷰 (A)**")
        fig_prev, ax_prev = plt.subplots(figsize=(4.5, 4.5 * (grid_h / grid_w)))
        ax_prev.imshow(target, cmap='Blues', origin='upper', interpolation='nearest',
                       extent=[0, grid_w * unit_scale, grid_h * unit_scale, 0])
        ax_prev.set_title("Current Drawn Target (A)", fontsize=11, fontweight='bold')
        ax_prev.set_xlabel("X (nm)", fontsize=9)
        ax_prev.set_ylabel("Y (nm)", fontsize=9)
        ax_prev.xaxis.set_minor_locator(MultipleLocator(unit_scale))
        ax_prev.yaxis.set_minor_locator(MultipleLocator(unit_scale))
        ax_prev.grid(which='minor', color='gray', linestyle=':', linewidth=0.5, alpha=0.6)
        st.pyplot(fig_prev)

else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV (0 and 1 matrix)", type=["csv"])
    if uploaded_file is not None:
        target = np.genfromtxt(uploaded_file, delimiter=",").astype(float)
        target = np.nan_to_num(target, nan=0.0)
    else:
        target = np.zeros((30, 30), dtype=float)
        target[10:20, 10:20] = 1.0
    layout_name = "Uploaded_CSV"

# 세션 상태 관리 (마스크 동기화)
if "current_mask" not in st.session_state or st.session_state.get("last_layout") != layout_name:
    st.session_state.current_mask = target.copy()
    st.session_state.last_layout = layout_name

if st.session_state.current_mask.shape != target.shape:
    st.session_state.current_mask = target.copy()

# 광학 파라미터 제어
st.sidebar.subheader("Optical Lithography Parameters")
kernel_type = st.sidebar.radio("Optical Blur Kernel", ["Standard 5x5 Matrix", "2D Gaussian Function"])
sigma = st.sidebar.slider("Gaussian Diffusion Radius (sigma, px)", 1.0, 3.0, 1.8, 0.1)
threshold = st.sidebar.slider("Resist Threshold (I_th)", 0.20, 0.80, 0.55, 0.01)
show_grid = st.sidebar.checkbox("Display Unit Grid", value=True)

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

# 광학 노광 및 현상 연산
exposure = convolve(st.session_state.current_mask, kernel, mode='constant', cval=0.0)
max_val = np.max(exposure) if np.max(exposure) > 0 else 1.0
norm_exposure = exposure / max_val
printed = (norm_exposure >= threshold).astype(float)

# 정량적 분석 지표 (IoU & 오차)
intersection = np.logical_and(target == 1, printed == 1).sum()
union = np.logical_or(target == 1, printed == 1).sum()
iou = (intersection / union) * 100 if union > 0 else 0.0
error_pixels = int(union - intersection)

h_px, w_px = target.shape
extent_nm = [0, w_px * unit_scale, h_px * unit_scale, 0]
aspect_ratio = h_px / w_px

# 메트릭 표시
st.subheader("Quantitative Evaluation")
c_m1, c_m2, c_m3 = st.columns(3)
c_m1.metric("Grid Dimension", f"{w_px} × {h_px} px")
c_m2.metric("Physical Size", f"{w_px * unit_scale:.1f} × {h_px * unit_scale:.1f} nm")
c_m3.metric("Pattern Fidelity (IoU)", f"{iou:.2f}%", delta=f"{error_pixels} px Error", delta_color="inverse")

# 4. 하단 4개 플롯 동적 배치
def generate_analysis_plot():
    is_tall = aspect_ratio >= 1.4
    if is_tall:
        fig, axes_flat = plt.subplots(1, 4, figsize=(22, 7), constrained_layout=True)
    else:
        fig, axes = plt.subplots(2, 2, figsize=(13, 13), constrained_layout=True)
        axes_flat = axes.flatten()

    # (A) Target Pattern
    axes_flat[0].imshow(target, cmap='Blues', origin='upper', extent=extent_nm, interpolation='nearest')
    axes_flat[0].set_title("(A) Target Pattern", fontsize=12, fontweight='bold')

    # (B) Current Mask
    axes_flat[1].imshow(st.session_state.current_mask, cmap='bone_r', origin='upper', extent=extent_nm, interpolation='nearest')
    axes_flat[1].set_title("(B) Current Mask (Input)", fontsize=12, fontweight='bold')

    # (C) Aerial Image
    im_c = axes_flat[2].imshow(norm_exposure, cmap='inferno', origin='upper', extent=extent_nm, interpolation='bilinear')
    axes_flat[2].set_title("(C) Aerial Image Intensity", fontsize=12, fontweight='bold')
    plt.colorbar(im_c, ax=axes_flat[2], fraction=0.046, pad=0.04)

    # (D) Superposition Analysis
    diff_view = np.zeros((*target.shape, 3))
    diff_view[(target == 1) & (printed == 1)] = [0.1, 0.85, 0.1]  # Green: 일치
    diff_view[(target == 1) & (printed == 0)] = [0.95, 0.1, 0.1]  # Red: 결손
    diff_view[(target == 0) & (printed == 1)] = [0.1, 0.4, 0.95]  # Blue: 번짐

    axes_flat[3].imshow(diff_view, origin='upper', extent=extent_nm, interpolation='nearest')
    axes_flat[3].set_title("(D) Superposition (G: Match, R: Under, B: Over)", fontsize=11, fontweight='bold')

    for ax in axes_flat:
        ax.set_xlabel("X (nm)", fontsize=10)
        ax.set_ylabel("Y (nm)", fontsize=10)
        ax.xaxis.set_major_locator(MultipleLocator(21.0))
        ax.yaxis.set_major_locator(MultipleLocator(31.5))
        if show_grid:
            ax.xaxis.set_minor_locator(MultipleLocator(unit_scale))
            ax.yaxis.set_minor_locator(MultipleLocator(unit_scale))
            ax.grid(which='minor', color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
            ax.grid(which='major', color='black', linestyle='--', linewidth=0.7, alpha=0.6)

    return fig

st.subheader("Lithography & OPC Analysis Result")
fig_result = generate_analysis_plot()
st.pyplot(fig_result)

# 5. 마스크 제어 및 내보내기
st.subheader("OPC Modification & Export")
ctrl_cols = st.columns(5 if layout_mode == "Interactive Grid Drawing" else 4)

with ctrl_cols[0]:
    if st.button("Reset Mask to Target"):
        st.session_state.current_mask = target.copy()
        st.rerun()

with ctrl_cols[1]:
    if st.button("Apply Quick Bias/OPC"):
        from scipy.ndimage import binary_dilation
        dilated = binary_dilation(target, iterations=1)
        st.session_state.current_mask = np.where(target == 1, target, (dilated.astype(float) * 0.5))
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
