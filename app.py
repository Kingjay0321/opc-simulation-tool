import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.ndimage import convolve
import io

st.set_page_config(page_title="OPC Simulation Tool", layout="wide")
st.title("Optical Proximity Correction (OPC) Tool")
st.markdown("**7nm FinFET Standard Cell Metal1 Layout Simulation & Correction Platform**")
st.caption("Standard Unit: 1 Grid = 3.5 nm × 3.5 nm (λ = 3.5 nm, Metal1 Wire Width = 3 Grid = 10.5 nm)")

# 1. 사이드바 설정
st.sidebar.header("Simulation Settings")
layout_mode = st.sidebar.radio("Layout Input Mode", ["Preset Standard Cells", "Interactive Grid Drawing", "CSV File Upload"])

unit_scale = 3.5  # nm per pixel

# 2. 타겟 레이아웃 생성
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

if layout_mode == "Preset Standard Cells":
    preset_choice = st.sidebar.selectbox("Select Target Cell", ["INV_X1", "NAND2_X4"])
    target = get_inv_x1() if preset_choice == "INV_X1" else get_nand2_x4()
    layout_name = preset_choice

elif layout_mode == "Interactive Grid Drawing":
    st.sidebar.subheader("Canvas Grid Size")
    grid_w = st.sidebar.number_input("Grid Width (X, pixels)", min_value=10, max_value=60, value=20, step=1)
    grid_h = st.sidebar.number_input("Grid Height (Y, pixels)", min_value=10, max_value=60, value=20, step=1)
    
    st.info("Draw pattern below: Click or drag to draw wires (White = Pattern, Black = Space).")
    target = np.zeros((int(grid_h), int(grid_w)), dtype=float)
    
    try:
        from streamlit_drawable_canvas import st_canvas
        pixel_size = 15
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 1.0)",
            stroke_width=pixel_size,
            stroke_color="#FFFFFF",
            background_color="#000000",
            width=int(grid_w * pixel_size),
            height=int(grid_h * pixel_size),
            drawing_mode="freedraw",
            key=f"canvas_{grid_w}_{grid_h}",
        )
        
        # 런타임 에러 방지 안전 블록
        if canvas_result is not None:
            raw_data = getattr(canvas_result, "image_data", None)
            if raw_data is not None and len(raw_data.shape) == 3:
                gray = raw_data[:, :, 0]
                # 격자 단위 리샘플링
                y_indices = np.linspace(0, gray.shape[0] - 1, int(grid_h)).astype(int)
                x_indices = np.linspace(0, gray.shape[1] - 1, int(grid_w)).astype(int)
                sampled = gray[np.ix_(y_indices, x_indices)]
                target = (sampled > 100).astype(float)
    except Exception:
        st.warning("Using basic blank canvas mode.")
        target = np.zeros((int(grid_h), int(grid_w)), dtype=float)
        
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

# 세션 상태 관리
if "current_mask" not in st.session_state or st.session_state.get("last_layout") != layout_name:
    st.session_state.current_mask = target.copy()
    st.session_state.last_layout = layout_name

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

# 4. 동적 플롯 생성 (길쭉하면 1x4, 정사각/가로형이면 2x2)
def generate_adaptive_plot():
    is_tall = aspect_ratio >= 1.4  # 세로가 긴 형태 판별
    
    if is_tall:
        fig, axes_flat = plt.subplots(1, 4, figsize=(22, 8), constrained_layout=True)
    else:
        fig, axes = plt.subplots(2, 2, figsize=(14, 14), constrained_layout=True)
        axes_flat = axes.flatten()

    # (A) Target Pattern
    axes_flat[0].imshow(target, cmap='Blues', origin='upper', extent=extent_nm, interpolation='nearest')
    axes_flat[0].set_title("(A) Target Pattern", fontsize=13, fontweight='bold')
    
    # (B) Current Mask
    axes_flat[1].imshow(st.session_state.current_mask, cmap='bone_r', origin='upper', extent=extent_nm, interpolation='nearest')
    axes_flat[1].set_title("(B) Current Mask (Input)", fontsize=13, fontweight='bold')
    
    # (C) Aerial Image
    im_c = axes_flat[2].imshow(norm_exposure, cmap='inferno', origin='upper', extent=extent_nm, interpolation='bilinear')
    axes_flat[2].set_title("(C) Aerial Image Intensity", fontsize=13, fontweight='bold')
    plt.colorbar(im_c, ax=axes_flat[2], fraction=0.046, pad=0.04)
    
    # (D) Superposition Analysis
    diff_view = np.zeros((*target.shape, 3))
    diff_view[(target == 1) & (printed == 1)] = [0.1, 0.85, 0.1]  # 일치 (초록)
    diff_view[(target == 1) & (printed == 0)] = [0.95, 0.1, 0.1]  # 수축/미전사 (빨강)
    diff_view[(target == 0) & (printed == 1)] = [0.1, 0.4, 0.95]  # 번짐/과도현상 (파랑)
    
    axes_flat[3].imshow(diff_view, origin='upper', extent=extent_nm, interpolation='nearest')
    axes_flat[3].set_title("(D) Superposition (G: Match, R: Under, B: Over)", fontsize=12, fontweight='bold')
    
    # 축 눈금 포맷
    for ax in axes_flat:
        ax.set_xlabel("X Position (nm)", fontsize=10)
        ax.set_ylabel("Y Position (nm)", fontsize=10)
        ax.xaxis.set_major_locator(MultipleLocator(21.0))
        ax.yaxis.set_major_locator(MultipleLocator(31.5))
        if show_grid:
            ax.xaxis.set_minor_locator(MultipleLocator(unit_scale))
            ax.yaxis.set_minor_locator(MultipleLocator(unit_scale))
            ax.grid(which='minor', color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
            ax.grid(which='major', color='black', linestyle='--', linewidth=0.7, alpha=0.6)
            
    return fig

fig_result = generate_adaptive_plot()
st.pyplot(fig_result)

# 5. 마스크 제어 및 정밀 다운로드 기능
st.subheader("OPC Modification & Export")
c1, c2, c3, c4 = st.columns(4)

with c1:
    if st.button("Reset Mask to Target"):
        st.session_state.current_mask = target.copy()
        st.rerun()

with c2:
    if st.button("Apply Quick Bias/OPC"):
        from scipy.ndimage import binary_dilation
        # 모서리 Serif 및 선 끝 확장 효과
        dilated = binary_dilation(target, iterations=1)
        st.session_state.current_mask = np.where(target == 1, target, (dilated.astype(float) * 0.5))
        st.rerun()

with c3:
    csv_buf = io.StringIO()
    # 0과 1의 깔끔한 정수형(int) 포맷으로 내보내기
    np.savetxt(csv_buf, st.session_state.current_mask.astype(int), delimiter=",", fmt="%d")
    st.download_button(
        label="Download Mask (CSV)",
        data=csv_buf.getvalue(),
        file_name=f"{layout_name}_mask.csv",
        mime="text/csv"
    )

with c4:
    img_buf = io.BytesIO()
    # 고해상도(300 DPI) 논문용 분석 이미지 내보내기
    fig_result.savefig(img_buf, format='png', dpi=300, bbox_inches='tight')
    st.download_button(
        label="Download Analysis Plot (PNG)",
        data=img_buf.getvalue(),
        file_name=f"{layout_name}_analysis_plot.png",
        mime="image/png"
    )
