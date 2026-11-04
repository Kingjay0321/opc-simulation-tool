import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.ndimage import convolve
import io

st.set_page_config(page_title="OPC Simulation Tool", layout="wide")
st.title("Optical Proximity Correction (OPC) Tool")
st.markdown("**7nm FinFET Standard Cell Metal1 Layout Simulation & Correction Platform**")
st.caption("Standard Unit: 1 Grid = 3.5 nm × 3.5 nm (λ = 3.5 nm, Metal1 Wire Width = 3 Grid = 10.5 nm)")

unit_scale = 3.5  # nm/pixel

# 1. 사이드바 설정
st.sidebar.header("Simulation Settings")
layout_mode = st.sidebar.radio("Layout Input Mode", ["Preset Standard Cells", "Interactive Grid Drawing", "CSV File Upload"])

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

custom_drawn_matrix = None

if layout_mode == "Preset Standard Cells":
    preset_choice = st.sidebar.selectbox("Select Target Cell", ["INV_X1", "NAND2_X4"])
    target = get_inv_x1() if preset_choice == "INV_X1" else get_nand2_x4()
    layout_name = preset_choice

elif layout_mode == "Interactive Grid Drawing":
    st.sidebar.subheader("Canvas Grid Size")
    grid_w = st.sidebar.number_input("Grid Width (X, columns)", min_value=5, max_value=40, value=15, step=1)
    grid_h = st.sidebar.number_input("Grid Height (Y, rows)", min_value=5, max_value=40, value=20, step=1)

    # 세션 상태에 편집 그리드 보관
    grid_key = f"grid_{grid_w}_{grid_h}"
    if grid_key not in st.session_state:
        st.session_state[grid_key] = pd.DataFrame(
            False,
            index=[f"Y{r}" for r in range(grid_h)],
            columns=[f"X{c}" for c in range(grid_w)]
        )

    st.subheader("Interactive Pattern Editor")
    col_editor, col_preview = st.columns([1.2, 1])

    with col_editor:
        st.markdown("**지뢰찾기 스타일 격자판**: 클릭하여 패턴(True=배선)을 토글하세요.")
        edited_df = st.data_editor(
            st.session_state[grid_key],
            use_container_width=True,
            height=min(450, grid_h * 28 + 40),
            key=f"editor_{grid_w}_{grid_h}"
        )
        c_rst1, c_rst2 = st.columns(2)
        with c_rst1:
            if st.button("격자판 전체 초기화"):
                st.session_state[grid_key] = pd.DataFrame(
                    False,
                    index=[f"Y{r}" for r in range(grid_h)],
                    columns=[f"X{c}" for c in range(grid_w)]
                )
                st.rerun()

    target = edited_df.to_numpy().astype(float)
    custom_drawn_matrix = target.copy()
    layout_name = f"Custom_{grid_w}x{grid_h}"

    with col_preview:
        st.markdown("**실시간 패턴 프리뷰 (A)**")
        fig_prev, ax_prev = plt.subplots(figsize=(5, 5 * (grid_h / grid_w)))
        ax_prev.imshow(target, cmap='Blues', origin='upper', interpolation='nearest',
                       extent=[0, grid_w * unit_scale, grid_h * unit_scale, 0])
        ax_prev.set_title("Current Drawn Pattern", fontsize=11, fontweight='bold')
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

# 세션 상태 관리 (마스크 싱크)
if "current_mask" not in st.session_state or st.session_state.get("last_layout") != layout_name:
    st.session_state.current_mask = target.copy()
    st.session_state.last_layout = layout_name

# 크기 변경 시 마스크 갱신
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

# 4. 하단 4개 플롯 최적 정렬 함수
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

    # 축 눈금 공통 설정
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

# 5. 마스크 제어 및 다운로드 파트
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
    # 보정된 마스크 CSV
    mask_csv_buf = io.StringIO()
    np.savetxt(mask_csv_buf, st.session_state.current_mask.astype(int), delimiter=",", fmt="%d")
    st.download_button(
        label="Download Mask (CSV)",
        data=mask_csv_buf.getvalue(),
        file_name=f"{layout_name}_mask.csv",
        mime="text/csv"
    )

# 3번 요구사항: 패턴 그리기 모드일 때만 나타나는 원본 Target CSV 다운로드 버튼
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
    # 4개 그림 종합 분석 고해상도 플롯 다운로드
    img_buf = io.BytesIO()
    fig_result.savefig(img_buf, format='png', dpi=300, bbox_inches='tight')
    st.download_button(
        label="Download Analysis Plot (PNG)",
        data=img_buf.getvalue(),
        file_name=f"{layout_name}_analysis_plot.png",
        mime="image/png"
    )
