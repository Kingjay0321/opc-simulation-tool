import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.ndimage import convolve
import io

st.set_page_config(page_title="OPC Simulation Tool", layout="wide")
st.title("Optical Proximity Correction (OPC) Tool")
st.markdown("**7nm FinFET Standard Cell Metal1 레이아웃 광학 근접효과 보정 시뮬레이터**")
st.caption("기본 격자 단위: 1 Grid = 3.5 nm × 3.5 nm (λ = 3.5 nm, Metal1 선폭 = 3 Grid = 10.5 nm)")

# 1. 사이드바 설정 (파라미터 제어)
st.sidebar.header("공정 및 시뮬레이션 파라미터")
layout_choice = st.sidebar.selectbox("대상 레이아웃 선택", ["INV_X1", "NAND2_X4", "사용자 직접 업로드 (CSV)"])

uploaded_file = None
if layout_choice == "사용자 직접 업로드 (CSV)":
    uploaded_file = st.sidebar.file_uploader("0과 1로 구성된 CSV 파일 업로드", type=["csv"])

kernel_type = st.sidebar.radio("빛 번짐 커널 선택", ["과제 제공 5x5 행렬", "2D 가우시안"])
sigma = st.sidebar.slider("가우시안 확산 반경 (sigma, pixels)", 1.0, 3.0, 1.8, 0.1)
threshold = st.sidebar.slider("현상 임계값 (Threshold)", 0.2, 0.8, 0.45, 0.01)

# 격자 표시 옵션
show_grid = st.sidebar.checkbox("픽셀 모눈 격자(Grid) 표시", value=True)
unit_scale = 3.5  # nm per pixel

# 2. 기본 타겟 레이아웃 생성 함수
def get_target_layout(name, file_obj=None):
    if name == "INV_X1":
        # 18 x 54 grid (63 nm x 189 nm)
        mask = np.zeros((54, 18), dtype=float)
        mask[0:3, :] = 1.0
        mask[51:54, :] = 1.0
        mask[3:21, 3:6] = 1.0
        mask[33:51, 3:6] = 1.0
        mask[18:36, 12:15] = 1.0
        return mask
    elif name == "NAND2_X4":
        # 45 x 54 grid (157.5 nm x 189 nm)
        mask = np.zeros((54, 45), dtype=float)
        mask[0:3, :] = 1.0
        mask[51:54, :] = 1.0
        mask[3:12, 6:9] = 1.0
        mask[3:12, 21:24] = 1.0
        mask[3:12, 36:39] = 1.0
        mask[12:27, 12:15] = 1.0
        mask[12:27, 30:33] = 1.0
        mask[24:27, 12:33] = 1.0
        mask[27:42, 21:24] = 1.0
        mask[42:51, 6:9] = 1.0
        mask[42:51, 15:18] = 1.0
        mask[42:51, 27:30] = 1.0
        mask[42:51, 36:39] = 1.0
        return mask
    else:
        if file_obj is not None:
            return np.genfromtxt(file_obj, delimiter=",").astype(float)
        else:
            sample = np.zeros((30, 30), dtype=float)
            sample[10:20, 10:20] = 1.0
            return sample

target = get_target_layout(layout_choice, uploaded_file)

if "current_mask" not in st.session_state or st.session_state.get("last_choice") != layout_choice:
    st.session_state.current_mask = target.copy()
    st.session_state.last_choice = layout_choice

# 3. 광학 커널 생성
if kernel_type == "과제 제공 5x5 행렬":
    kernel = np.array([
        [0.0, 0.5, 1.0, 0.5, 0.0],
        [0.5, 4.0, 5.0, 4.0, 0.5],
        [1.0, 5.0, 10.0, 5.0, 1.0],
        [0.5, 4.0, 5.0, 4.0, 0.5],
        [0.0, 0.5, 1.0, 0.5, 0.0]
    ])
    kernel = kernel / np.sum(kernel)
else:
    size = int(round(sigma * 6)) | 1
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    kernel = kernel / np.sum(kernel)

# 4. 시뮬레이션 연산
exposure = convolve(st.session_state.current_mask, kernel, mode='constant', cval=0.0)
max_val = np.max(exposure) if np.max(exposure) > 0 else 1.0
norm_exposure = exposure / max_val
printed = (norm_exposure >= threshold).astype(float)

# IoU 및 정량 지표 계산
intersection = np.logical_and(target == 1, printed == 1).sum()
union = np.logical_or(target == 1, printed == 1).sum()
iou = (intersection / union) * 100 if union > 0 else 0.0
error_pixels = int(union - intersection)

# 5. 화면 표시 (시각화 파트)
st.subheader("패턴 및 노광 결과 분석")
m1, m2, m3 = st.columns(3)
h_pixels, w_pixels = target.shape
m1.metric("격자 크기 (Pixels)", f"{w_pixels} × {h_pixels} 칸")
m2.metric("실제 레이아웃 치수 (nm)", f"{w_pixels * unit_scale:.1f} × {h_pixels * unit_scale:.1f} nm")
m3.metric("목표 대비 일치도 (IoU)", f"{iou:.2f}%", delta=f"{error_pixels} px 오차", delta_color="inverse")

fig, axes = plt.subplots(1, 4, figsize=(22, 9), constrained_layout=True)

# 축 스케일링 설정 (nm 단위 범위)
extent_nm = [0, w_pixels * unit_scale, h_pixels * unit_scale, 0]

# (a) Target Pattern
im0 = axes[0].imshow(target, cmap='Blues', origin='upper', extent=extent_nm, interpolation='nearest')
axes[0].set_title(f"(a) Target Pattern\n(1 칸 = {unit_scale} nm)", fontsize=13, fontweight='bold')

# (b) Current Mask (Input)
im1 = axes[1].imshow(st.session_state.current_mask, cmap='bone_r', origin='upper', extent=extent_nm, interpolation='nearest')
axes[1].set_title(f"(b) Current Mask (Input)\n(1 칸 = {unit_scale} nm)", fontsize=13, fontweight='bold')

# (c) Aerial Image
im2 = axes[2].imshow(norm_exposure, cmap='inferno', origin='upper', extent=extent_nm, interpolation='bilinear')
axes[2].set_title("(c) Aerial Image (Exposure)\n(광량 분포 히트맵)", fontsize=13, fontweight='bold')
plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

# (d) Superposition (비교 중첩)
diff_view = np.zeros((*target.shape, 3))
diff_view[(target == 1) & (printed == 1)] = [0.1, 0.85, 0.1]  # 일치 영역 (초록)
diff_view[(target == 1) & (printed == 0)] = [0.95, 0.1, 0.1]  # 미전사 결손 영역 (빨강)
diff_view[(target == 0) & (printed == 1)] = [0.1, 0.4, 0.95]  # 과도 번짐 영역 (파랑)

im3 = axes[3].imshow(diff_view, origin='upper', extent=extent_nm, interpolation='nearest')
axes[3].set_title("(d) Superposition Analysis\n(Green: 일치 / Red: 결손 / Blue: 번짐)", fontsize=13, fontweight='bold')

# 픽셀 모눈선 및 nm 축 레이블 공통 적용
for ax in axes:
    ax.set_xlabel("X (nm)", fontsize=11)
    ax.set_ylabel("Y (nm)", fontsize=11)
    ax.xaxis.set_major_locator(MultipleLocator(21.0))
    ax.yaxis.set_major_locator(MultipleLocator(31.5))
    
    if show_grid:
        ax.xaxis.set_minor_locator(MultipleLocator(unit_scale))
        ax.yaxis.set_minor_locator(MultipleLocator(unit_scale))
        ax.grid(which='minor', color='gray', linestyle=':', linewidth=0.5, alpha=0.6)
        ax.grid(which='major', color='black', linestyle='--', linewidth=0.8, alpha=0.7)

st.pyplot(fig)

# 6. 마스크 보정 컨트롤러
st.subheader("마스크 형태 보정 (OPC 조작)")
c1, c2, c3 = st.columns(3)
with c1:
    if st.button("마스크 초기화 (Target과 동일하게)"):
        st.session_state.current_mask = target.copy()
        st.rerun()

with c2:
    if st.button("자동 간단 OPC 적용 (Serif & Line Extension)"):
        from scipy.ndimage import binary_dilation
        corrected = binary_dilation(target, iterations=1).astype(float)
        st.session_state.current_mask = corrected
        st.rerun()

with c3:
    csv_buffer = io.StringIO()
    np.savetxt(csv_buffer, st.session_state.current_mask, delimiter=",", fmt="%.1f")
    csv_data = csv_buffer.getvalue()
    
    st.download_button(
        label="현재 마스크 배열 다운로드 (CSV)",
        data=csv_data,
        file_name=f"{layout_choice}_mask.csv",
        mime="text/csv"
    )
