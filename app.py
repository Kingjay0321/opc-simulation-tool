import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

st.set_page_config(page_title="OPC Simulation Tool", layout="wide")
st.title("Optical Proximity Correction (OPC) Tool")
st.write("7nm FinFET Standard Cell Metal1 레이아웃 광학 근접효과 보정 시뮬레이터")

# 1. 사이드바 설정 (파라미터 제어)
st.sidebar.header("공정 및 시뮬레이션 파라미터")
layout_choice = st.sidebar.selectbox("대상 레이아웃 선택", ["INV_X1", "NAND2_X4"])

kernel_type = st.sidebar.radio("빛 번짐 커널 선택", ["과제 제공 5x5 행렬", "2D 가우시안"])
sigma = st.sidebar.slider("가우시안 확산 반경 (sigma, pixels)", 1.0, 3.0, 1.8, 0.1)
threshold = st.sidebar.slider("현상 임계값 (Threshold)", 0.2, 0.8, 0.45, 0.01)

# 2. 기본 타겟 레이아웃 생성 함수 (단위: 1 pixel = 3.5 nm)
def get_target_layout(name):
    if name == "INV_X1":
        # 18 x 54 grid
        mask = np.zeros((54, 18), dtype=float)
        mask[0:3, :] = 1.0     # 상단 가로바
        mask[51:54, :] = 1.0   # 하단 가로바
        mask[3:21, 3:6] = 1.0  # 좌상단 세로선
        mask[33:51, 3:6] = 1.0 # 좌하단 세로선
        mask[18:36, 12:15] = 1.0 # 우측 중앙 세로선
        return mask
    else:
        # NAND2_X4: 45 x 54 grid
        mask = np.zeros((54, 45), dtype=float)
        mask[0:3, :] = 1.0     # 상단 가로바
        mask[51:54, :] = 1.0   # 하단 가로바
        # 세로 핀 및 내부 배선 (3λ 선폭)
        mask[3:12, 6:9] = 1.0
        mask[3:12, 21:24] = 1.0
        mask[3:12, 36:39] = 1.0
        mask[12:27, 12:15] = 1.0
        mask[12:27, 30:33] = 1.0
        mask[24:27, 12:33] = 1.0 # 중앙 가로 연결
        mask[27:42, 21:24] = 1.0
        mask[42:51, 6:9] = 1.0
        mask[42:51, 15:18] = 1.0
        mask[42:51, 27:30] = 1.0
        mask[42:51, 36:39] = 1.0
        return mask

target = get_target_layout(layout_choice)

# 세션 상태에 수정 가능한 마스크 저장
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
    # 2D Gaussian Kernel
    size = int(round(sigma * 6)) | 1
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    kernel = kernel / np.sum(kernel)

# 4. 시뮬레이션 계산
exposure = convolve(st.session_state.current_mask, kernel, mode='constant', cval=0.0)
max_val = np.max(exposure) if np.max(exposure) > 0 else 1.0
norm_exposure = exposure / max_val
printed = (norm_exposure >= threshold).astype(float)

# IoU 계산
intersection = np.logical_and(target == 1, printed == 1).sum()
union = np.logical_or(target == 1, printed == 1).sum()
iou = (intersection / union) * 100 if union > 0 else 0.0

# 5. 화면 표시 (시각화)
col_stat1, col_stat2 = st.columns(2)
col_stat1.metric("목표 패턴 대비 일치도 (IoU)", f"{iou:.2f}%")
col_stat2.metric("불일치 오차 픽셀 수", f"{int(union - intersection)} pixels")

fig, axes = plt.subplots(1, 4, figsize=(18, 6))

axes[0].imshow(target, cmap='Blues', origin='upper')
axes[0].set_title("(a) Target Pattern")
axes[0].axis('off')

axes[1].imshow(st.session_state.current_mask, cmap='Purples', origin='upper')
axes[1].set_title("(b) Current Mask (Input)")
axes[1].axis('off')

im = axes[2].imshow(norm_exposure, cmap='hot', origin='upper')
axes[2].set_title("(c) Aerial Image (Exposure)")
plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
axes[2].axis('off')

# 중첩 비교 (Target vs Printed)
diff_view = np.zeros((*target.shape, 3))
diff_view[(target == 1) & (printed == 1)] = [0, 0.8, 0]   # 정상 (초록)
diff_view[(target == 1) & (printed == 0)] = [1, 0, 0]     # 부족/수축 (빨강)
diff_view[(target == 0) & (printed == 1)] = [0, 0.4, 1]   # 과도/번짐 (파랑)

axes[3].imshow(diff_view, origin='upper')
axes[3].set_title("(d) Superposition (Green: Match)")
axes[3].axis('off')

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
        corrected = target.copy()
        # 모서리 Serif 및 선 끝 확장 픽셀 추가 예시
        from scipy.ndimage import binary_dilation
        corrected = binary_dilation(target, iterations=1).astype(float)
        st.session_state.current_mask = corrected
        st.rerun()

with c3:
    st.download_button(
        label="현재 마스크 배열 다운로드 (CSV)",
        data=np.savetxt("mask.csv", st.session_state.current_mask, delimiter=","),
        file_name=f"{layout_choice}_mask.csv",
        mime="text/csv"
    )
