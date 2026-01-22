import numpy as np
import os

# 1. 모델 입력 설정 (사용자의 모델에 맞게)
# INPUT Shape: [1, 1000, 1]
# INPUT dtype: float32
SHAPE = (1, 1000, 1)
FILENAME = "calib_data.npy"

# 2. 랜덤 데이터 생성 (0.0 ~ 1.0 사이 값)
# 실제로는 PPG 데이터의 범위와 비슷한 값(예: 정규화된 값)을 넣는 게 정확도에 좋습니다.
data = np.random.uniform(0.0, 1.0, SHAPE).astype(np.float32)

# 3. .npy 파일로 저장
np.save(FILENAME, data)

print(f"✅ '{FILENAME}' 생성 완료! (Shape: {data.shape})")
print("이제 아래 명령어를 터미널에 복사해서 실행하세요.")