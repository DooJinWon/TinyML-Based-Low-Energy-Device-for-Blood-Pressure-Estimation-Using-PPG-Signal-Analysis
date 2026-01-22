import tensorflow as tf
import h5py
import numpy as np
import os
import matplotlib.pyplot as plt

# ================= CONFIG =================
TFLITE_PATH = "saved_model/model_float32.tflite"
DATA_PATH = "ppg_val.h5" 
NUM_PLOT_SAMPLES = 1000
# ==========================================

def plot_results():
    if not os.path.exists(TFLITE_PATH):
        print(f"❌ 파일을 찾을 수 없습니다: {TFLITE_PATH}")
        return

    print(f"🔄 모델 로드 중: {TFLITE_PATH}")
    interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # [수정 1] 모델이 실제로 원하는 입력 형태(Shape)를 가져옵니다.
    target_shape = input_details[0]['shape']
    print(f"ℹ️ 모델이 원하는 입력 형태: {target_shape}")

    print(f"📂 데이터 로드 중: {DATA_PATH}")
    with h5py.File(DATA_PATH, 'r') as f:
        X_val = f['X'][:NUM_PLOT_SAMPLES]
        Y_val = f['Y'][:NUM_PLOT_SAMPLES]

    predictions = []
    targets = []
    
    print("🚀 추론 및 데이터 수집 중...")

    for i in range(len(X_val)):
        input_data = X_val[i].astype(np.float32)
        
        # 전처리: 0~1 정규화 (MinMax Scaling)
        d_min = np.min(input_data)
        d_max = np.max(input_data)
        if d_max - d_min != 0:
            input_data = (input_data - d_min) / (d_max - d_min)
        else:
            input_data = np.zeros_like(input_data)

        # [수정 2] 모델이 원하는 모양으로 '자동 Reshape'
        try:
            input_data = input_data.reshape(target_shape)
        except ValueError:
            print(f"❌ 데이터 크기 오류! (Data: {input_data.shape} -> Target: {target_shape})")
            continue

        # 추론 실행
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        # 결과 저장
        output_data = interpreter.get_tensor(output_details[0]['index'])
        pred = output_data[0][0]
        target = Y_val[i]
        if isinstance(target, np.ndarray): target = target.item()
            
        predictions.append(pred)
        targets.append(target)

    # === 📊 그래프 그리기 ===
    plt.figure(figsize=(12, 6))
    
    # 1. 실제값 (정답)
    plt.plot(targets, label='Actual BP (Ground Truth)', color='blue', linewidth=2, linestyle='-')
    
    # 2. 예측값 (모델)
    plt.plot(predictions, label='Predicted BP (TFLite float32)', color='red', linewidth=2, linestyle='--', alpha=0.8)

    plt.title(f'Waveform Comparison: Actual vs Predicted (First {len(predictions)} samples)', fontsize=16)
    plt.xlabel('Sample Index', fontsize=12)
    plt.ylabel('Blood Pressure (mmHg)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_results()