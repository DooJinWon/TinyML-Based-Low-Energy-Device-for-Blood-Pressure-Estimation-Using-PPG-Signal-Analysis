import os

# ================= 설정 =================
# 사진 속에 있는 파일 중 가장 용량이 작고 최적화된 파일 선택
# (경로가 다르다면 이 부분을 수정하세요)
TFLITE_PATH = "output_int8/model_full_integer_quant.tflite" 

OUTPUT_HEADER = "model_data.h"
VAR_NAME = "g_model_data"
# ========================================

def hex_to_c_array(data, var_name):
    c_str = f"// TensorFlow Lite Micro Model (Int8 Quantized)\n"
    c_str += f"// Original File: {TFLITE_PATH}\n"
    c_str += f"// Size: {len(data)} bytes\n\n"
    c_str += f"#include <stdint.h>\n\n"
    c_str += f"const unsigned int {var_name}_len = {len(data)};\n"
    c_str += f"alignas(16) const unsigned char {var_name}[] = {{\n"
    
    for i, val in enumerate(data):
        c_str += f"0x{val:02x}, "
        if (i + 1) % 12 == 0:
            c_str += "\n"
    c_str += "};\n"
    return c_str

def main():
    # 전역 변수 값을 로컬 변수에 담아서 사용 (에러 방지)
    current_path = TFLITE_PATH

    # 경로 확인 및 자동 수정 로직
    if not os.path.exists(current_path):
        # 혹시 파일이 현재 폴더에 바로 있는지 확인
        filename_only = os.path.basename(current_path)
        if os.path.exists(filename_only):
            current_path = filename_only
            print(f"ℹ️ 경로 수정됨: {current_path} (현재 폴더에서 발견)")
        else:
            print(f"❌ 에러: '{current_path}' 파일을 찾을 수 없습니다.")
            print(f"   현재 위치: {os.getcwd()}")
            return

    print(f"📂 '{current_path}' 읽는 중... (크기: {os.path.getsize(current_path)} bytes)")
    
    with open(current_path, "rb") as f:
        model_data = f.read()

    with open(OUTPUT_HEADER, "w") as f:
        f.write(f"#ifndef MODEL_DATA_H\n")
        f.write(f"#define MODEL_DATA_H\n\n")
        f.write(hex_to_c_array(model_data, VAR_NAME))
        f.write(f"\n#endif // MODEL_DATA_H\n")

    print(f"✅ 변환 완료! '{OUTPUT_HEADER}' 파일이 생성되었습니다.")
    print(f"   이제 이 파일을 Segger 프로젝트 폴더로 복사하세요.")

if __name__ == "__main__":
    main()