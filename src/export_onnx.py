import torch
import torch.nn as nn
import torch.onnx
import numpy as np

# ================= CONFIG =================
PT_FILENAME = "cnn_only_best.pt"  # 저장된 모델 파일명
ONNX_FILENAME = "model.onnx"      # 출력할 ONNX 파일명
SIG_LEN = 1000                     # [중요] PPG 신호 길이 (본인 데이터에 맞게 수정!)
# ==========================================

# --- 1. 모델 클래스 정의 (보내주신 코드 복사) ---
class AttentionPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.att = nn.Linear(dim, 1)
    def forward(self, x):
        # x: [B, T, D]
        w = torch.softmax(self.att(x), dim=1)
        return (x * w).sum(dim=1)

class CNN_ONLY_BP(nn.Module):
    def __init__(self, use_attention=True):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1,   32, kernel_size=9, padding=4),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32,  64, kernel_size=9, padding=4),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(64, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(4),
        )
        self.use_attention = use_attention
        if use_attention:
            self.pool = AttentionPool(128)
        else:
            self.pool = None
        
        self.head = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        z = x.permute(0, 2, 1) # [B, T, 1] -> [B, 1, T]
        z = self.cnn(z)
        z = z.permute(0, 2, 1) # [B, D, T'] -> [B, T', D]
        if self.use_attention:
            z = self.pool(z)
        else:
            z = z.mean(dim=1)
        y = self.head(z)
        return y

# --- 2. 변환 실행 ---
def export():
    # 모델 생성 및 가중치 로드
    device = torch.device('cpu')
    model = CNN_ONLY_BP(use_attention=True)
    
    try:
        # map_location='cpu' 필수
        state_dict = torch.load(PT_FILENAME, map_location=device)
        model.load_state_dict(state_dict)
        print(f"✅ {PT_FILENAME} 로드 완료")
    except FileNotFoundError:
        print(f"⚠️ 파일을 찾을 수 없어 랜덤 가중치로 진행합니다.")
    
    model.eval()

    # 더미 입력 생성 (Batch=1, Time=SIG_LEN, Channel=1)
    # MCU에서는 배치 크기 1로 고정하는 것이 좋습니다.
    dummy_input = torch.randn(1, SIG_LEN, 1)

    # ONNX Export
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_FILENAME,
        export_params=True,
        opset_version=13,        # 호환성 좋은 버전
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output']
    )
    print(f"✅ 변환 완료: {ONNX_FILENAME}")

if __name__ == "__main__":
    export()