import h5py
import numpy as np
from scipy.signal import butter, filtfilt
from tqdm import tqdm

# --------- CONFIG ---------
FS = 125
WIN_SEC = 8
WIN = FS * WIN_SEC          # 1000 samples
STRIDE = WIN // 2           # 50% overlap
FILES = ["Part_1.mat", "Part_2.mat", "Part_3.mat", "Part_4.mat"]
# --------------------------

def bandpass_ppg(x, fs=125, low=0.5, high=8, order=3):
    b, a = butter(order, [low/(fs/2), high/(fs/2)], btype='band')
    return filtfilt(b, a, x)

def load_signals(file):
    f = h5py.File(file, "r")
    
    part_key = [k for k in f.keys() if "Part" in k][0]
    part = f[part_key]        # shape = (3000,1)
    refs = f["#refs#"]

    ppgs, abps = [], []

    for i in range(len(part)):
        try:
            # dereference MATLAB cell pointer
            idx = part[i][0]
            seg = np.array(refs[idx])  # shape (N,3)
        except:
            continue

        if seg.ndim != 2 or seg.shape[1] < 2:
            continue

        ppg = seg[:,0]
        abp = seg[:,1]

        if len(ppg) < WIN:  # signal too short
            continue
        
        if np.std(ppg) < 0.01 or np.std(abp) < 0.01:
            continue

        ppgs.append(ppg.astype(np.float32))
        abps.append(abp.astype(np.float32))

    f.close()
    return ppgs, abps


def sliding(ppgs, abps):
    X, Y = [], []
    
    for ppg, abp in zip(ppgs, abps):
        # 1) Bandpass filter
        try:
            ppg = bandpass_ppg(ppg)
        except:
            continue

        for s in range(0, len(ppg) - WIN, STRIDE):
            p = ppg[s:s+WIN]
            a = abp[s:s+WIN]

            if np.std(p) < 0.01 or np.std(a) < 0.01:
                continue

            # Normalize (z-score then clip)
            p = (p - np.mean(p)) / (np.std(p) + 1e-6)
            p = np.clip(p, -5, 5)

            map_val = np.mean(a)  # MAP target

            X.append(p)
            Y.append(map_val)

    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


# ================= MAIN =================
print("✅ Start UCI PPG→MAP preprocessing")

allX, allY = [], []

for f in FILES:
    print(f"\n📂 Processing {f}...")
    ppgs, abps = load_signals(f)
    X, Y = sliding(ppgs, abps)

    print(f"  ➜ Windows: {len(X)}")
    
    if len(X) > 0:
        allX.append(X)
        allY.append(Y)

if not allX:
    raise RuntimeError("❌ No windows created. Check data!")

X = np.concatenate(allX)
Y = np.concatenate(allY)

print(f"\n✅ Final dataset: {X.shape}")

with h5py.File("ppg_dataset.h5", "w") as hf:
    hf.create_dataset("X", data=X, dtype='float32', compression="gzip")
    hf.create_dataset("Y", data=Y, dtype='float32', compression="gzip")

print("\n💾 Saved: ppg_dataset.h5 ✅")
