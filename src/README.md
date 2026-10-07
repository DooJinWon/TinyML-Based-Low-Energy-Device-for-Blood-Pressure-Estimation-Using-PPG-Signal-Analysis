# Model Development Scripts

[Project overview](../README.md) · [Intermediate report](../docs/intermediate-report.pdf)

| Script | Role |
|---|---|
| `preprocessing.py` | PPG filtering and training window preparation |
| `train_onlycnn_beat.py` | PyTorch CNN training |
| `export_onnx.py` | ONNX model export |
| `convert_to_tflite.py` | Sample calibration-data generation; not a complete TFLite converter |
| `convert_to_header.py` | TFLite model bytes to a C header |
| `waveform.py` | TFLite inference and waveform inspection |

Run scripts from this directory after supplying the expected dataset/model files and configuring their local paths. These scripts represent the earlier model-development stage; the final Edge Impulse module and Zephyr build configuration are not included.
