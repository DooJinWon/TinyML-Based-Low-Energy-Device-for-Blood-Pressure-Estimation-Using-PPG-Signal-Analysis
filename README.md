# TinyML-Based Low-Power Blood Pressure Estimation Device  
Using PPG Signal Analysis

This repository contains the implementation and system design of a low-power wearable blood pressure estimation device based on TinyML.  
The project focuses on estimating blood pressure from photoplethysmography (PPG) signals without using a traditional cuff, enabling continuous and long-term monitoring in low-power embedded environments.

---

## Project Overview

Blood pressure is one of the most important physiological indicators of cardiovascular health. However, conventional cuff-based measurement methods are not suitable for continuous or long-term monitoring due to discomfort and usability limitations.

This project proposes a **TinyML-based, cuffless blood pressure estimation system** that analyzes PPG signals using lightweight deep learning models deployed directly on a microcontroller. The system is designed to operate under strict memory and power constraints while maintaining clinically meaningful estimation accuracy.

---

## Key Objectives

- Estimate blood pressure using **single-channel PPG signals**
- Design and train deep learning models (LSTM / GRU-based)
- Apply model compression techniques for **TinyML deployment**
- Run real-time inference on low-power MCUs
- Enable long-term operation suitable for wearable devices

---

## System Architecture

The system consists of the following pipeline:

1. **PPG Signal Acquisition**  
   - Optical PPG sensor collects pulse waveform data
   - Sampling frequency: 100–200 Hz

2. **Signal Preprocessing**  
   - Bandpass filtering (0.5–8 Hz)
   - Window-based segmentation
   - Normalization and noise rejection

3. **TinyML Inference**  
   - Lightweight recurrent neural network
   - Integer-quantized (INT8) model
   - On-device inference without cloud dependency

4. **Result Output**  
   - Estimated Mean Arterial Pressure (MAP)
   - Displayed locally or transmitted wirelessly

---

## Dataset

- **UCI Cuffless Blood Pressure Estimation Dataset**
- Contains synchronized PPG and arterial blood pressure (ABP) signals
- Sampling rate: 125 Hz
- Data from over 900 subjects

### Preprocessing Highlights
- Sliding window length: 8 seconds (1000 samples)
- 50% overlap between windows
- Removal of low-variance and corrupted segments
- MAP calculated as the mean of ABP within each window

---

## Model Design

- Recurrent neural network architectures:
  - LSTM
  - GRU
- Input: windowed PPG time-series
- Output: Mean Arterial Pressure (regression)
- Performance metrics:
  - MAE
  - RMSE
  - Pearson correlation coefficient

Hyperparameters such as window size, number of hidden units, and learning rate were optimized to balance accuracy and generalization.

---

## Model Compression and TinyML Deployment

To enable execution on resource-constrained microcontrollers, the trained model was optimized using:

- INT8 quantization
- Weight pruning
- Operator-level optimization

### Deployment Pipeline
1. PyTorch → ONNX
2. ONNX → TensorFlow Lite
3. TFLite → C array (`model_data.h`)
4. Integrated into embedded firmware

---

## Embedded Platform

- **MCU**: nRF52840 (ARM Cortex-M4)
- **Development Environment**: Segger Embedded Studio
- **Memory Usage**:
  - Flash: ~180 KB
  - RAM: ~122 KB
- **Inference Latency**: < 200 ms
- **Update Rate**: ≥ 1 Hz

The optimized model fits within the memory and real-time constraints of the target MCU while maintaining acceptable prediction accuracy.

---

## Experimental Results

- Comparable accuracy between float32 and INT8 models
- Slight increase in MAE after quantization, within acceptable limits
- Stable real-time inference confirmed on MCU
- Quantization significantly reduced memory footprint and power consumption

---

## Power and Wearable Considerations

- Designed for battery-powered operation
- Optimized sensor duty cycle and inference frequency
- Suitable for long-term monitoring scenarios
- No continuous wireless connection required

---

## Current Status

- Model training and validation completed
- TinyML deployment verified on MCU
- Static inference tests successful
- Real-time sensor integration and live validation in progress

---

## Future Work

- Real-time PPG sensor integration
- User-specific calibration strategies
- BLE-based mobile application interface
- Extended power consumption profiling
- Long-term field testing

---

## License

This project is released under the MIT License.
