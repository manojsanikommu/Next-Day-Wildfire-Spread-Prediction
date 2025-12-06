Next-Day Wildfire Spread Prediction Using Deep Learning
*UNet-Based Segmentation Model for Disaster Management*

This repository contains the full implementation of a deep learning pipeline for predicting **next-day wildfire spread** using satellite and environmental features. The project follows a standard research workflow:

- TFRecord → NumPy preprocessing  
- Baseline tile-level models (Logistic Regression, CNN)  
- Pixel-level UNet segmentation model  
- Training, evaluation, and experiment tracking  

The goal is to explore how deep learning can support wildfire disaster management by forecasting fire regions at the **pixel level**.

---
Project Structure

```
ndws/
│
├── dataset/
│   ├── next_day_wildfire_spread_*.tfrecord
│   ├── X_train.npy
│   ├── Y_train.npy
│   ├── X_val.npy
│   ├── Y_val.npy
│
├── Screenshots     
│
├── Intermediate Update 1&2/
│   ├── data_preprocessing.py
│   ├── baseline_logisticregression.py
│   ├── baseline_CNN.py
│   ├── train_unet_small.py
│   ├── utils.py
│   
│
├── README.txt
└── requirements.txt
```

---

Step 1 — Install Dependencies

```bash
pip install -r requirements.txt
```

---

Step 2 — Download Dataset (Kaggle)

Place TFRecord files inside:

```
ndws/data/
```

---

Step 3 — Convert TFRecords → NumPy

```bash
python src/data_preprocessing.py
```

---

Step 4 — Baseline Models

### Logistic Regression

```bash
python src/baseline_logisticregression.py
```

CNN Baseline

```bash
python src/baseline_CNN.py
```

---

Step 5 — Train UNet Segmentation Model

```bash
python src/train_unet_small.py
```

---

Evaluation Metrics

- ROC-AUC, PR-AUC  
- Dice Coefficient  
- Dice,loss curves  

---

Purpose

Supports reproducible wildfire forecasting experiments and comparison between ML and deep learning.

---

