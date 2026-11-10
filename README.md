# Classifying Heartbeats from ECG Signals

**Machine Learning — Project 03 (Fall 2026)**
FAST NUCES, CFD Campus

A neural network built **entirely from scratch in PyTorch** that reads a single heartbeat from an ECG recording and classifies it into one of five beat types. No ready-made layers, no optimiser library and no ready-made loss function are used anywhere — every weight, gradient and update is written by hand.

**Group members:** Shahzaib Hasnain (23F-6098) and Asad Ali (23F-6065)

---

## 1. The problem and the dataset

Each row of the dataset is one heartbeat: **187 signal values** (the shape of the beat) plus one label in the last column. The data comes from the MIT-BIH Arrhythmia Database, resampled to a fixed length (dataset: *Heartbeat Categorization Dataset*, Kaggle). The signals are already scaled, so they are used as they are.

| Class | Meaning | Training rows | Test rows |
|---|---|---|---|
| N | Normal beat | 72,471 | 18,118 |
| S | Supraventricular premature beat | 2,223 | 556 |
| V | Premature ventricular contraction | 5,788 | 1,448 |
| F | Fusion of ventricular and normal beat | 641 | 162 |
| Q | Unknown / paced beat | 6,431 | 1,608 |
| | **Total** | **87,554** | **21,892** |

Signal values in the training file range from a minimum of ______ to a maximum of ______.

The classes are heavily imbalanced: class N alone makes up most of the data, so a model that always answers "N" would already score **82.76%** on the test set (18,118 / 21,892) without learning anything. Accuracy is therefore always read together with the per-class accuracies and the balanced accuracy.

