# 🔐 머신러닝/딥러닝 기반 보안 데이터 분석

> XGBoost와 Autoencoder를 활용한 네트워크 침입 탐지 및 이상 탐지 실습

---

## 📁 실습 목록

| 실습 | 데이터셋 | 모델 | 결과 |
|------|---------|------|------|
| [01. XGBoost 침입 탐지](./01_xgboost_intrusion_detection.pdf) | NSL-KDD | XGBoost | 정확도 85%, F1-score 0.85 |
| [02. Autoencoder 이상 탐지](./02_autoencoder_anomaly_detection.pdf) | Edge-IIoTset | Autoencoder | 비지도 학습 기반 이상 탐지 한계 분석 |

---

## 01. XGBoost 기반 침입 탐지

| 항목 | 내용 |
|------|------|
| 데이터셋 | NSL-KDD (네트워크 침입 탐지 공개 데이터셋) |
| 모델 | XGBoost (eXtreme Gradient Boosting) |
| 분류 방식 | 이진 분류 (정상 / 공격) |
| 환경 | Python, xgboost, scikit-learn, pandas |

**주요 결과**

| 지표 | 정상 | 공격 |
|------|------|------|
| 정밀도 | 0.75 | 0.97 |
| 재현율 | 0.97 | 0.75 |
| F1-score | 0.85 | 0.85 |
| 전체 정확도 | 85% | - |

📄 [보고서 보기](./01_xgboost_intrusion_detection.pdf) | 💻 [코드 보기](./xgboost_intrusion_detection.py)

---

## 02. Autoencoder 기반 이상 탐지

| 항목 | 내용 |
|------|------|
| 데이터셋 | Edge-IIoTset (IIoT 환경 보안 데이터셋) |
| 모델 | Autoencoder (비지도 학습) |
| 탐지 방식 | 복원 오차(MSE) 기반 이상 판단 |
| 환경 | Python, TensorFlow, Keras, scikit-learn |

**주요 결과 및 분석**
- 공격 데이터 재현율 5%로 탐지 성능 낮음
- 원인: 데이터 불균형 (공격 85%), 복원 오차 분포 겹침, 임계값 설정 한계
- 개선 방안: SMOTE 적용, ROC/PR 곡선 기반 임계값 재설정, Isolation Forest 등 비교 분석

📄 [보고서 보기](./02_autoencoder_anomaly_detection.pdf) | 💻 [코드 보기](./autoencoder_anomaly_detection.py)

---

## 🛠️ 사용 기술

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-189fdd?logoColor=white)

---

> 보안데이터분석 / 인공지능빅데이터분석 수업 실습 과제입니다.
