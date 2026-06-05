import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# 1. 데이터 불러오기 및 전처리
data = pd.read_csv('ML-EdgeIIoT-dataset.csv')  # 파일 경로는 실행 환경에 맞게 수정

# 결측치 제거
data = data.dropna()

# 레이블 확인
print("고유 레이블 값:", data['Attack_label'].unique())
print("레이블 분포:", data['Attack_label'].value_counts())
print("Attack_label 데이터 타입:", data['Attack_label'].dtype)

# 수치형 특성 선택 (Attack_label과 Attack_type 제외)
data_features = data.drop(columns=['Attack_label', 'Attack_type'])
numeric_columns = data_features.select_dtypes(include=['float64', 'int64']).columns
print(f"수치형 열 개수: {len(numeric_columns)}")
data_numeric = data_features[numeric_columns]

# 정규화
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data_numeric)

# Edge-IIoTset 데이터셋에서 정상 데이터 레이블 확인 후 설정
normal_label = 0  # 정상 데이터의 레이블 값

# 정상 데이터 추출
normal_data = data[data['Attack_label'] == normal_label]
print(f"정상 데이터 샘플 수: {len(normal_data)}")

if len(normal_data) == 0:
    print("정상 데이터가 없습니다. 다른 레이블로 재시도합니다.")
    most_common_label = data['Attack_label'].value_counts().idxmax()
    print(f"가장 흔한 레이블: {most_common_label}")
    normal_data = data[data['Attack_label'] == most_common_label]

normal_features = normal_data.drop(columns=['Attack_label', 'Attack_type'])
normal_numeric = normal_features[numeric_columns]
normal_scaled = scaler.transform(normal_numeric)

# 학습/검증/테스트 분할
X_train, X_temp = train_test_split(normal_scaled, test_size=0.3, random_state=42)
X_val, X_test_normal = train_test_split(X_temp, test_size=0.5, random_state=42)

# 공격 데이터 포함한 최종 테스트셋 구성
attack_data = data[data['Attack_label'] != normal_label].copy()
attack_features = attack_data.drop(columns=['Attack_label', 'Attack_type'])
attack_numeric = attack_features[numeric_columns]
X_test_attack = scaler.transform(attack_numeric)
y_test_attack = np.ones(len(attack_data))
y_test_normal = np.zeros(len(X_test_normal))

X_test = np.vstack([X_test_normal, X_test_attack])
y_test = np.concatenate([y_test_normal, y_test_attack])

# 2. Autoencoder 모델 구성
input_dim = X_train.shape[1]
input_layer = Input(shape=(input_dim,))
encoded = Dense(64, activation='relu')(input_layer)
encoded = Dense(32, activation='relu')(encoded)
encoded = Dense(16, activation='relu')(encoded)
decoded = Dense(32, activation='relu')(encoded)
decoded = Dense(64, activation='relu')(decoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)

autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# 3. 모델 학습
es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = autoencoder.fit(
    X_train, X_train,
    epochs=100,
    batch_size=128,
    validation_data=(X_val, X_val),
    callbacks=[es],
    verbose=1
)

# 4. 복원 오차 계산 및 이상 판단
reconstructions = autoencoder.predict(X_test)
mse = np.mean(np.power(X_test - reconstructions, 2), axis=1)

# 임계값 설정 (정상 데이터 복원 오차의 95번째 퍼센타일)
threshold = np.percentile(mse, 95)
predictions = (mse > threshold).astype(int)

# 5. 성능 평가
print("\n======= 분류 보고서 =======")
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

# 6. 복원 오차 분포 시각화
plt.figure(figsize=(10, 6))
plt.hist(mse[y_test == 0], bins=50, alpha=0.6, label='Normal')
plt.hist(mse[y_test == 1], bins=50, alpha=0.6, label='Attack')
plt.axvline(threshold, color='r', linestyle='--', label='Threshold')
plt.legend()
plt.title('Reconstruction Error Distribution')
plt.xlabel('Mean Squared Error')
plt.ylabel('Frequency')
plt.show()
