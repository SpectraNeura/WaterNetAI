import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 고전 모델용 라이브러리 (ARIMA, SARIMA)
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from sklearn.preprocessing import StandardScaler

# device 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== 기본 파라미터 설정 =====
LOOKBACK = 60  # 시퀀스 길이 (예: 60분)
FEATURES = ['Q1', 'Q2', 'Q3', 'Q4', 'P1']  # 사용할 입력 피처
TARGET = 'P1_flag'  # 타겟 (압력계 1번 이상 여부)
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-3


# ===== 1. 데이터 로드 및 전처리 =====
def load_and_preprocess_data():
    # datasets 폴더 내에 있는 TRAIN_A.csv와 TRAIN_B.csv 파일 읽기
    train_a_path = os.path.join('datasets', 'train', 'TRAIN_A.csv')
    train_b_path = os.path.join('datasets', 'train', 'TRAIN_B.csv')

    df_A = pd.read_csv(train_a_path)
    df_B = pd.read_csv(train_b_path)

    # timestamp 컬럼 파싱 (예: "24/05/27 00:00" 형태)
    df_A['timestamp'] = pd.to_datetime(df_A['timestamp'], format='%y/%m/%d %H:%M')
    df_B['timestamp'] = pd.to_datetime(df_B['timestamp'], format='%y/%m/%d %H:%M')

    # 두 데이터 병합 후 timestamp 기준 정렬
    df_train = pd.concat([df_A, df_B], ignore_index=True)
    df_train.sort_values('timestamp', inplace=True)
    df_train.reset_index(drop=True, inplace=True)

    # 입력 피처 스케일링 (선택한 FEATURES만 사용)
    scaler = StandardScaler()
    # 여기서는 DataFrame 그대로 전달하여 컬럼 이름 유지
    df_train[FEATURES] = scaler.fit_transform(df_train[FEATURES])

    return df_train, scaler


# ===== 2. 시퀀스 데이터 생성 함수 =====
def create_sequences(df, lookback, feature_cols, target_col):
    X, y = [], []
    for i in range(len(df) - lookback):
        X.append(df.loc[i:i + lookback - 1, feature_cols].values)
        y.append(df.loc[i + lookback, target_col])
    return np.array(X), np.array(y)


# ===== 3. PyTorch 딥러닝 모델 정의 =====

# 1) RNN 모델
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]  # 마지막 시점의 출력 사용
        out = self.fc(out)
        return torch.sigmoid(out)


# 2) LSTM 모델
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return torch.sigmoid(out)


# 3) GRU 모델
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return torch.sigmoid(out)


# 4) Transformer 기반 모델 (batch_first 옵션 추가)
class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model=64, nhead=8, num_layers=2, dim_feedforward=128, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.input_linear = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # batch_first 옵션 추가
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (batch, seq, features)
        x = self.input_linear(x)  # (batch, seq, d_model)
        x = self.transformer_encoder(x)  # (batch, seq, d_model)
        x = x[:, -1, :]  # 마지막 시점의 출력 사용
        x = self.fc(x)
        return torch.sigmoid(x)


# 5) CNN + Transformer 모델 (batch_first 옵션 추가)
class CNN_TransformerModel(nn.Module):
    def __init__(self, input_size, num_filters=32, kernel_size=3,
                 d_model=64, nhead=8, num_layers=2, dim_feedforward=128, dropout=0.1):
        super(CNN_TransformerModel, self).__init__()
        # Conv1d 입력: (batch, channels, seq)
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=num_filters,
                               kernel_size=kernel_size, padding=kernel_size // 2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=num_filters, out_channels=d_model,
                               kernel_size=kernel_size, padding=kernel_size // 2)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # batch_first 옵션 추가
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (batch, seq, features) -> Conv1d expects (batch, features, seq)
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)  # (batch, d_model, seq)
        x = x.transpose(1, 2)  # (batch, seq, d_model)
        x = self.transformer_encoder(x)  # (batch, seq, d_model)
        x = x[:, -1, :]  # 마지막 시점의 출력 사용
        x = self.fc(x)
        return torch.sigmoid(x)


# 딥러닝 모델 딕셔너리
deep_models = {
    'RNN': RNNModel(input_size=len(FEATURES)),
    'LSTM': LSTMModel(input_size=len(FEATURES)),
    'GRU': GRUModel(input_size=len(FEATURES)),
    'Transformer': TransformerModel(input_size=len(FEATURES)),
    'CNN_Transformer': CNN_TransformerModel(input_size=len(FEATURES))
}


# ===== 4. 모델 학습 함수 (PyTorch) =====
def train_dl_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LR):
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_x.size(0)
        val_loss /= len(val_loader.dataset)
        print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss} | Val Loss: {val_loss}")
    return model


# 딥러닝 모델 예측 함수 (T+1분 예측; 단일 타겟 -> threshold 0.5)
def predict_with_dl_model(model, test_seq):
    model.eval()
    with torch.no_grad():
        # test_seq: (LOOKBACK, features) -> (1, LOOKBACK, features)
        x = torch.tensor(test_seq, dtype=torch.float32).unsqueeze(0).to(device)
        output = model(x)
        pred = output.item()
    return int(pred > 0.5)


# ===== 5. 고전 모델 (ARIMA, SARIMA) 학습 및 예측 =====
def train_arima_model(series):
    model = ARIMA(series, order=(1, 1, 1))
    model_fit = model.fit()
    return model_fit


def train_sarima_model(series):
    model = SARIMAX(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    model_fit = model.fit(disp=False)
    return model_fit


def predict_with_classical_model(model_fit):
    forecast = model_fit.forecast(steps=1)[0]
    return int(forecast > 0.5)


if __name__ == '__main__':
    # ===== 6. 전체 데이터 로드 및 시퀀스 생성 =====
    df_train, scaler = load_and_preprocess_data()
    X, y = create_sequences(df_train, LOOKBACK, FEATURES, TARGET)

    # 시간 순서를 유지하며 학습/검증 데이터 분할
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    # 텐서 변환
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(-1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ===== 7. 고전 모델 학습 (ARIMA, SARIMA) =====
    series_train = df_train[TARGET].values
    arima_model_fit = train_arima_model(series_train)
    sarima_model_fit = train_sarima_model(series_train)

    # ===== 8. 딥러닝 모델 학습 =====
    trained_models = {}
    for name, model in deep_models.items():
        print(f"\n==== Training {name} Model ====")
        trained_model = train_dl_model(model, train_loader, val_loader)
        trained_models[name] = trained_model
        print(f"==== Finished Training {name} Model ====")

    # ===== 9. 테스트 데이터 예측 및 결과 CSV 파일 저장 =====
    # 테스트 데이터: datasets/test/C 및 datasets/test/D 폴더 내 CSV 파일들
    test_files_C = glob.glob(os.path.join('datasets', 'test', 'C', '*.csv'))
    test_files_D = glob.glob(os.path.join('datasets', 'test', 'D', '*.csv'))
    test_files = test_files_C + test_files_D

    # 예측 결과 저장 딕셔너리 (딥러닝 및 고전 모델)
    results = {model_name: [] for model_name in list(deep_models.keys()) + ['ARIMA', 'SARIMA']}

    for file in test_files:
        try:
            # engine='python' 옵션 추가하여 파일 읽기
            test_df = pd.read_csv(file, engine='python')
        except Exception as e:
            print(f"Error reading {file}: {e}")
            continue  # 에러 발생 시 해당 파일 건너뛰기

        # test 파일에서 선택한 FEATURES만 추출하고, DataFrame 그대로 scaler.transform 호출
        test_seq = scaler.transform(test_df[FEATURES])

        # 딥러닝 모델 예측: 마지막 LOOKBACK 분 데이터 사용 (데이터 길이 부족 시 0 패딩)
        if len(test_seq) >= LOOKBACK:
            test_seq = test_seq[-LOOKBACK:]
        else:
            pad_length = LOOKBACK - len(test_seq)
            pad = np.zeros((pad_length, len(FEATURES)))
            test_seq = np.vstack([pad, test_seq])

        # 각 딥러닝 모델 예측
        for name, model in trained_models.items():
            pred_flag = predict_with_dl_model(model, test_seq)
            results[name].append({
                'ID': os.path.basename(file).split('.')[0],
                'flag_list': [pred_flag]
            })

        # 고전 모델 예측 (TEST 파일에 TARGET 컬럼이 없으므로 학습 데이터 기반 예시 사용)
        try:
            pred_arima = predict_with_classical_model(arima_model_fit)
        except Exception as e:
            print(f"ARIMA 예측 에러: {e}")
            pred_arima = 0
        results['ARIMA'].append({
            'ID': os.path.basename(file).split('.')[0],
            'flag_list': [pred_arima]
        })

        try:
            pred_sarima = predict_with_classical_model(sarima_model_fit)
        except Exception as e:
            print(f"SARIMA 예측 에러: {e}")
            pred_sarima = 0
        results['SARIMA'].append({
            'ID': os.path.basename(file).split('.')[0],
            'flag_list': [pred_sarima]
        })

    # 각 모델별로 submission CSV 파일 생성
    for model_name, predictions in results.items():
        df_pred = pd.DataFrame(predictions)
        out_file = f'submission_{model_name}.csv'
        df_pred.to_csv(out_file, index=False)
        print(f"{model_name} 예측 결과가 {out_file}로 저장되었습니다.")
