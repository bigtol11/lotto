import numpy as np
import random
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from utils import load_data, compute_traj, build_features

def train_models(path: str):
    df = load_data(path)
    traj = compute_traj(df)
    max_draw = df['회차'].max() - 1
    # v35
    mlp35 = MLPClassifier((64,32), max_iter=300, random_state=42)
    X35, y35 = [], []
    for d in range(1, max_draw+1):
        X35.append(build_features(df, traj, d))
        win = set(df[df['회차']==d+1][[f'번호{i}' for i in range(1,7)]].iloc[0])
        y35.extend([1 if n in win else 0 for n in range(1,46)])
    mlp35.fit(np.vstack(X35), np.array(y35))
    # v36...
    # (이하 생략 – 앞서 가이드된 코드 그대로 붙여넣으세요)
    return mlp35, mlp36, meta

def predict_draw(df, mlp35, mlp36, meta, draw: int):
    # v40.0 GA 로직: p35, p36, p37 계산 후 GA 최적화
    # (앞서 가이드된 predict_draw 함수 전체 복사하세요)
    return final_combinations  # 10-튜플 리스트
