# streamlit_app.py

import streamlit as st
from google.oauth2.service_account import Credentials
import gspread
import pandas as pd
import numpy as np
import collections, math, random
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

# ── 1) Streamlit 페이지 설정 ────────────────────────────────
st.set_page_config(page_title="Lotto Predictor v40.0", layout="wide")
st.title("🎯 Lotto Prediction Web App (v40.0 GA Optimized)")

# ── 2) 구글 서비스 계정 인증 ────────────────────────────────
#    key.json 파일이 레포 루트에 있어야 합니다.
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]
creds = Credentials.from_service_account_file("key.json", scopes=SCOPES)
gc    = gspread.authorize(creds)
sheet = gc.open("lotto").sheet1

@st.cache_data(ttl=3600)
def load_sheet() -> pd.DataFrame:
    data = sheet.get_all_records()
    df   = pd.DataFrame(data)
    df.columns = ["회차"] + [f"번호{i}" for i in range(1,7)]
    return df

df   = load_sheet()
nums = [f"번호{i}" for i in range(1,7)]

# ── 3) 피쳐 계산 함수 ───────────────────────────────────────
def coord(n): 
    return ((n-1) % 7, (n-1) // 7)

def compute_traj(df: pd.DataFrame) -> dict:
    traj = {}
    for _, row in df.iterrows():
        d    = row["회차"]
        arr  = sorted(row[nums].tolist())
        crd  = [coord(n) for n in arr]
        dists = [
            math.hypot(x2-x1, y2-y1)
            for (x1,y1),(x2,y2) in zip(crd, crd[1:])
        ]
        traj[d] = (np.mean(dists), np.std(dists))
    return traj

traj = compute_traj(df)

def build_features(draw:int, s=30, m=100) -> np.ndarray:
    mf, sa = traj[draw]
    past   = df[df["회차"]<=draw][nums].values.flatten()
    mid    = df[(df["회차"]>draw-m)&(df["회차"]<=draw)][nums].values.flatten()
    short  = df[(df["회차"]>draw-s)&(df["회차"]<=draw)][nums].values.flatten()
    cg, cm, cs = (
        collections.Counter(past),
        collections.Counter(mid),
        collections.Counter(short),
    )
    Mg, Mm, Ms = max(cg.values()), max(cm.values()) if cm else 1, max(cs.values()) if cs else 1
    feats = []
    for n in range(1,46):
        feats.append([
            mf,
            sa,
            cg[n]/Mg,
            cm[n]/Mm,
            cs[n]/Ms,
        ])
    return np.array(feats)

# ── 4) 모델 학습 (v35, v36, meta(v38)) ────────────────────
#    (처음 한 번만 돌리고, 중간에 주석 처리 가능)

# 시드 고정
random.seed(42); np.random.seed(42)

# v35 MLP
mlp35 = MLPClassifier((64,32), max_iter=300, random_state=42)
X35, y35 = [], []
for d in range(1, df["회차"].max()):
    X35.append(build_features(d))
    win = set(df[df["회차"]==d+1][nums].iloc[0])
    y35.extend([1 if n in win else 0 for n in range(1,46)])
mlp35.fit(np.vstack(X35), np.array(y35))

# v36 MLP + 오버샘플링
mlp36 = MLPClassifier((64,32), max_iter=300, random_state=42)
X36, y36, w36 = [], [], []
for d in range(1, df["회차"].max()):
    F   = build_features(d)
    win = set(df[df["회차"]==d+1][nums].iloc[0])
    for i,n in enumerate(range(1,46)):
        X36.append(F[i])
        y36.append(1 if n in win else 0)
        w36.append(3 if n in win else 1)
mlp36.fit(
    np.repeat(np.array(X36), w36, axis=0),
    np.repeat(np.array(y36), w36, axis=0),
)

# v38 메타 모델 (RF)
meta_X, meta_y = [], []
for d in range(2, df["회차"].max()+1):
    p35 = mlp35.predict_proba(build_features(d-1))[:,1]
    p36 = mlp36.predict_proba(build_features(d-1))[:,1]
    sp  = np.sort(p35)[::-1]
    p37 = p35 if sp[:6].mean()-sp[6:12].mean()>=0.05 else np.ones(45)/45
    meta_X.append(np.vstack([p35,p36,p37]).T)
    now = set(df[df["회차"]==d][nums].iloc[0])
    meta_y.extend([1 if n in now else 0 for n in range(1,46)])
meta = RandomForestClassifier(100, random_state=42)
meta.fit(np.vstack(meta_X), np.array(meta_y))

# ── 5) 다음 회차 예측 + GA 최적화 ─────────────────────────
def predict_draw(draw:int):
    p35 = mlp35.predict_proba(build_features(draw-1))[:,1]
    p36 = mlp36.predict_proba(build_features(draw-1))[:,1]
    sp  = np.sort(p35)[::-1]
    p37 = p35 if sp[:6].mean()-sp[6:12].mean()>=0.05 else np.ones(45)/45
    p_final = meta.predict_proba(np.vstack([p35,p36,p37]).T)[:,1]

    # GA
    def fitness(c): return sum(p_final[n-1] for n in c)
    pop = [
        tuple(sorted(np.random.choice(range(1,46),6,False,p=p_final/p_final.sum())))
        for _ in range(200)
    ]
    for _ in range(50):
        pop = sorted(pop, key=lambda c:-fitness(c))[:50]
        new = pop.copy()
        while len(new)<200:
            a,b = random.sample(pop,2)
            child = tuple(sorted(a[:3]+b[3:]))
            if random.random()<0.3:
                lst = list(child)
                i   = random.randrange(6)
                lst[i] = np.random.choice(range(1,46),p=p_final/p_final.sum())
                child = tuple(sorted(lst))[:6]
            new.append(child)
        pop = new
    final = []
    for c in sorted(pop, key=lambda c:-fitness(c)):
        if all(len(set(c)&set(x))<5 for x in final):
            final.append(c)
        if len(final)==10:
            break
    return final

# ── 6) UI: 백테스트 or 최신 예측 ───────────────────────────
mode = st.sidebar.selectbox("⏱️ 모드 선택", ["백테스트", "최신 예측"])
if mode=="백테스트":
    n0 = st.sidebar.number_input("시작 회차", min_value=1, max_value=int(df["회차"].max())-1, value=1)
    n1 = st.sidebar.number_input("끝 회차",   min_value=n0+1, max_value=int(df["회차"].max()), value=int(df["회차"].max()))
    if st.sidebar.button("▶ 백테스트 실행"):
        results=[]
        for d in range(n0, n1):
            pred = predict_draw(d)
            actual = set(df[df["회차"]==d+1][nums].iloc[0])
            hits = [len(set(s)&actual) for s in pred]
            results.append({"회차":d+1, "max_hits":max(hits)})
        bt = pd.DataFrame(results)
        st.write("■ 평균 최대 적중 수:", bt["max_hits"].mean())
        st.write("■ 3개 이상 적중 비율:", (bt["max_hits"]>=3).mean())
        st.dataframe(bt)

else:
    draw = df["회차"].max()+1
    if st.button(f"▶ {draw}회차 10세트 예측"):
        sets = predict_draw(draw)
        st.write(f"▶ {draw}회차 예측 결과:")
        st.table(pd.DataFrame({"세트":range(1,11), "조합":sets}))
