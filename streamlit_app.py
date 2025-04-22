import streamlit as st

# ▶ 반드시 맨 위에 한 줄로!
st.set_page_config(page_title="Lotto Predictor v40.0", layout="wide")

# ▶ (선택) 로드된 Secret 키 확인
st.write("Loaded secrets keys:", list(st.secrets.keys()))

import pandas as pd
import numpy as np
import collections, math, random
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

st.title("🎯 Lotto Prediction Web App (v40.0 GA Optimized)")

# 1) 구글 시트 인증
scope = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive",
]
creds = ServiceAccountCredentials.from_json_keyfile_dict(
    st.secrets["gcp_service_account"], scope
)
gc = gspread.authorize(creds)
ws = gc.open("lotto").sheet1

@st.cache_data(ttl=3600)
def load_sheet() -> pd.DataFrame:
    data = ws.get_all_records()
    df = pd.DataFrame(data)
    df.columns = ["회차"] + [f"번호{i}" for i in range(1,7)]
    return df

df = load_sheet()
nums = [f"번호{i}" for i in range(1,7)]

# 2) 피처 계산
def coord(n):
    return ((n-1)%7, (n-1)//7)

@st.cache_data(ttl=3600)
def compute_traj(df):
    traj = {}
    for _, row in df.iterrows():
        d = row["회차"]
        arr = sorted(row[nums].tolist())
        coords = [coord(n) for n in arr]
        dists = [math.hypot(x2-x1, y2-y1)
                 for (x1,y1),(x2,y2) in zip(coords, coords[1:])]
        traj[d] = (np.mean(dists), np.std(dists))
    return traj

def build_features(df, traj, draw, s=30, m=100):
    if draw not in traj:
        draw = max(traj.keys())
    mf, sa = traj[draw]
    past  = df[df["회차"]<=draw][nums].values.flatten()
    mid   = df[(df["회차"]>draw-m)&(df["회차"]<=draw)][nums].values.flatten()
    short = df[(df["회차"]>draw-s)&(df["회차"]<=draw)][nums].values.flatten()
    cg, cm, cs = (
        collections.Counter(past),
        collections.Counter(mid),
        collections.Counter(short),
    )
    Mg, Mm, Ms = max(cg.values()), max(cm.values()) if cm else 1, max(cs.values()) if cs else 1
    return np.array([[mf, sa, cg[n]/Mg, cm[n]/Mm, cs[n]/Ms] for n in range(1,46)])

# 3) 모델 학습
@st.cache_resource
def train_models(df):
    traj = compute_traj(df)
    max_draw = df["회차"].max() - 1

    # v35
    mlp35 = MLPClassifier(hidden_layer_sizes=(64,32), max_iter=300, random_state=42)
    X35, y35 = [], []
    for d in range(1, max_draw+1):
        X35.append(build_features(df, traj, d))
        win = set(df[df["회차"]==d+1][nums].iloc[0])
        y35.extend([1 if n in win else 0 for n in range(1,46)])
    mlp35.fit(np.vstack(X35), np.array(y35))

    # v36
    mlp36, X36, y36, w36 = MLPClassifier(hidden_layer_sizes=(64,32), max_iter=300, random_state=42), [], [], []
    for d in range(1, max_draw+1):
        feats = build_features(df, traj, d)
        win = set(df[df["회차"]==d+1][nums].iloc[0])
        for i,n in enumerate(range(1,46)):
            X36.append(feats[i])
            y36.append(1 if n in win else 0)
            w36.append(3 if n in win else 1)
    mlp36.fit(np.repeat(np.array(X36), w36, axis=0),
              np.repeat(np.array(y36), w36, axis=0))

    # v38 (Stacking)
    meta_X, meta_y = [], []
    for d in range(2, max_draw+2):
        p35 = mlp35.predict_proba(build_features(df, traj, d-1))[:,1]
        p36 = mlp36.predict_proba(build_features(df, traj, d-1))[:,1]
        sp  = np.sort(p35)[::-1]
        p37 = p35 if sp[:6].mean() - sp[6:12].mean() >= 0.05 else np.ones(45)/45
        meta_X.append(np.vstack([p35,p36,p37]).T)
        win = set(df[df["회차"]==d][nums].iloc[0])
        meta_y.extend([1 if n in win else 0 for n in range(1,46)])
    meta = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    meta.fit(np.vstack(meta_X), np.array(meta_y))

    return mlp35, mlp36, meta

mlp35, mlp36, meta = train_models(df)

# 4) GA 최적화 예측
def predict_draw(df, mlp35, mlp36, meta, draw):
    traj = compute_traj(df)
    p35 = mlp35.predict_proba(build_features(df, traj, draw-1))[:,1]
    p36 = mlp36.predict_proba(build_features(df, traj, draw-1))[:,1]
    sp  = np.sort(p35)[::-1]
    p37 = p35 if sp[:6].mean() - sp[6:12].mean() >= 0.05 else np.ones(45)/45
    pf  = meta.predict_proba(np.vstack([p35,p36,p37]).T)[:,1]

    def fitness(c): return sum(pf[n-1] for n in c)
    def make_child(a,b):
        child = list(a[:3] + b[3:])
        if random.random()<0.3:
            i = random.randrange(6)
            child[i] = np.random.choice(range(1,46), p=pf/pf.sum())
        u = sorted(set(child))
        while len(u)<6:
            x = np.random.choice(range(1,46), p=pf/pf.sum())
            if x not in u: u.append(x)
        return tuple(sorted(u))

    pop = [tuple(sorted(np.random.choice(range(1,46),6,False,p=pf/pf.sum()))) for _ in range(200)]
    for _ in range(50):
        pop = sorted(pop, key=lambda c:-fitness(c))[:50]
        new = pop.copy()
        while len(new)<200:
            new.append(make_child(*random.sample(pop,2)))
        pop = new

    final = []
    for c in sorted(pop, key=lambda c:-fitness(c)):
        if all(len(set(c)&set(x))<5 for x in final):
            final.append(c)
        if len(final)==10: break
    idx=0
    while len(final)<10:
        if sorted(pop, key=lambda c:-fitness(c))[idx] not in final:
            final.append(sorted(pop, key=lambda c:-fitness(c))[idx])
        idx+=1
    return final

# 5) UI – 백테스트 or 다음 회차 예측
mode = st.sidebar.selectbox("🔧 Mode", ["Backtest","Next Draw"])

if mode=="Backtest":
    start = st.sidebar.number_input("Start Draw (≥1)", 1, int(df["회차"].max())-1, 1)
    end   = st.sidebar.number_input("End Draw (≤max)", start+1, int(df["회차"].max()), start+1)
    if st.sidebar.button("▶ Run Backtest"):
        records=[]
        for d in range(start+1, end+1):
            sets   = predict_draw(df[df["회차"]<d], mlp35, mlp36, meta, d)
            actual = set(df[df["회차"]==d][nums].iloc[0])
            hits   = [len(actual & set(s)) for s in sets]
            records.append({"회차":d, "max_hits":max(hits)})
        bt = pd.DataFrame(records)
        st.dataframe(bt, use_container_width=True)
        st.write("▶ Average max hits:", bt["max_hits"].mean())

else:
    nd = int(df["회차"].max())+1
    st.sidebar.write(f"Next Draw → {nd}")
    if st.sidebar.button("▶ Predict Next Draw"):
        sets = predict_draw(df, mlp35, mlp36, meta, nd)
        st.table(pd.DataFrame({"Set":range(1,11), "Combination":sets}))
        inp = st.text_input("Enter actual numbers (comma separated)", "")
        if st.button("💾 Save Actual"):
            row = [nd] + [int(x.strip()) for x in inp.split(",")]
            ws.append_row(row)
            st.success(f"Saved actual {row[1:]} for draw {nd}")
