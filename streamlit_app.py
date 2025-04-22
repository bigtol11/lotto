# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import collections, math, random
import gspread
from google.oauth2.service_account import Credentials
import pickle

# 디버그: 로드된 secret 키 확인
st.write("Loaded secrets keys:", list(st.secrets.keys()))

# 페이지 설정 (반드시 다른 st.* 호출 전에)
st.set_page_config(page_title="Lotto Predictor v40.0", layout="wide")
st.title("🎯 Lotto Prediction Web App (v40.0 GA Optimized)")

# 구글 시트 인증
scope = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive",
]
creds = Credentials.from_service_account_info(
    st.secrets["gcp_service_account"],
    scopes=scope,
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

def coord(n):
    return ((n-1)%7, (n-1)//7)

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

traj = compute_traj(df)

def build_features(draw, s=30, m=100):
    mf, sa = traj[draw]
    past  = df[df["회차"]<=draw][nums].values.flatten()
    mid   = df[(df["회차"]>draw-m)&(df["회차"]<=draw)][nums].values.flatten()
    short = df[(df["회차"]>draw-s)&(df["회차"]<=draw)][nums].values.flatten()
    cg, cm, cs = (collections.Counter(past),
                  collections.Counter(mid),
                  collections.Counter(short))
    Mg, Mm, Ms = max(cg.values()), max(cm.values()) if cm else 1, max(cs.values()) if cs else 1
    return np.array([[mf, sa, cg[n]/Mg, cm[n]/Mm, cs[n]/Ms] for n in range(1,46)])

# 학습된 모델 로드
with open("models/v35.pkl","rb") as f: model35 = pickle.load(f)
with open("models/v36.pkl","rb") as f: model36 = pickle.load(f)
with open("models/meta.pkl","rb") as f:   meta    = pickle.load(f)

def predict_draw(draw):
    p35 = model35.predict_proba(build_features(draw-1))[:,1]
    p36 = model36.predict_proba(build_features(draw-1))[:,1]
    sp  = np.sort(p35)[::-1]
    p37 = p35 if sp[:6].mean()-sp[6:12].mean()>=0.05 else np.ones(45)/45
    pfinal = meta.predict_proba(np.vstack([p35,p36,p37]).T)[:,1]

    def fitness(c): return sum(pfinal[n-1] for n in c)
    pop = [tuple(sorted(
        np.random.choice(range(1,46),6,False,p=pfinal/pfinal.sum())))
           for _ in range(200)]
    for _ in range(50):
        pop = sorted(pop, key=fitness, reverse=True)[:50]
        new = pop.copy()
        while len(new)<200:
            a,b = random.sample(pop,2)
            child = tuple(sorted(a[:3]+b[3:]))
            if random.random()<0.3:
                lst = list(child)
                i = random.randrange(6)
                lst[i] = np.random.choice(range(1,46), p=pfinal/pfinal.sum())
                child = tuple(sorted(set(lst))[:6])
            new.append(child)
        pop = new

    final = []
    for c in sorted(pop, key=fitness, reverse=True):
        if all(len(set(c)&set(x))<5 for x in final):
            final.append(c)
            if len(final)==10: break
    return final

# 누적 백테스트
st.header("▶ 누적 백테스트 (1151회차부터)")
results=[]
for d in range(1151, df["회차"].max()+1):
    sets   = predict_draw(d)
    actual = set(df[df["회차"]==d][nums].iloc[0])
    maxhit = max(len(set(s)&actual) for s in sets)
    results.append({"회차":d, "max_hits":maxhit})
bt = pd.DataFrame(results)
st.write("평균 최대 적중 수:", bt["max_hits"].mean())
st.write("3개 이상 적중 비율:", (bt["max_hits"]>=3).mean())

# 다음 회차 예측
st.header("▶ 다음 회차 예측")
next_draw = df["회차"].max()+1
preds = predict_draw(next_draw)
st.write(f"{next_draw}회차 예측 10세트:")
st.dataframe(pd.DataFrame({"세트":range(1,11), "조합":preds}))
