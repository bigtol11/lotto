# streamlit_app.py
import streamlit as st

# ⚠️ 반드시 최상단에 둡니다
st.set_page_config(page_title="Lotto Predictor v40.0", layout="wide")

import pandas as pd
import numpy as np
import collections, math, random
import pickle
import gspread
from google.oauth2.service_account import Credentials
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

# 1) 시트 인증
SCOPES = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive"
]
creds = Credentials.from_service_account_info(
    st.secrets["gcp_service_account"],
    scopes=SCOPES
)
gc = gspread.authorize(creds)
ws = gc.open("lotto").sheet1

# 2) 데이터 로드
@st.cache_data(ttl=3600)
def load_data() -> pd.DataFrame:
    data = ws.get_all_records()
    df = pd.DataFrame(data)
    df.columns = ["회차"] + [f"번호{i}" for i in range(1,7)]
    return df

df = load_data()
nums = [f"번호{i}" for i in range(1,7)]

# 3) 궤적(feature) 계산
def coord(n): return ((n-1)%7, (n-1)//7)
traj = {}
for _, row in df.iterrows():
    d = row["회차"]
    arr = sorted([row[n] for n in nums])
    coords = [coord(n) for n in arr]
    dists = [math.hypot(x2-x1,y2-y1)
             for (x1,y1),(x2,y2) in zip(coords, coords[1:])]
    traj[d] = (np.mean(dists), np.std(dists))

def build_features(draw, s=30, m=100):
    mf, sa = traj[draw]
    past  = df[df["회차"]<=draw][nums].values.flatten()
    mid   = df[(df["회차"]>draw-m)&(df["회차"]<=draw)][nums].values.flatten()
    short = df[(df["회차"]>draw-s)&(df["회차"]<=draw)][nums].values.flatten()
    cg, cm, cs = collections.Counter(past), collections.Counter(mid), collections.Counter(short)
    Mg, Mm, Ms = max(cg.values()), max(cm.values()) if cm else 1, max(cs.values()) if cs else 1
    return np.array([[mf, sa, cg[n]/Mg, cm[n]/Mm, cs[n]/Ms] for n in range(1,46)])

# 4) 모델 로드
with open("models/v35.pkl","rb") as f: m35 = pickle.load(f)
with open("models/v36.pkl","rb") as f: m36 = pickle.load(f)
with open("models/meta.pkl","rb") as f: meta = pickle.load(f)

# 5) 예측 함수
def predict_draw(draw:int):
    p35 = m35.predict_proba(build_features(draw-1))[:,1]
    p36 = m36.predict_proba(build_features(draw-1))[:,1]
    sp = np.sort(p35)[::-1]
    p37 = p35 if sp[:6].mean()-sp[6:12].mean()>=0.05 else np.ones(45)/45
    pf = meta.predict_proba(np.vstack([p35,p36,p37]).T)[:,1]
    # GA 최적화
    def fitness(c): return sum(pf[n-1] for n in c)
    pop = [tuple(sorted(np.random.choice(range(1,46),6,False,p=pf/pf.sum()))) for _ in range(200)]
    for _ in range(50):
        pop = sorted(pop, key=lambda c:-fitness(c))[:50]
        new = pop.copy()
        while len(new)<200:
            a,b = random.sample(pop,2)
            child = tuple(sorted(a[:3]+b[3:]))
            if random.random()<0.3:
                lst = list(child)
                i = random.randrange(6)
                lst[i] = np.random.choice(range(1,46),p=pf/pf.sum())
                child = tuple(sorted(set(lst))[:6])
            new.append(child)
        pop = new
    final = []
    for c in sorted(pop, key=lambda c:-fitness(c)):
        if all(len(set(c)&set(x))<5 for x in final):
            final.append(c)
        if len(final)==10: break
    return final

# 6) UI
st.title("🎯 Lotto Prediction Web App (v40.0 GA Optimized)")

tab1, tab2 = st.tabs(["예측","백테스트"])
with tab1:
    draw = st.number_input("예측 회차", min_value=int(df["회차"].max()+1), value=int(df["회차"].max()+1))
    if st.button("10세트 생성"):
        sets = predict_draw(draw)
        st.dataframe(pd.DataFrame({"세트":range(1,11),"조합":sets}))

with tab2:
    st.write("1151회차부터 마지막 회차까지 누적 백테스트")
    if st.button("백테스트 시작"):
        results=[]
        for d in range(1151, int(df["회차"].max())):
            pred = predict_draw(d)
            actual = set(df[df["회차"]==d+1][nums].iloc[0])
            hits = [len(set(s)&actual) for s in pred]
            results.append({"회차":d+1, "max_hits":max(hits)})
        bt = pd.DataFrame(results)
        st.write("평균 최고 적중수:", bt["max_hits"].mean())
        st.write(">3개 적중률:", (bt["max_hits"]>=3).mean())

