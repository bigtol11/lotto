# streamlit_app.py
import streamlit as st
import pandas as pd, numpy as np, collections, math, random
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

# —— 페이지 설정은 가장 첫 줄에! —— 
st.set_page_config(page_title="Lotto Predictor v40.0", layout="wide")
st.title("🎯 Lotto Prediction Web App (v40.0 GA Optimized)")

# 1) 구글 시트 인증 & 로드
scope = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive",
]
creds = ServiceAccountCredentials.from_json_keyfile_dict(
    st.secrets["gcp_service_account"], scope
)
gc = gspread.authorize(creds)
ws = gc.open("lotto").sheet1

@st.cache_data(ttl=600)
def load_sheet() -> pd.DataFrame:
    data = ws.get_all_records()
    df = pd.DataFrame(data)
    df.columns = ["회차"] + [f"번호{i}" for i in range(1,7)]
    return df

df = load_sheet()
nums = [f"번호{i}" for i in range(1,7)]

# 2) 궤적(feature) 계산
def coord(n): return ((n-1)%7, (n-1)//7)
traj = {}
for _, row in df.iterrows():
    d = row["회차"]
    arr = sorted(row[nums].tolist())
    coords = [coord(n) for n in arr]
    dists = [math.hypot(x2-x1, y2-y1)
             for (x1,y1),(x2,y2) in zip(coords,coords[1:])]
    traj[d] = (np.mean(dists), np.std(dists))

def build_features(draw, s=30, m=100):
    mf, sa = traj[draw]
    past  = df[df["회차"]<=draw][nums].values.flatten()
    mid   = df[(df["회차"]>draw-m)&(df["회차"]<=draw)][nums].values.flatten()
    short = df[(df["회차"]>draw-s)&(df["회차"]<=draw)][nums].values.flatten()
    cg, cm, cs = collections.Counter(past), collections.Counter(mid), collections.Counter(short)
    Mg, Mm, Ms = max(cg.values()), max(cm.values()) if cm else 1, max(cs.values()) if cs else 1
    return np.array([[mf, sa, cg[n]/Mg, cm[n]/Mm, cs[n]/Ms] for n in range(1,46)])

# 3) 시드 고정
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# 4) 모델 학습 (v35, v36, 메타 v38)
# — v35
mlp35 = MLPClassifier(hidden_layer_sizes=(64,32), max_iter=300, random_state=SEED)
X35, y35 = [], []
for d in range(1, df["회차"].max()):
    X35.append(build_features(d))
    win = set(df[df["회차"]==d+1][nums].iloc[0])
    y35.extend([1 if n in win else 0 for n in range(1,46)])
mlp35.fit(np.vstack(X35), np.array(y35))

# — v36 (오버샘플링)
mlp36 = MLPClassifier(hidden_layer_sizes=(64,32), max_iter=300, random_state=SEED)
X36, y36, w36 = [], [], []
for d in range(1, df["회차"].max()):
    F = build_features(d)
    win = set(df[df["회차"]==d+1][nums].iloc[0])
    for i,n in enumerate(range(1,46)):
        X36.append(F[i]); y36.append(1 if n in win else 0)
        w36.append(3 if n in win else 1)
mlp36.fit(np.repeat(np.array(X36), w36, axis=0),
          np.repeat(np.array(y36), w36, axis=0))

# — 메타 RandomForest (v38)
meta_X, meta_y = [], []
for d in range(2, df["회차"].max()+1):
    p35 = mlp35.predict_proba(build_features(d-1))[:,1]
    p36 = mlp36.predict_proba(build_features(d-1))[:,1]
    sp = np.sort(p35)[::-1]
    p37 = p35 if sp[:6].mean()-sp[6:12].mean()>=0.05 else np.ones(45)/45
    meta_X.append(np.vstack([p35,p36,p37]).T)
    now = set(df[df["회차"]==d][nums].iloc[0])
    meta_y.extend([1 if n in now else 0 for n in range(1,46)])
meta = RandomForestClassifier(n_estimators=100, random_state=SEED)
meta.fit(np.vstack(meta_X), np.array(meta_y))

# 5) 유전 알고리즘으로 최종 10세트 생성 함수
def fitness(c): return sum(p_final[n-1] for n in c)
def make_child(a,b,pf):
    ch = list(a[:3]+b[3:])
    if random.random()<0.3:
        i = random.randrange(6)
        ch[i] = np.random.choice(range(1,46), p=pf/pf.sum())
    uniq = tuple(sorted(set(ch)))
    while len(uniq)<6:
        x = np.random.choice(range(1,46), p=pf/pf.sum())
        if x not in uniq: uniq = tuple(sorted(uniq+(x,)))
    return uniq

def predict_draw(draw):
    global p_final
    p35 = mlp35.predict_proba(build_features(draw-1))[:,1]
    p36 = mlp36.predict_proba(build_features(draw-1))[:,1]
    sp = np.sort(p35)[::-1]
    p37 = p35 if sp[:6].mean()-sp[6:12].mean()>=0.05 else np.ones(45)/45
    p_final = meta.predict_proba(np.vstack([p35,p36,p37]).T)[:,1]

    # 초기 population
    pop = [tuple(sorted(np.random.choice(range(1,46),6,False,p=p_final/p_final.sum())))
           for _ in range(200)]
    # 진화
    for _ in range(50):
        pop = sorted(pop, key=lambda c:-fitness(c))[:50]
        new = pop.copy()
        while len(new)<200:
            a,b = random.sample(pop,2)
            new.append(make_child(a,b,p_final))
        pop = new

    # 상위 10세트, 중복 5개 이상 배제
    final = []
    for c in sorted(pop, key=lambda c:-fitness(c)):
        if all(len(set(c)&set(x))<5 for x in final):
            final.append(c)
            if len(final)==10: break
    return final

# 6) 예측 및 백테스트 UI
st.header("▶ Next Draw Prediction")
draw_next = df["회차"].max()+1
sets = predict_draw(draw_next)
st.write(f"{draw_next}회차 예측 10세트:")
st.dataframe(pd.DataFrame({"세트":range(1,11),"조합":sets}))

st.header("▶ Cumulative Backtest")
start = st.number_input("Start draw", min_value=1, max_value=df["회차"].max()-1, value=1150)
if st.button("Run backtest"):
    results = []
    for d in range(start, df["회차"].max()):
        pred = predict_draw(d)
        actual = set(df[df["회차"]==d][nums].iloc[0])
        max_hit = max(len(set(c)&actual) for c in pred)
        results.append({"회차":d, "max_hits":max_hit})
    bt = pd.DataFrame(results)
    st.write("평균 max 적중 수:", bt["max_hits"].mean())
    st.write("3개 이상 적중 비율:", (bt["max_hits"]>=3).mean())
    st.dataframe(bt)
