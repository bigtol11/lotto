import streamlit as st, json
from google.oauth2.service_account import Credentials
import gspread, pandas as pd, numpy as np, collections, math, random
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

# ① must be first
st.set_page_config(page_title="Lotto Predictor v40.0", layout="wide")
st.title("🎯 Lotto Prediction Web App (v40.0 GA Optimized)")

# ② load & parse secret JSON
raw = st.secrets["gcp"]["json"]
raw = raw.strip()                # 앞뒤 공백/줄바꿈 제거
sa_info = json.loads(raw)        # 여기서 JSONDecodeError 안 나야 함

# ③ Google Sheets auth
SCOPES = ["https://spreadsheets.google.com/feeds","https://www.googleapis.com/auth/drive"]
creds = Credentials.from_service_account_info(sa_info, scopes=SCOPES)
gc    = gspread.authorize(creds)
ws    = gc.open("lotto").sheet1

# ④ load data
@st.cache_data(ttl=3600)
def load_data():
    df = pd.DataFrame(ws.get_all_records())
    df.columns = ["회차"] + [f"번호{i}" for i in range(1,7)]
    return df

df   = load_data()
nums = [f"번호{i}" for i in range(1,7)]
st.write(f"▶ Loaded {len(df)} records from Google Sheet")

# ⑤ features
def coord(n): return ((n-1)%7,(n-1)//7)
traj = {}
for _,r in df.iterrows():
    d   = r["회차"]
    arr = sorted(r[nums])
    cs  = [coord(x) for x in arr]
    dts = [math.hypot(x2-x1,y2-y1) for (x1,y1),(x2,y2) in zip(cs,cs[1:])]
    traj[d] = (np.mean(dts), np.std(dts))

def build_features(draw,s=30,m=100):
    mf,sa = traj[draw]
    past   = df[df["회차"]<=draw][nums].values.flatten()
    mid    = df[(df["회차"]>draw-m)&(df["회차"]<=draw)][nums].values.flatten()
    short  = df[(df["회차"]>draw-s)&(df["회차"]<=draw)][nums].values.flatten()
    cg,cm,cs = collections.Counter(past),collections.Counter(mid),collections.Counter(short)
    Mg = max(cg.values()); Mm = max(cm.values()) if cm else 1; Ms = max(cs.values()) if cs else 1
    return np.array([[mf,sa,cg[n]/Mg,cm[n]/Mm,cs[n]/Ms] for n in range(1,46)])

# ⑥ load models (must be pre-committed under models/)
import pickle
m35  = pickle.load(open("models/v35.pkl","rb"))
m36  = pickle.load(open("models/v36.pkl","rb"))
meta = pickle.load(open("models/meta.pkl","rb"))

# ⑦ predict function
def predict_draw(dr):
    F   = build_features(dr-1)
    p35 = m35.predict_proba(F)[:,1]
    p36 = m36.predict_proba(F)[:,1]
    sp  = np.sort(p35)[::-1]
    p37 = p35 if sp[:6].mean()-sp[6:12].mean()>=0.05 else np.ones(45)/45
    pf  = meta.predict_proba(np.vstack([p35,p36,p37]).T)[:,1]
    def fit(c): return sum(pf[n-1] for n in c)
    pop = [tuple(sorted(np.random.choice(range(1,46),6,False,p=pf/pf.sum()))) for _ in range(200)]
    for _ in range(50):
        pop = sorted(pop, key=fit, reverse=True)[:50]
        new = pop.copy()
        while len(new)<200:
            a,b = random.sample(pop,2)
            c   = tuple(sorted(a[:3]+b[3:]))
            if random.random()<0.3:
                L=list(c); i=random.randrange(6)
                L[i]=np.random.choice(range(1,46),p=pf/pf.sum())
                c=tuple(sorted(set(L))[:6])
            new.append(c)
        pop=new
    final=[]
    for c in sorted(pop,key=fit,reverse=True):
        if all(len(set(c)&set(x))<5 for x in final):
            final.append(c)
            if len(final)==10: break
    return final

# ⑧ UI tabs
tab1,tab2 = st.tabs(["Predict","Backtest"])
with tab1:
    nd = int(df["회차"].max()+1)
    st.subheader(f"Predict Draw {nd}")
    if st.button("Generate"):
        out = pd.DataFrame({"Set":range(1,11),"Nums":predict_draw(nd)})
        st.dataframe(out)

with tab2:
    st.subheader("Backtest 1151→Last")
    if st.button("Run"):
        res=[]
        for d in range(1151,int(df["회차"].max())):
            hits=[len(set(s)&set(df[df["회차"]==d+1][nums].iloc[0])) for s in predict_draw(d+1)]
            res.append({"Draw":d+1,"MaxHit":max(hits)})
        bt=pd.DataFrame(res)
        st.write("Avg MaxHit:", bt["MaxHit"].mean())
        st.write("Hit≥3 Ratio:", (bt["MaxHit"]>=3).mean())
        st.dataframe(bt)
