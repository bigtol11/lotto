# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import collections, math, random
import gspread
from google.oauth2.service_account import Credentials
import pickle

# ─────────── 1) 페이지 설정 (반드시 첫 번째!) ────────────────────────
st.set_page_config(page_title="Lotto Predictor v40.0", layout="wide")
st.title("🎯 Lotto Prediction Web App (v40.0 GA Optimized)")

# ─────────── 2) 서비스 계정 키 (JSON) 직접 삽입 ────────────────────
service_account_info = {
    "type": "service_account",
    "project_id": "make-442407",
    "private_key_id": "2a3e0e485c4e7846318b2bfefd1f6d13eb7d446d",
    "private_key": "-----BEGIN PRIVATE KEY-----\n"
        "MIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQDXaUhlqXNYXwHE\n"
        "BxzV+CxP6Z7n97owb62M5QGEmes/IoGKFDM93pLF2TzHaa/8GIrUdL7KrVx7zQ4h\n"
        "irYMcXEiN+pZDVf4+CswNurLadQtk7dB/29UqueDJYxP8N+L6Ukhpe/VCrwcW1Fy\n"
        "iD6SNv+GU4+Z0kPKG8mxeZAK9hsta8cvW2YaxYvSx1dO/E4+AZhy3vse7X2oSDe5\n"
        "9rhl2QdXgvq3JoKoTykzCz7s7DEU3MSew4gk3yBPyfEU2sUumBuH3Nr9HuTjSV+f\n"
        "qvAY3dAGxFDCe4dy4OPCLVLOCK/x3SewOw3Vvx7syLQORTgUn6dcnj5CLa/9YTa2\n"
        "Suc82YP3AgMBAAECggEAVQKEIzaxwgzQbhOegiSsHCdm4kWl8WqJU6KmrDVwHNzZ\n"
        "wWvCYya8xp66OTpQzMzEsR6XkvXCm0rryjnrKVL1olrtvZIiByutI5xwobEUnp3+\n"
        "gumy/ndp4RxG2N+G4TjB9yj80pcncItrQ6dYBiz1P4YnlD1iKlc4DDWcrm68f/og\n"
        "uY+PkJ7pvKkTZ0fO15o8CVTpQ5L+AxWAkErhQ+MwGrjUUY8DVru+vyo+2WKJHDuF\n"
        "vfvPRXbcoGEwlVc6OphIplC9h4XWpl7ExsV/7X0cSqW1LZIlgP5ySrhfH+QXYGLs\n"
        "Law29osovAh+8bKrt2Olr61fMNEiCijdaxrbFYHCwQKBgQD1uuQqDg7pP/TH6v9q\n"
        "oxYTmPA0X1zNLE6xbEuC4rVo80NBVHcaqAKegF4seipx9CdXyPsPvGefaYS92j7Z\n"
        "u+Ts8gQ0AnrMfBUbAer/+2Ze06D5eQLkPbNtf9jMbcwEE0CSl2tceR8udfhtcUKT\n"
        "DjyJ8OaLoV/DAr4/OrFf+UFJtwKBgQDgagFtCJ2mDpNZoBtfw1phpyaj8RP3oFIJ\n"
        "fr6FCn8SjUb76W9U8J1V25SCfmIgOICg+nM9dQVxD2OQHhbpvgyapuqsXvWxlt1K\n"
        "tAIFUSYzWo/wnbIxwQxjtpyVshPUPoV+kV8XKVJUkmCnrB+ln4mDkO8VgOeNqEKs\n"
        "DzexiQqXwQKBgGERElAKfZllyiuuiHZ3NaFIKJqHkQD7H5q2TJ3HMCHk9gw4cVP2\n"
        "ShKSYqDvIRUifOgQXBw7MDOoWucj7u/TaPqwHzjsQdXErhGdEFdN14Jd1pi1VI8U\n"
        "UGxQtDMMrCpv8HH3nlFJBygzMY8JzmKInSFgJ7HAbTN7Qet4I9jlfQTBAoGBAKi4\n"
        "Cgpnh97o5m9jqDD/NlxkxvBKt7BcoFDzMVnk4lSYUt3iSwmZPTDnvWe+jRecY1ij\n"
        "8zApYsX1w+z+MkvZzrAW/ihJ3H5/5i/b1gkZUZcaZ02HwgkWErKFAISrNa1EtCjM\n"
        "Mqm/L17WDkUBa2mh4ElCFf4cw/oEntutNToMxiIBAoGBAJVRDWZdG1esJaDK1Pvd\n"
        "F9Tbl+Pem4UM41cnXqvaBNUugNzTxdo25QH0Cn33SVrAjWlWV2T2lMin7bY4Yf8+\n"
        "bjuQLoE9Z54OFKC4aXDTd8LnUy2FTwM4Gt13CgarvCHPtwvaBqh5cUt24wSvRulv\n"
        "KysdCPqOagfQUF0n+61ocbP\n"
        "-----END PRIVATE KEY-----\n",
    "client_email": "lotto-app-sa@make-442407.iam.gserviceaccount.com",
    "client_id": "112609156907569631402",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/lotto-app-sa%40make-442407.iam.gserviceaccount.com",
    "universe_domain": "googleapis.com"
}

# ─────────── 3) 구글 시트 인증 ─────────────────────────────────────────
scope = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive",
]
creds = Credentials.from_service_account_info(service_account_info, scopes=scope)
gc = gspread.authorize(creds)
ws = gc.open("lotto").sheet1

@st.cache_data(ttl=3600)
def load_sheet() -> pd.DataFrame:
    df = pd.DataFrame(ws.get_all_records())
    df.columns = ["회차"] + [f"번호{i}" for i in range(1,7)]
    return df

df = load_sheet()
nums = [f"번호{i}" for i in range(1,7)]

# ─────────── 4) 패턴(feature) 계산 ────────────────────────────────────
def coord(n): return ((n-1)%7, (n-1)//7)

def compute_traj():
    traj = {}
    for _, row in df.iterrows():
        d = row["회차"]
        arr = sorted(row[nums].tolist())
        coords = [coord(x) for x in arr]
        dists = [
            math.hypot(x2-x1, y2-y1)
            for (x1,y1),(x2,y2) in zip(coords, coords[1:])
        ]
        traj[d] = (np.mean(dists), np.std(dists))
    return traj

traj = compute_traj()

def build_features(draw, s=30, m=100):
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

# ─────────── 5) 모델 로드 (pickle) ────────────────────────────────────
with open("models/v35.pkl","rb") as f: m35 = pickle.load(f)
with open("models/v36.pkl","rb") as f: m36 = pickle.load(f)
with open("models/meta.pkl","rb")   as f: meta = pickle.load(f)

def predict_draw(draw):
    p35 = m35.predict_proba(build_features(draw-1))[:,1]
    p36 = m36.predict_proba(build_features(draw-1))[:,1]
    sp  = np.sort(p35)[::-1]
    p37 = p35 if sp[:6].mean()-sp[6:12].mean()>=0.05 else np.ones(45)/45
    pf  = meta.predict_proba(np.vstack([p35,p36,p37]).T)[:,1]

    def fit(c): return sum(pf[n-1] for n in c)
    pop = [
        tuple(sorted(
            np.random.choice(range(1,46),6,False,p=pf/pf.sum())))
        for _ in range(200)
    ]
    for _ in range(50):
        pop = sorted(pop, key=fit, reverse=True)[:50]
        new = pop.copy()
        while len(new)<200:
            a,b = random.sample(pop,2)
            ch   = tuple(sorted(a[:3]+b[3:]))
            if random.random()<0.3:
                lst = list(ch); i = random.randrange(6)
                lst[i] = np.random.choice(range(1,46), p=pf/pf.sum())
                ch = tuple(sorted(set(lst))[:6])
            new.append(ch)
        pop = new

    final=[]
    for c in sorted(pop, key=fit, reverse=True):
        if all(len(set(c)&set(x))<5 for x in final):
            final.append(c)
            if len(final)==10: break
    return final

# ─────────── 6) 누적 백테스트 & 다음 회차 예측 UI ──────────────────────
st.header("▶ 누적 백테스트 (1151→최종)")
results = []
for d in range(1151, df["회차"].max()+1):
    sets   = predict_draw(d)
    actual = set(df[df["회차"]==d][nums].iloc[0])
    maxhit = max(len(set(s)&actual) for s in sets)
    results.append({"회차":d, "max_hits":maxhit})
bt = pd.DataFrame(results)
st.write("평균 최대 적중 수:", bt["max_hits"].mean())
st.write("3개 이상 적중 비율:", (bt["max_hits"]>=3).mean())

st.header("▶ 다음 회차 예측")
nd    = df["회차"].max()+1
preds = predict_draw(nd)
st.write(f"{nd}회차 예측 10세트:")
st.table(pd.DataFrame({"Set":range(1,11), "Combination":preds}))
