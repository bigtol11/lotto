import pandas as pd
import numpy as np
import collections, math

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, engine='openpyxl')
    df.columns = ['회차'] + [f'번호{i}' for i in range(1,7)]
    return df

def coord(n: int) -> tuple:
    return ((n-1)%7, (n-1)//7)

def compute_traj(df: pd.DataFrame) -> dict:
    traj = {}
    nums = [f'번호{i}' for i in range(1,7)]
    for _, row in df.iterrows():
        d = row['회차']
        arr = sorted(row[nums].tolist())
        coords = [coord(n) for n in arr]
        dists = [math.hypot(x2-x1, y2-y1)
                 for (x1,y1),(x2,y2) in zip(coords, coords[1:])]
        traj[d] = (np.mean(dists), np.std(dists))
    return traj

def build_features(df, traj: dict, draw: int, s=30, m=100) -> np.ndarray:
    nums = [f'번호{i}' for i in range(1,7)]
    if draw not in traj:
        draw = max(traj.keys())
    mf, sa = traj[draw]
    past  = df[df['회차']<=draw][nums].values.flatten()
    mid   = df[(df['회차']>draw-m)&(df['회차']<=draw)][nums].values.flatten()
    short = df[(df['회차']>draw-s)&(df['회차']<=draw)][nums].values.flatten()
    cg, cm, cs = collections.Counter(past), collections.Counter(mid), collections.Counter(short)
    Mg, Mm, Ms = max(cg.values()), max(cm.values()) if cm else 1, max(cs.values()) if cs else 1
    return np.array([[mf, sa, cg[n]/Mg, cm[n]/Mm, cs[n]/Ms] for n in range(1,46)])
