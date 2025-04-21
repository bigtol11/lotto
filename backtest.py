import pandas as pd
from predict import train_models, predict_draw

def run_backtest(path, start: int, end: int):
    df = pd.read_excel(path, engine='openpyxl')
    mlp35, mlp36, meta = train_models(path)
    nums = [f'번호{i}' for i in range(1,7)]
    rec = []
    for draw in range(start+1, end+1):
        sets = predict_draw(df[df['회차']<draw], mlp35, mlp36, meta, draw)
        actual = set(df[df['회차']==draw][nums].iloc[0])
        rec.append({'회차':draw, 'max_hits': max(len(set(s)&actual) for s in sets)})
    return pd.DataFrame(rec)
