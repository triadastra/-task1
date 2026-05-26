"""Quick diagnostic for label distribution."""
import sys, os
os.chdir("d:/0521数据清洗")
sys.path.insert(0, ".")
import numpy as np, datetime as dt
from pathlib import Path
from ml.train_xgb_multibuy_policy import (
    detect_feature_columns, load_clean_rows,
    build_supervised_dataset, label_by_return, LONG_WINDOW,
)

csv_path = Path("brk_b_data/brk_b_daily.csv")
fc = detect_feature_columns(csv_path)
dates, values, _ = load_clean_rows(csv_path, fc)
ci = fc.index("Close")
hi = fc.index("High") if "High" in fc else ci
li = fc.index("Low")  if "Low"  in fc else ci

print("LONG_WINDOW =", LONG_WINDOW)
print("feature cols:", fc)

x, y, idx = build_supervised_dataset(
    dates, values, ci, hi, li,
    dt.date(2000, 1, 1), dt.date(2025, 3, 10),
    14, 1, 0.003, 0.008,
)
print(f"y shape={y.shape}  dtype={y.dtype}")
print(f"y[:5] = {y[:5]}")
print(f"y range: {y.min():.4f} to {y.max():.4f}")
print(f"x feature dims: {x.shape[1]}")

yc = np.array([label_by_return(float(r), 0.003, 0.008) for r in y])
vals, cnts = np.unique(yc, return_counts=True)
for v, c in zip(vals, cnts):
    print(f"  class {int(v)}: {int(c)}")
