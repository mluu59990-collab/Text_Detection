import os
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CRNN_ROOT = os.path.join(BASE_DIR, "data", "crnn_recognition")

CSV_PATH = os.path.join(CRNN_ROOT, "labels.csv")
TRAIN_CSV = os.path.join(CRNN_ROOT, "train_labels.csv")
VAL_CSV = os.path.join(CRNN_ROOT, "val_labels.csv")

df = pd.read_csv(CSV_PATH)

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

train_df.to_csv(TRAIN_CSV, index=False)
val_df.to_csv(VAL_CSV, index=False)

print("Train samples:", len(train_df))
print("Val samples:", len(val_df))