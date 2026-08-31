import pandas as pd
from datasets import Dataset

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
train_data = Dataset.from_pandas(pd.read_csv(os.path.join(BASE_DIR, "train_data.csv")))
val_data = Dataset.from_pandas(pd.read_csv(os.path.join(BASE_DIR, "val_data.csv")))
print("load successful")
