import os
import pandas as pd

def store_data(path: str, new_row: dict) -> None:

    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame()

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(path, index=False)
