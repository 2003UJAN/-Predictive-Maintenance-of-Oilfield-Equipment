import pandas as pd

def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        raise Exception(f"File not found: {filepath}")
