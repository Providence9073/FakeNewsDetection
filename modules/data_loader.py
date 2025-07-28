import pandas as pd
import os

def load_data(filename, label):
    filepath = os.path.join(os.path.dirname(__file__), '..', 'data', filename)
    df = pd.read_csv(filepath)
    df['label'] = label
    return df[['text', 'label']]