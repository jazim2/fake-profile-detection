import os
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(filepath=None):
    if filepath is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(base_dir, 'dataset.csv')
    
    df = pd.read_csv(filepath)

    # Map binary Yes/No columns
    binary_map = {'Yes': 1, 'No': 0}
    df['profile_pic'] = df['profile_pic'].map(binary_map)
    df['extern_url'] = df['extern_url'].map(binary_map)
    df['private'] = df['private'].map(binary_map)

    # Map categorical 'sim_name_username'
    # Check unique values first if unsure, but observed: 'No match', 'Partial match', 'Full match' (and maybe 'Match'?)
    # Let's use a safe mapping or label encoding
    sim_map = {'No match': 0, 'Partial match': 1, 'Full match': 2}
    # Use replace to handle potential unobserved values gracefully or just map
    df['sim_name_username'] = df['sim_name_username'].map(sim_map).fillna(0) # Default to 0 if unknown

    # Select features and target
    # Dropping any potentially irrelevant columns if necessary, but all look relevant
    # The first column in view_file output was an index '0, 1, 2...', but CSVs often have that.
    # Let's inspect columns to be sure. 
    # Based on previous `view_file`: 
    # 1: ,fake,profile_pic,ratio_numlen_username...
    # It seems there is an unnamed index column at the start. existing pandas read_csv might pick it up as 'Unnamed: 0'
    
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    X = df.drop(columns=['fake'])
    y = df['fake']

    return train_test_split(X, y, test_size=0.2, random_state=42)

def clean_data(df):
    # Any additional cleaning if needed
    return df
