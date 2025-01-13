import pandas as pd
import numpy as np
import random

import os

# Define the columns and their meanings
columns = [
    'uid',               # User ID
    'st_mids',           # Super Topic MIDs
    'act_cnt',           # Action Count
    'check_cnt',         # Check Count
    'ts_cnt',            # Timestamp Count
    'st_act_cnt',        # Super Topic Action Count
    'tab_vv',            # Tab View Count
    'tab_mids',          # Tab MIDs
    'tab_times',         # Tab Times
    'label',             # Labels
]

# Generate synthetic data
np.random.seed(42)  # For reproducibility
data = {
    'uid': [f'user_{i}' for i in range(1, 501)],  # Unique user IDs
    'st_mids': [random.randint(1, 100) for _ in range(500)],  # Random Super Topic MIDs
    'act_cnt': np.random.randint(0, 100, size=500),  # Random action counts
    'check_cnt': np.random.randint(0, 50, size=500),  # Random check counts
    'ts_cnt': np.random.randint(0, 200, size=500),  # Random timestamp counts
    'st_act_cnt': np.random.randint(0, 150, size=500),  # Random Super Topic action counts
    'tab_vv': np.random.randint(0, 300, size=500),  # Random tab view counts
    'tab_mids': [random.randint(1, 100) for _ in range(500)],  # Random tab MIDs
    'tab_times': np.random.randint(0, 500, size=500),  # Random tab times
    'label': np.random.randint(0, 2, size=500),
}

# Create a DataFrame
df = pd.DataFrame(data, columns=columns)

# Save to CSV
data_root = 'debug/data/2nd_tab/'
if not os.path.exists(data_root):
    os.makedirs(data_root)
df.to_csv(f'{data_root}/training_data.csv', index=False)

print(f"CSV file '{data_root}/training_data.csv' created successfully!")
