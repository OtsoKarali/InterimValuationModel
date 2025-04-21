import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
np.random.seed(42)

def create_synthetic_data(n_samples=7697, buyout=True):
    data = {
        'investment_id': [str(i) for i in range(n_samples)],
        'interim_multiple': np.random.lognormal(mean=0, sigma=1, size=n_samples),
        'staleness_freq': np.random.uniform(0, 1, size=n_samples),
        'markdown_freq': np.random.uniform(0, 1, size=n_samples),
        'year': np.random.randint(1, 6, size=n_samples),
        'fund_type': ['buyout' if buyout else 'vc'] * n_samples
    }
    if buyout:
        intercept = [1.590, 1.763, 1.336, 1.059, 0.772]
        coef_interim = [-0.073, -0.169, -0.088, -0.087, -0.042]
        coef_stale = [-0.338, -0.657, -0.459, -0.306, -0.083]
        coef_markdown = [-0.830, -1.557, -1.451, -1.215, -1.042]
    else:
        intercept = [0.593, 1.169, 0.825, 0.303, 0.743]
        coef_interim = [-0.341, -0.194, -0.163, 0.041, -0.049]
        coef_stale = [0.264, -0.645, -0.504, -0.161, -0.632]
        coef_markdown = [0.206, -1.331, -0.789, -0.640, -1.324]
    
    exit_minus_interim = np.zeros(n_samples)
    for i in range(n_samples):
        yr = data['year'][i] - 1
        exit_minus_interim[i] = (
            intercept[yr] +
            coef_interim[yr] * data['interim_multiple'][i] +
            coef_stale[yr] * data['staleness_freq'][i] +
            coef_markdown[yr] * data['markdown_freq'][i] +
            np.random.normal(0, 1)
        )
    data['exit_multiple'] = data['interim_multiple'] + exit_minus_interim
    data['valuation_increase'] = (data['exit_multiple'] > data['interim_multiple']).astype(int)
    return pd.DataFrame(data)

# Generate data
logging.info("Generating synthetic data...")
buyout_data = create_synthetic_data(n_samples=4395, buyout=True)
vc_data = create_synthetic_data(n_samples=3374, buyout=False)
data = pd.concat([buyout_data, vc_data], ignore_index=True)

# Save to SQLite
logging.info("Saving data to SQLite...")
engine = create_engine('sqlite:///pe_data.db')
data.to_sql('investments', engine, if_exists='replace', index=False)
logging.info("Data saved to pe_data.db, table: investments")