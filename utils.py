
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import ttest_ind, pearsonr

def load_pcl5_data(path='data/pcl5_responses.csv', cutoff=32):
    pcl5 = pd.read_csv(path)
    pcl5["Total Score"] = pcl5.sum(axis=1)
    pcl5["Diagnosis"] = 0
    pcl5.loc[:24, "Diagnosis"] = 1

    clusters = {
        "Cluster 1": [f"Item_{i}" for i in range(1, 6)],
        "Cluster 2": [f"Item_{i}" for i in range(6, 8)],
        "Cluster 3": [f"Item_{i}" for i in range(8, 15)],
        "Cluster 4": [f"Item_{i}" for i in range(15, 21)],
    }
    for name, items in clusters.items():
        pcl5[name] = pcl5[items].sum(axis=1)

    pcl5["Mean-Diagnosis"] = (pcl5["Total Score"] >= cutoff).astype(int)
    return pcl5

def compute_summary(df, columns, label="Group"):
    desc = pd.DataFrame(index=["Mean", "Std.Dev.", "Median"])
    for col in columns:
        desc[col] = [df[col].mean(), df[col].std(), df[col].median()]
    print(f"--- {label} ---")
    return desc.round(3)

def negative_log_likelihood_q(data, alpha, beta):
    QA, QB = 0.6, 0.6
    nll = 0
    for choice, reward in zip(data['Choice'], data['Reward']):
        pA = np.exp(beta * QA) / (np.exp(beta * QA) + np.exp(beta * QB))
        prob = pA if choice == 'A' else 1 - pA
        nll -= np.log(prob + 1e-9)
        delta = reward - (QA if choice == 'A' else QB)
        if choice == 'A':
            QA += alpha * delta
        else:
            QB += alpha * delta
    return nll

def fit_model_q(data, start_params=(0.3, 8.0)):
    result = minimize(lambda x: negative_log_likelihood_q(data, *x),
                      start_params, method='Nelder-Mead')
    return result.x, result.fun
