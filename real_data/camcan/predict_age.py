from glob import glob
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA

from osl_dynamics.utils.misc import load, save


def get_demographics():
    df = pd.DataFrame(columns=["data_path", "participant_id"])
    df["data_path"] = sorted(
        glob("/well/woolrich/projects/camcan/winter23/src/sub*/sflip_parc.npy")
    )
    df["participant_id"] = df["data_path"].apply(
        lambda x: os.path.basename(os.path.dirname(x))
    )
    demo_df = pd.read_csv("/well/woolrich/projects/camcan/participants.tsv", sep="\t")
    df = df.merge(demo_df, on="participant_id")
    return df


df = get_demographics()
targets = df["age"]

hmm_id = 1
hive_id = 1

results_dir = "results"
hmm_dir = f"results/hmm/run{hmm_id}"
hive_dir = f"results/hive/run{hive_id}"

hmm_covs = load(f"{hmm_dir}/dual_estimates/covs.npy")
hive_covs = load(f"{hive_dir}/inf_params/covs.npy")

n_subjects = hmm_covs.shape[0]
n_states = hmm_covs.shape[1]
n_channels = hmm_covs.shape[2]

m, n = np.tril_indices(n_channels)
hmm_features = hmm_covs[:, :, m, n].reshape(n_subjects, -1)
hive_features = hive_covs[:, :, m, n].reshape(n_subjects, -1)

pipe = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("pca", PCA(whiten=True)),
        ("reg", Ridge()),
    ]
)
param_grid = {
    "reg__alpha": np.logspace(-5, 3, 9),
    "pca__n_components": [5, 20, 50, 100],
}

scores = defaultdict(list)
kf = KFold(n_splits=20, shuffle=True, random_state=0)
for train_index, test_index in tqdm(
    kf.split(targets), desc="Cross validation", total=20
):
    targets_train, targets_test = targets[train_index], targets[test_index]
    hmm_features_train, hmm_features_test = (
        hmm_features[train_index],
        hmm_features[test_index],
    )
    hive_features_train, hive_features_test = (
        hive_features[train_index],
        hive_features[test_index],
    )

    hmm_pipe = GridSearchCV(pipe, param_grid, cv=5, n_jobs=16)
    hmm_pipe.fit(hmm_features_train, targets_train)
    scores["hmm"].append(hmm_pipe.score(hmm_features_test, targets_test))

    hive_pipe = GridSearchCV(pipe, param_grid, cv=5, n_jobs=16)
    hive_pipe.fit(hive_features_train, targets_train)
    scores["hive"].append(hive_pipe.score(hive_features_test, targets_test))

save("results/age_prediction_scores.pkl", scores)

print("HMM (mean, std): ", np.mean(scores["hmm"]), np.std(scores["hmm"]))
print("HIVE (mean, std): ", np.mean(scores["hive"]), np.std(scores["hive"]))
