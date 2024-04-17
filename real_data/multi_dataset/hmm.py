from sys import argv

if len(argv) != 2:
    print(f"Please pass the run id, e.g. python {argv[0]}.py 1")
    exit()
id = argv[1]

import os
from glob import glob

import pickle
import numpy as np
import pandas as pd

from osl_dynamics import run_pipeline
from osl_dynamics.data import Data
from osl_dynamics.inference import tf_ops
from osl_dynamics.utils import plotting

tf_ops.select_gpu(1)
tf_ops.gpu_growth()


def match_subjects():
    """Match the demographics distribution of each dataset."""

    # Get subject ids from each dataset
    notts_subjects = [
        os.path.basename(f)[-6:]
        for f in sorted(glob("/well/woolrich/projects/mrc_meguk/notts/ec/src/sub-*"))
        if os.path.exists(f + "/sflip_parc-raw.fif")
    ]
    camcan_subjects = [
        os.path.basename(f)
        for f in sorted(glob("/well/woolrich/projects/camcan/winter23/src/sub*"))
        if os.path.exists(f + "/sflip_parc.npy")
    ]

    # Load the demographic data
    notts_demo = pd.read_csv(
        "/well/woolrich/projects/mrc_meguk/raw/Demographics/nottingham_demographics_anonymised.csv"
    )
    notts_demo.rename({"gender": "sex"}, inplace=True, axis=1)
    camcan_demo = pd.read_csv(
        "/well/woolrich/projects/camcan/participants.tsv", sep="\t"
    )

    # Match the demographics
    notts_demo = notts_demo[notts_demo["id"].isin(notts_subjects)]
    notts_groups = notts_demo.groupby(["age_range", "sex"]).size()

    # Propcess the camcan demographics
    # turn sex to lower case and bin the ages
    camcan_demo["sex"] = camcan_demo["sex"].map(str.lower)
    camcan_demo = camcan_demo[camcan_demo["participant_id"].isin(camcan_subjects)]
    bins = [18, 24, 30, 36, 42, 48, 54, 60, 100]
    labels = ["{0}-{1}".format(bins[i], bins[i + 1]) for i in range(len(bins) - 2)] + [
        "60+"
    ]
    camcan_demo["age_range"] = pd.cut(
        camcan_demo["age"], bins, right=False, labels=labels
    )

    # Match the demographics (Cam-CAN has more subjects than notts in all groups)
    camcan_matched_subjects = []
    for age in labels[:-1]:
        for sex in ["male", "female"]:
            n_subjects = notts_groups[age][sex]
            camcan_matched_subjects.extend(
                camcan_demo[
                    np.all(
                        [camcan_demo["age_range"] == age, camcan_demo["sex"] == sex],
                        axis=0,
                    )
                ]
                .participant_id.iloc[:n_subjects]
                .to_list()
            )

    # There's no male subjects in the 60+ group in notts
    n_subjects = notts_groups["60+"]["female"]
    camcan_matched_subjects.extend(
        camcan_demo[
            np.all(
                [camcan_demo["age_range"] == "60+", camcan_demo["sex"] == "female"],
                axis=0,
            )
        ]
        .participant_id.iloc[:n_subjects]
        .to_list()
    )

    # Now slice the camcan demo to only include the matched subjects
    camcan_demo = camcan_demo[
        camcan_demo["participant_id"].isin(camcan_matched_subjects)
    ]

    df = pd.DataFrame(columns=["id", "dataset", "age_range", "sex", "data_path"])
    df["id"] = notts_demo["id"].to_list() + camcan_demo["participant_id"].to_list()
    df["dataset"] = ["notts"] * len(notts_demo) + ["camcan"] * len(camcan_demo)
    df["age_range"] = (
        notts_demo["age_range"].to_list() + camcan_demo["age_range"].to_list()
    )
    df.age_range = pd.Categorical(df.age_range, categories=labels, ordered=True)
    df["sex"] = notts_demo["sex"].to_list() + camcan_demo["sex"].to_list()

    notts_data_dir = "/well/woolrich/projects/mrc_meguk/notts/ec/src"
    camcan_data_dir = "/well/woolrich/projects/camcan/winter23/src"
    df["data_path"] = [
        f"{notts_data_dir}/sub-{s}/sflip_parc-raw.fif"
        for s in notts_demo["id"].to_list()
    ] + [
        f"{camcan_data_dir}/{s}/sflip_parc.npy"
        for s in camcan_demo["participant_id"].to_list()
    ]
    return df


def load_data(store_dir, use_tfrecord=True, buffer_size=2000, n_jobs=16):
    """Load data."""
    df = match_subjects()
    training_data = Data(
        df.data_path.to_list(),
        picks="misc",
        sampling_frequency=250,
        reject_by_annotation="omit",
        use_tfrecord=use_tfrecord,
        buffer_size=buffer_size,
        n_jobs=n_jobs,
        store_dir=store_dir,
    )
    methods = {
        "tde_pca": {"n_embeddings": 15, "n_pca_components": 80},
        "standardize": {},
    }
    training_data.prepare(methods)
    return training_data


def plot_loss(data, output_dir):
    """Plot the training loss with epoch."""

    # Directories
    model_dir = f"{output_dir}/model"
    plot_dir = output_dir

    history = pickle.load(open(f"{model_dir}/history.pkl", "rb"))
    plotting.plot_line(
        [range(1, len(history["loss"]) + 1)],
        [history["loss"]],
        x_label="Epoch",
        y_label="Loss",
        title="Training loss",
        filename=f"{plot_dir}/loss.png",
    )


# Set directories
output_dir = f"results/hmm/run{id}"
tmp_dir = f"tmp/hmm/run{id}"

config = """
    train_hmm:
        config_kwargs:
            n_states: 6
            learn_means: False
            learn_covariances: True
    dual_estimation:
        n_jobs: 16
"""

training_data = load_data(tmp_dir)
run_pipeline(
    config,
    output_dir=output_dir,
    data=training_data,
)
