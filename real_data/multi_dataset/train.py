from sys import argv

if len(argv) != 2:
    print(f"Please pass the run id, e.g. python {argv[0]} 1")
    exit()
id = argv[1]

import os
from glob import glob
from pathlib import Path

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

from osl_dynamics import run_pipeline
from osl_dynamics.data import Data
from osl_dynamics.inference import tf_ops, metrics
from osl_dynamics.analysis import power
from osl_dynamics.utils import plotting
from osl_dynamics.utils.misc import load, override_dict_defaults

tf_ops.gpu_growth()


# Helper functions
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
    camcan_groups = camcan_demo.groupby(["age_range", "sex"]).size()

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


def plot_subject_embeddings(data, output_dir, view_init_kwargs=None):
    """Plot the subject embeddings."""

    if view_init_kwargs is None:
        view_init_kwargs = {"elev": 10, "azim": 0}

    # Directories
    inf_params_dir = f"{output_dir}/inf_params"
    plot_dir = f"{output_dir}/subject_embeddings"
    os.makedirs(plot_dir, exist_ok=True)

    subject_embeddings = np.load(f"{inf_params_dir}/subject_embeddings.npy")
    subject_embeddings -= subject_embeddings.mean(axis=0)
    subject_embeddings /= subject_embeddings.std(axis=0)
    covs = np.load(f"{inf_params_dir}/covs.npy")

    # Pairwise L2 distances of covariances and subject embeddings
    covs_pw_l2 = metrics.pairwise_l2_distance(
        np.transpose(covs, (1, 0, 2, 3)), batch_dims=1
    )
    se_pw_l2 = metrics.pairwise_l2_distance(subject_embeddings)

    # Plot the pairwise L2 distances of covariances vs subject embeddings
    m, n = np.tril_indices(len(covs), k=-1)
    plotting.plot_scatter(
        [se_pw_l2[m, n]] * covs.shape[1],
        covs_pw_l2[:, m, n],
        labels=[f"state {i + 1}" for i in range(covs.shape[1])],
        x_label="Subject embeddings",
        y_label="Covariances",
        filename=f"{plot_dir}/pw_l2_se_vs_covs.png",
    )

    # Plot the pairwise L2 distances of covariances
    plotting.plot_matrices(
        covs_pw_l2,
        filename=f"{plot_dir}/covs_l2.png",
    )

    # Pairwise correlation between covariances and subject embeddings
    covs_pw_corr = np.zeros((covs.shape[1], covs.shape[0], covs.shape[0]))
    for i in range(covs.shape[1]):
        covs_pw_corr[i] = metrics.pairwise_matrix_correlations(covs[:, i])

    se_corr = np.corrcoef(subject_embeddings)

    plotting.plot_matrices(
        covs_pw_corr,
        filename=f"{plot_dir}/covs_corr.png",
    )

    df = match_subjects()

    # Normalise and PCA
    pca = PCA(n_components=2)
    pca_subject_embeddings = pca.fit_transform(subject_embeddings)

    # Get the centroids of the clusters
    centroids = np.array(
        [
            np.mean(subject_embeddings[df["dataset"] == "notts"], axis=0),
            np.mean(subject_embeddings[df["dataset"] == "camcan"], axis=0),
        ]
    )
    pca_centroids = pca.transform(centroids)

    # 3D plot
    sns.set_palette("colorblind", n_colors=2)
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(**view_init_kwargs)
    ax.scatter(
        subject_embeddings[df["dataset"] == "notts", 0],
        subject_embeddings[df["dataset"] == "notts", 1],
        subject_embeddings[df["dataset"] == "notts", 2],
        label="Nottingham",
    )
    ax.scatter(
        subject_embeddings[df["dataset"] == "camcan", 0],
        subject_embeddings[df["dataset"] == "camcan", 1],
        subject_embeddings[df["dataset"] == "camcan", 2],
        label="Cam-CAN",
    )
    ax.plot(
        centroids[0][0],
        centroids[0][1],
        centroids[0][2],
        color="black",
        marker="*",
        markersize=15,
    )
    ax.plot(
        centroids[1][0],
        centroids[1][1],
        centroids[1][2],
        color="black",
        marker="*",
        markersize=15,
    )
    ax.legend()
    fig.savefig(f"{plot_dir}/se_3d.png")
    plt.close()
    sns.reset_defaults()

    sns.set_palette("viridis", n_colors=8)
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(**view_init_kwargs)
    for age_range in [
        "18-24",
        "24-30",
        "30-36",
        "36-42",
        "42-48",
        "48-54",
        "54-60",
        "60+",
    ]:
        ax.scatter(
            subject_embeddings[df.age_range == age_range, 0],
            subject_embeddings[df.age_range == age_range, 1],
            subject_embeddings[df.age_range == age_range, 2],
            label=age_range,
        )

    ax.legend()
    fig.savefig(f"{plot_dir}/se_3d_age.png")
    plt.close()
    sns.reset_defaults()

    # Plot the pairwise corr and L2 distances of subject embeddings
    fig, ax = plotting.plot_matrices(
        se_corr,
    )
    ax[0][0].set_xticks(
        ticks=[64 // 2, 64, 64 + 64 // 2], labels=["notts", "", "CamCAN"]
    )
    ax[0][0].set_yticks(
        ticks=[64 // 2, 64, 64 + 64 // 2], labels=["notts", "", "CamCAN"]
    )
    plt.setp(ax[0][0].get_yticklabels(), rotation=90)
    fig.savefig(f"{plot_dir}/se_corr.png")
    plt.close()

    fig, ax = plotting.plot_matrices(
        se_pw_l2,
    )
    ax[0][0].set_xticks(
        ticks=[64 // 2, 64, 64 + 64 // 2],
        labels=["Nottingham", "", "Cam-CAN"],
        fontsize=15,
    )
    ax[0][0].set_yticks(
        ticks=[64 // 2, 64, 64 + 64 // 2],
        labels=["Nottingham", "", "Cam-CAN"],
        fontsize=15,
    )
    plt.setp(ax[0][0].get_yticklabels(), rotation=90, va="center")
    fig.savefig(f"{plot_dir}/se_l2.png")
    plt.close()

    # 2D plot
    sns.set_palette("colorblind", n_colors=2)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(
        pca_subject_embeddings[df["dataset"] == "notts", 0],
        pca_subject_embeddings[df["dataset"] == "notts", 1],
        label="Nottingham",
    )
    ax.scatter(
        pca_subject_embeddings[df["dataset"] == "camcan", 0],
        pca_subject_embeddings[df["dataset"] == "camcan", 1],
        label="Cam-CAN",
    )
    ax.scatter(
        pca_centroids[:, 0],
        pca_centroids[:, 1],
        color="black",
        marker="*",
        s=100,
    )
    ax.legend()
    fig.savefig(f"{plot_dir}/se_2d.png")
    plt.close()

    sns.set_palette("viridis", n_colors=8)
    fig, ax = plt.subplots(figsize=(5, 5))
    for age_range in [
        "18-24",
        "24-30",
        "30-36",
        "36-42",
        "42-48",
        "48-54",
        "54-60",
        "60+",
    ]:
        ax.scatter(
            pca_subject_embeddings[df.age_range == age_range, 0],
            pca_subject_embeddings[df.age_range == age_range, 1],
            label=age_range,
        )
    ax.legend()
    fig.savefig(f"{plot_dir}/se_2d_age.png")
    plt.close()


def plot_dataset_networks_diff(
    data,
    output_dir,
    nnmf_file,
    mask_file=None,
    parcellation_file=None,
    component=0,
    power_save_kwargs=None,
):
    """Plot the difference in networks between datasets."""
    power_save_kwargs = {} if power_save_kwargs is None else power_save_kwargs

    def _get_nearest_neighbours(candidates, target, n_neighbours):
        """Get the nearest neighbours."""
        distances = np.linalg.norm(target[None, ...] - candidates, axis=1)
        return np.argsort(distances)[:n_neighbours]

    # Directories
    inf_params_dir = f"{output_dir}/inf_params"
    spectra_dir = output_dir + "/spectra"
    plot_dir = f"{output_dir}/networks_diff"
    os.makedirs(plot_dir, exist_ok=True)

    # Load the NNMF components
    nnmf_file = output_dir + "/" + nnmf_file
    if Path(nnmf_file).exists():
        nnmf = load(nnmf_file)
    else:
        raise ValueError(f"{nnmf_file} not found.")

    # Load spectra
    f = load(f"{spectra_dir}/f.npy")
    psd = load(f"{spectra_dir}/psd.npy")

    # Load the subject embeddings
    subject_embeddings = np.load(f"{inf_params_dir}/subject_embeddings.npy")
    df = match_subjects()

    # Normalise and PCA
    subject_embeddings -= subject_embeddings.mean(axis=0)
    subject_embeddings /= subject_embeddings.std(axis=0)

    # Get the centroids of the clusters
    centroids = np.array(
        [
            np.mean(subject_embeddings[df["dataset"] == "notts"], axis=0),
            np.mean(subject_embeddings[df["dataset"] == "camcan"], axis=0),
        ]
    )

    # Get the nearest neighbours of the centroids
    notts_neighbours = _get_nearest_neighbours(
        candidates=subject_embeddings,
        target=centroids[0],
        n_neighbours=10,
    )
    camcan_neighbours = _get_nearest_neighbours(
        candidates=subject_embeddings,
        target=centroids[1],
        n_neighbours=10,
    )
    cluster_gpsd = [
        np.average(psd[notts_neighbours], axis=0),
        np.average(psd[camcan_neighbours], axis=0),
    ]

    notts_power_map = power.variance_from_spectra(f, cluster_gpsd[0], nnmf)
    camcan_power_map = power.variance_from_spectra(f, cluster_gpsd[1], nnmf)
    notts_minus_camcan_power_map = notts_power_map - camcan_power_map

    # Save power maps
    default_power_save_kwargs = {
        "mask_file": mask_file,
        "parcellation_file": parcellation_file,
        "component": component,
        "filename": f"{plot_dir}/pow_.png",
        "subtract_mean": False,
    }
    power_save_kwargs = override_dict_defaults(
        default_power_save_kwargs, power_save_kwargs
    )
    power.save(
        notts_minus_camcan_power_map,
        **power_save_kwargs,
    )

    # Difference in PSDs
    P_notts = cluster_gpsd[0]
    P_camcan = cluster_gpsd[1]

    P_notts_minus_camcan = P_notts - P_camcan
    P_max = np.max(P_notts_minus_camcan)
    P_min = np.min(P_notts_minus_camcan)
    sns.set_palette("viridis", P_notts_minus_camcan.shape[1])

    for j in range(P_notts_minus_camcan.shape[0]):
        fig, ax = plotting.plot_line(
            [f] * P_notts_minus_camcan.shape[1],
            P_notts_minus_camcan[j],
            x_range=[0, 45],
            y_range=[P_min, P_max],
            y_label="PSD (a.u.)",
        )
        ax.plot(f, np.mean(P_notts_minus_camcan[j], axis=0), color="red")
        ax.axhline(0, color="black", linestyle="--")
        ax.set_xticklabels(ax.get_xticks(), fontsize=15)
        fig.savefig(f"{plot_dir}/psd_{j}.png", dpi=300)
        plt.close(fig)

    sns.reset_defaults()


# Set directories
output_dir = f"results/run{id}"
tmp_dir = f"tmp/run{id}"

config = """
    train_sehmm:
        config_kwargs:
            n_states: 6
            learn_means: False
            learn_covariances: True
            subject_embeddings_dim: 3
            n_epochs: 40
            do_kl_annealing: True
            kl_annealing_curve: tanh
            kl_annealing_sharpness: 10
            n_kl_annealing_epochs: 20
    multitaper_spectra:
        kwargs:
            frequency_range: [1, 45]
            n_jobs: 16
        nnmf_components: 2
    plot_loss: {}
    plot_subject_embeddings:
        view_init_kwargs: {elev: 10, azim: 70}
    plot_alpha:
        kwargs: {n_samples: 2000}
    plot_group_nnmf_tde_hmm_networks:
        nnmf_file: spectra/nnmf_2.npy
        mask_file: MNI152_T1_8mm_brain.nii.gz
        parcellation_file: fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz
        power_save_kwargs:
            plot_kwargs: {views: [lateral]}
    plot_hmm_network_summary_stats: {}
    plot_dataset_networks_diff:
        nnmf_file: spectra/nnmf_2.npy
        mask_file: MNI152_T1_8mm_brain.nii.gz
        parcellation_file: fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz
        power_save_kwargs:
            plot_kwargs: {views: [lateral]}
"""
training_data = load_data(tmp_dir)
run_pipeline(
    config,
    output_dir=output_dir,
    data=training_data,
    extra_funcs=[plot_loss, plot_subject_embeddings, plot_dataset_networks_diff],
)
