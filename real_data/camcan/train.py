from sys import argv

if len(argv) != 2:
    print(f"Need to pass the run id, e.g. python {argv[0]} 1")
    exit()
id = argv[1]

import os
from glob import glob
from pathlib import Path

import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from osl_dynamics import run_pipeline
from osl_dynamics.data import Data
from osl_dynamics.inference import tf_ops
from osl_dynamics.analysis import power
from osl_dynamics.utils import plotting
from osl_dynamics.utils.misc import load, override_dict_defaults

tf_ops.gpu_growth()


def load_data(data_dir, store_dir, use_tfrecord=True, buffer_size=2000, n_jobs=1):
    """Load the data."""

    data_paths = sorted(glob(f"{data_dir}/sub*/sflip_parc.npy"))
    training_data = Data(
        data_paths,
        sampling_frequency=250,
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


def get_demographics():
    """Get the demographics."""
    df = pd.DataFrame(columns=["data_path", "participant_id"])
    df["data_path"] = sorted(
        glob("/well/woolrich/projects/camcan/winter23/src/sub*/sflip_parc.npy")
    )
    df["participant_id"] = df["data_path"].apply(
        lambda x: os.path.basename(os.path.dirname(x))
    )
    demo_df = pd.read_csv("/well/woolrich/projects/camcan/participants.tsv", sep="\t")
    df = df.merge(demo_df, on="participant_id")

    categories = [
        "18-24",
        "24-30",
        "30-36",
        "36-42",
        "42-48",
        "48-54",
        "54-60",
        "60-66",
        "66-72",
        "72-78",
        "78-84",
        "84+",
    ]
    # Bin the age
    df["age_range"] = pd.cut(
        df["age"],
        bins=[18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 100],
        labels=categories,
        right=False,
    )
    df["age_range"] = pd.Categorical(
        df["age_range"],
        categories=categories,
        ordered=True,
    )
    return df


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


def plot_results(data, output_dir, view_init_kwargs=None):
    """Plot the subject embeddings."""

    def _get_nearest_neighbours(candidates, target, n_neighbours):
        """Get the nearest neighbours."""
        distances = np.linalg.norm(target[None, ...] - candidates, axis=1)
        return np.argsort(distances)[:n_neighbours]

    if view_init_kwargs is None:
        view_init_kwargs = {"elev": 10, "azim": 0}

    # Directories
    inf_params_dir = f"{output_dir}/inf_params"
    se_dir = f"{output_dir}/subject_embeddings"
    spectra_dir = f"{output_dir}/spectra"
    net_diff_dir = f"{output_dir}/networks_diff"

    os.makedirs(se_dir, exist_ok=True)
    os.makedirs(net_diff_dir, exist_ok=True)

    subject_embeddings = np.load(f"{inf_params_dir}/subject_embeddings.npy")
    subject_embeddings -= subject_embeddings.mean(axis=0)
    subject_embeddings /= subject_embeddings.std(axis=0)
    df = get_demographics()

    # Age categories
    categories = [
        "18-24",
        "24-30",
        "30-36",
        "36-42",
        "42-48",
        "48-54",
        "54-60",
        "60-66",
        "66-72",
        "72-78",
        "78-84",
        "84+",
    ]
    # Young and old centroids
    young_centroid = np.mean(
        subject_embeddings[df.age_range.isin(categories[:3])], axis=0
    )
    old_centroid = np.mean(
        subject_embeddings[df.age_range.isin(categories[-3:])], axis=0
    )

    # PCA
    pca = PCA(n_components=2)
    pca_subject_embeddings = pca.fit_transform(subject_embeddings)
    pca_young_centroid = np.squeeze(pca.transform(young_centroid.reshape(1, -1)))
    pca_old_centroid = np.squeeze(pca.transform(old_centroid.reshape(1, -1)))

    # 2D plot
    df["pca_se_0"] = list(pca_subject_embeddings[:, 0])
    df["pca_se_1"] = list(pca_subject_embeddings[:, 1])
    fig, ax = plt.subplots()
    sns.scatterplot(
        data=df,
        x="pca_se_0",
        y="pca_se_1",
        hue="age_range",
        ax=ax,
        palette="viridis",
    )
    ax.plot(
        pca_young_centroid[0],
        pca_young_centroid[1],
        marker="*",
        color="red",
        markersize=20,
    )
    ax.plot(
        pca_old_centroid[0],
        pca_old_centroid[1],
        marker="*",
        color="red",
        markersize=20,
    )
    fig.savefig(f"{se_dir}/2d_se.png")
    plt.close()

    # 3D plot
    sns.set_palette("viridis", n_colors=len(categories))
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(**view_init_kwargs)
    for cat in categories:
        idx = df.age_range == cat
        ax.scatter(
            subject_embeddings[idx, 0],
            subject_embeddings[idx, 1],
            subject_embeddings[idx, 2],
            label=cat,
        )
    ax.legend()
    plt.tight_layout()
    fig.savefig(f"{se_dir}/3d_se.png")
    plt.close()

    sns.set_palette("colorblind")
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(**view_init_kwargs)
    for sex in ["MALE", "FEMALE"]:
        idx = df.sex == sex
        ax.scatter(
            subject_embeddings[idx, 0],
            subject_embeddings[idx, 1],
            subject_embeddings[idx, 2],
            label=sex,
        )
    ax.legend()
    plt.tight_layout()
    fig.savefig(f"{se_dir}/3d_se_sex.png")
    plt.close()

    # Load the NNMF components
    nnmf_file = "spectra/nnmf_2.npy"
    nnmf_file = output_dir + "/" + nnmf_file
    if Path(nnmf_file).exists():
        nnmf = load(nnmf_file)
    else:
        raise ValueError(f"{nnmf_file} not found.")

    # Load spectra
    f = load(f"{spectra_dir}/f.npy")
    psd = load(f"{spectra_dir}/psd.npy")

    young_neighbours = _get_nearest_neighbours(
        subject_embeddings, young_centroid, n_neighbours=20
    )
    old_neighbours = _get_nearest_neighbours(
        subject_embeddings, old_centroid, n_neighbours=20
    )
    cluster_gpsd = [
        np.average(psd[young_neighbours], axis=0),
        np.average(psd[old_neighbours], axis=0),
    ]
    young_power_map = power.variance_from_spectra(f, cluster_gpsd[0], nnmf)
    old_power_map = power.variance_from_spectra(f, cluster_gpsd[1], nnmf)
    old_minus_young_power_map = old_power_map - young_power_map

    # Save power maps
    default_power_save_kwargs = {
        "mask_file": "MNI152_T1_8mm_brain.nii.gz",
        "parcellation_file": "fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz",
        "component": 0,
        "filename": f"{net_diff_dir}/pow_.png",
        "subtract_mean": False,
    }
    power_save_kwargs = override_dict_defaults(
        default_power_save_kwargs, {"plot_kwargs": {"views": ["lateral"]}}
    )
    power.save(
        old_minus_young_power_map,
        **power_save_kwargs,
    )

    P_young = cluster_gpsd[0]
    P_old = cluster_gpsd[1]
    P_old_minus_young = P_old - P_young
    P_max = np.max(P_old_minus_young)
    P_min = np.min(P_old_minus_young)
    sns.set_palette("viridis", n_colors=P_old_minus_young.shape[1])
    for j in range(P_old_minus_young.shape[0]):
        fig, ax = plotting.plot_line(
            [f] * P_old_minus_young.shape[1],
            P_old_minus_young[j],
            x_range=[0, 45],
            y_range=[P_min, P_max],
            y_label="PSD (a.u.)",
        )
        ax.plot(f, np.mean(P_old_minus_young[j], axis=0), color="red")
        ax.axhline(0, color="black", linestyle="--")
        ax.set_xticklabels(ax.get_xticks(), fontsize=15)
        fig.savefig(f"{net_diff_dir}/psd_{j}.png", dpi=300)
        plt.close(fig)


config = """
    train_sehmm:
        config_kwargs:
            n_states: 8
            learn_means: False
            learn_covariances: True
            subject_embeddings_dim: 3
            batch_size: 64
            n_epochs: 10
            learning_rate: 0.001
            do_kl_annealing: True
            kl_annealing_curve: tanh
            kl_annealing_sharpness: 10
            n_kl_annealing_epochs: 5
        n_jobs: 6
        init_kwargs:
            n_epochs: 1
            take: 0.4
            n_init: 10
    multitaper_spectra:
        kwargs:
            frequency_range: [1, 45]
            n_jobs: 6
        nnmf_components: 2
    plot_loss: {}
    plot_results:
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
"""

# Set directories
output_dir = f"results/run{id}"
tmp_dir = f"tmp/run{id}"
data_dir = "/well/woolrich/projects/camcan/winter23/src"

training_data = load_data(data_dir, tmp_dir, n_jobs=6)
run_pipeline(
    config,
    output_dir=output_dir,
    data=training_data,
    extra_funcs=[plot_loss, plot_results],
)
