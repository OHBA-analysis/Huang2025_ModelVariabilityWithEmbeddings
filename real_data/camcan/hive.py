from sys import argv

if len(argv) != 3:
    print(f"Need to pass the run id, e.g. python {argv[0]} 1 10")
    exit()
id = argv[1]
embeddings_dim = argv[2]

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


def load_data(data_dir, store_dir, use_tfrecord=True, buffer_size=2000, n_jobs=16):
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


def plot_results(data, output_dir):
    """Plot the subject embeddings."""

    def _get_nearest_neighbours(candidates, target, n_neighbours):
        """Get the nearest neighbours."""
        distances = np.linalg.norm(target[None, ...] - candidates, axis=1)
        return np.argsort(distances)[:n_neighbours]

    # Directories
    inf_params_dir = f"{output_dir}/inf_params"
    se_dir = f"{output_dir}/embeddings"
    spectra_dir = f"{output_dir}/spectra"
    net_diff_dir = f"{output_dir}/networks_diff"

    os.makedirs(se_dir, exist_ok=True)
    os.makedirs(net_diff_dir, exist_ok=True)

    embeddings = np.load(f"{inf_params_dir}/embeddings.npy")
    embeddings -= embeddings.mean(axis=0)
    embeddings /= embeddings.std(axis=0)
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
    young_centroid = np.mean(embeddings[df.age_range.isin(categories[:3])], axis=0)
    old_centroid = np.mean(embeddings[df.age_range.isin(categories[-3:])], axis=0)

    # PCA
    pca = PCA(n_components=2)
    pca_subject_embeddings = pca.fit_transform(embeddings)
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
        embeddings, young_centroid, n_neighbours=20
    )
    old_neighbours = _get_nearest_neighbours(embeddings, old_centroid, n_neighbours=20)
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

    P_mean = np.mean(P_old_minus_young, axis=1)
    P_std = np.std(P_old_minus_young, axis=1)
    for j in range(P_old_minus_young.shape[0]):
        fig, ax = plotting.plot_line(
            [f],
            [P_mean[j]],
            errors=[[P_mean[j] + P_std[j]], [P_mean[j] - P_std[j]]],
            x_range=[0, 45],
            y_range=[-0.05, 0.05],
        )
        ax.axhline(0, color="black", linestyle="--")
        ax.set_xticklabels(ax.get_xticks(), fontsize=15)
        ax.set_yticklabels(ax.get_yticks().round(2), fontsize=15)
        fig.savefig(f"{net_diff_dir}/psd_{j}.png", dpi=300)
        plt.close(fig)


config = f"""
    train_hive:
        config_kwargs:
            n_states: 8
            learn_means: False
            learn_covariances: True
            embeddings_dim: {embeddings_dim}
            learning_rate: 0.0025
            n_epochs: 30
            do_kl_annealing: True
            kl_annealing_curve: tanh
            kl_annealing_sharpness: 10
            n_kl_annealing_epochs: 15
        init_kwargs:
            n_epochs: 1
            n_init: 10
    plot_results: {{}}
"""

# Set directories
output_dir = f"results/hive_{embeddings_dim}/run{id}"
tmp_dir = f"tmp/hive_{embeddings_dim}/run{id}"
data_dir = "/well/woolrich/projects/camcan/winter23/src"

training_data = load_data(data_dir, tmp_dir, n_jobs=16)
run_pipeline(
    config,
    output_dir=output_dir,
    data=training_data,
)
