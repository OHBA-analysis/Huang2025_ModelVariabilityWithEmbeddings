import os
from collections import defaultdict

import numpy as np
import pickle
import pandas as pd
from scipy.stats import t
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import seaborn as sns

from osl_dynamics import simulation, data
from osl_dynamics.inference import metrics, modes, tf_ops
from osl_dynamics.models import hmm, hive
from osl_dynamics.utils import set_random_seed
from osl_dynamics.glm import DesignConfig, GLM

# GPU settings
tf_ops.gpu_growth()


def build_and_train(hmm_config, hive_config, training_data):
    """Build and train both models on the same data.

    Parameters
    ----------
    hmm_config : osl_dynamics.models.hmm.Config
        Configuration for the HMM model.
    hive_config : osl_dynamics.models.hive.Config
        Configuration for the HIVE model.
    training_data : osl_dynamics.data.Data
        Training data.

    Returns
    -------
    hmm_model : osl_dynamics.models.hmm.Model
        Trained HMM model.
    hive_model : osl_dynamics.models.hive.Model
        Trained HIVE model.
    """
    # Build and train models
    hmm_model = hmm.Model(hmm_config)
    hmm_model.set_regularizers(training_data)
    hmm_model.random_state_time_course_initialization(
        training_data, n_epochs=5, n_init=5, take=1
    )
    hmm_model.fit(training_data)

    hive_model = hive.Model(hive_config)
    hive_model.set_regularizers(training_data)
    hive_model.set_dev_parameters_initializer(training_data)
    hive_model.random_state_time_course_initialization(
        training_data, n_epochs=5, n_init=5, take=1
    )
    hive_model.fit(training_data)

    return hmm_model, hive_model


def get_acc(hmm_model, hive_model, sim, training_data):
    """Get the accuracy of session covariances.

    Parameters
    ----------
    hmm_model : osl_dynamics.models.hmm.Model
        Trained HMM model.
    hive_model : osl_dynamics.models.hive.Model
        Trained HIVE model.
    sim : osl_dynamics.simulation.MArr_HMM_MVN
        Simulation object.
    training_data : osl_dynamics.data.Data
        Training data.

    Returns
    -------
    hmm_acc : np.ndarray
        Accuracy of session covariances for the HMM model.
    hive_acc : np.ndarray
        Accuracy of session covariances for the HIVE model.
    """
    # Get the orders of the modes
    sim_alp = np.concatenate(sim.mode_time_course)
    hmm_alp = hmm_model.get_alpha(training_data, concatenate=True)
    hive_alp = hive_model.get_alpha(training_data, concatenate=True)
    hmm_order = modes.match_modes(sim_alp, hmm_alp, return_order=True)[1]
    hive_order = modes.match_modes(sim_alp, hive_alp, return_order=True)[1]

    # Get and order the covariances
    sim_session_covs = sim.session_covariances
    hmm_session_covs = hmm_model.dual_estimation(training_data)[1]
    hmm_session_covs = hmm_session_covs[:, hmm_order, :, :]
    hive_session_covs = hive_model.get_session_means_covariances()[1]
    hive_session_covs = hive_session_covs[:, hive_order, :, :]

    # Accuracy on session covariances
    n_states = hive_model.config.n_states
    n_sessions = hive_model.config.n_sessions
    hmm_acc = np.empty((n_sessions, n_states))
    hive_acc = np.empty((n_sessions, n_states))
    for i in range(n_sessions):
        hmm_acc[i] = metrics.alpha_correlation(
            hmm_session_covs[i].reshape(n_states, -1).T,
            sim_session_covs[i].reshape(n_states, -1).T,
        )
        hive_acc[i] = metrics.alpha_correlation(
            hive_session_covs[i].reshape(n_states, -1).T,
            sim_session_covs[i].reshape(n_states, -1).T,
        )
    return hmm_acc, hive_acc


def build_df(hmm_dict, hive_dict):
    """
    Build a dataframe from the results.

    Parameters
    ----------
    hmm_dict : dict
        Dictionary of HMM results.
    hive_dict : dict
        Dictionary of HIVE results.
    """
    df = pd.DataFrame(
        columns=[
            "model",
            "n_sessions",
            "session_covs_acc",
        ]
    )
    for n_sessions in n_sessions_list:
        acc_dict = {
            "model": ["HMM-DE", "HIVE"],
            "n_sessions": [n_sessions] * 2,
            "session_covs_acc": [
                np.concatenate(hmm_dict[n_sessions]).flatten(),
                np.concatenate(hive_dict[n_sessions]).flatten(),
            ],
        }
        df = pd.concat([df, pd.DataFrame(acc_dict)], ignore_index=True)

    df = df.explode(["session_covs_acc"])
    df["session_covs_acc"] = df["session_covs_acc"].astype(np.float32)
    df = df.dropna()
    df["model_binarised"] = df["model"].map({"HMM-DE": 0, "HIVE": 1})
    df["interaction"] = df["n_sessions"] * df["model_binarised"]
    return df


def fit_regression(df):
    """Fit a regression model.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe of the results.
    """
    n_samples = df.shape[0]
    features = [
        {
            "name": "Intercept",
            "values": np.ones(n_samples),
            "feature_type": "continuous",
        },
        {
            "name": "Model",
            "values": df["model_binarised"].to_numpy(),
            "feature_type": "continuous",
        },
        {
            "name": "Sessions",
            "values": df["n_sessions"].to_numpy(),
            "feature_type": "continuous",
        },
        {
            "name": "Interaction",
            "values": df["interaction"].to_numpy(),
            "feature_type": "continuous",
        },
    ]
    contrasts = [
        {"name": "Intercept", "values": [1, 0, 0, 0]},
        {"name": "Model", "values": [0, 1, 0, 0]},
        {"name": "Sessions", "values": [0, 0, 1, 0]},
        {"name": "Interaction", "values": [0, 0, 0, 1]},
    ]
    DC = DesignConfig(features, contrasts, standardize_features=False)
    design = DC.create_design()
    glm = GLM(design)
    glm.fit(df["session_covs_acc"].to_numpy())

    t_stats = glm.get_tstats()
    p_values = 1 - t.cdf(np.abs(t_stats), n_samples - 4)

    return glm.betas, t_stats, p_values


def plot_results(df, plot_dir):
    """Plot the results.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe of the results.
    plot_dir : str
        Directory to save the plots.
    """
    # Define color palette (same as in boxplot)
    palette = sns.color_palette("Set2")

    # Initialize figure and axes
    fig, ax = plt.subplots(figsize=(25, 12))

    # Boxplot with Mean
    boxplot = sns.boxplot(
        x="n_sessions",
        y="session_covs_acc",
        hue="model",
        data=df,
        ax=ax,
        dodge=True,
        linewidth=2,
        gap=0.2,
        palette=palette,
    )

    # Stripplot (scatter points)
    sns.stripplot(
        x="n_sessions",
        y="session_covs_acc",
        hue="model",
        data=df,
        ax=ax,
        dodge=True,
        alpha=0.15,
        jitter=True,
        marker="o",
        linewidth=1,
        edgecolor="black",
        legend=False,
        palette=palette,
    )

    # Compute means instead of medians
    means = df.groupby(["n_sessions", "model"])["session_covs_acc"].mean().reset_index()

    # Get x-tick positions
    unique_sessions = sorted(df["n_sessions"].unique())
    positions = {
        val: i for i, val in enumerate(unique_sessions)
    }  # Mapping of session -> x-position

    # Get dodge offset for multiple models
    models = df["model"].unique()
    n_models = len(models)
    width = 0.8 / n_models  # Width for dodge effect (same as in boxplot)
    model_offsets = {
        model: i - (n_models - 1) / 2 for i, model in enumerate(models)
    }  # Left/Right shift

    # Prepare for line connections
    model_means = {}  # Store (x, y) coordinates for each model

    # Overlay mean points and store their positions
    for model_idx, model in enumerate(models):
        model_means[model] = {"x": [], "y": []}
        color = palette[model_idx]  # Get matching color from boxplot

        for _, row in means[means["model"] == model].iterrows():
            x_pos = (
                positions[row["n_sessions"]] + width * model_offsets[model]
            )  # Adjusted x position
            y_pos = row["session_covs_acc"]

            # Store for line plotting
            model_means[model]["x"].append(x_pos)
            model_means[model]["y"].append(y_pos)

            # Plot the mean points (diamonds in matching colors)
            plt.scatter(
                x_pos,
                y_pos,
                color=color,
                s=200,
                marker="D",
                edgecolor="black",
                zorder=3,
            )

    # Connect means with lines using the same colors
    for model, coords in model_means.items():
        plt.plot(
            coords["x"],
            coords["y"],
            linestyle="--",
            marker="",
            color=palette[list(models).index(model)],
            linewidth=2,
            alpha=0.8,
        )

    # Labels & Styling
    ax.set_xlabel(r"Number of Sessions ($N$)", fontsize=30, labelpad=15)
    ax.set_ylabel("Accuracy (Correlation)", fontsize=30, labelpad=15)
    ax.set_ylim(-0.1, 1.1)

    # Adjust X & Y ticks
    ax.set_xticklabels(10 * (np.array(ax.get_xticks()).astype(int) + 1), fontsize=24)
    ax.set_yticklabels(np.array(ax.get_yticks()).round(1), fontsize=24)

    # Get existing legend handles and labels
    handles, labels = ax.get_legend_handles_labels()

    # Add custom legend items
    mean_legend = Line2D(
        [0],
        [0],
        color="black",
        marker="D",
        linestyle="None",
        markersize=10,
        label="Mean",
    )
    median_legend = Line2D(
        [0], [0], color="black", linestyle="-", linewidth=2, label="Median (Boxplot)"
    )

    # Update legend
    ax.legend(
        handles[:2] + [mean_legend, median_legend],
        labels[:2] + ["Mean", "Median (Boxplot)"],
        fontsize=26,
        loc="upper center",
        frameon=True,
        fancybox=True,
        shadow=True,
        ncol=4,
        bbox_to_anchor=(0.5, 1.03),
    )

    # Adjust layout
    fig.tight_layout()
    fig.savefig(f"{plot_dir}/session_covs_acc.png")


train = False
n_sessions_list = np.arange(10, 110, 10)

# Output directories
model_dir = "model/acc_increase_with_nsessions"
figures_dir = "figures/acc_increase_with_nsessions"

os.makedirs(model_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

set_random_seed(0)
if train:
    hmm_dict = defaultdict(list)
    hive_dict = defaultdict(list)
    for n_sessions in n_sessions_list:
        # Model configurations
        hmm_config = hmm.Config(
            n_states=5,
            n_channels=40,
            sequence_length=200,
            learn_means=False,
            learn_covariances=True,
            batch_size=64,
            learning_rate=0.005,
            lr_decay=0.05,
            n_epochs=40,
            learn_trans_prob=True,
            loss_calc="sum",
        )
        hive_config = hive.Config(
            n_states=5,
            n_channels=40,
            n_sessions=n_sessions,
            embeddings_dim=10,
            spatial_embeddings_dim=2,
            sequence_length=200,
            learn_means=False,
            learn_covariances=True,
            dev_n_layers=5,
            dev_n_units=32,
            dev_activation="tanh",
            dev_normalization="layer",
            dev_regularizer="l1",
            dev_regularizer_factor=10,
            batch_size=64,
            learning_rate=0.005,
            lr_decay=0.05,
            n_epochs=40,
            do_kl_annealing=True,
            kl_annealing_curve="tanh",
            kl_annealing_sharpness=10,
            n_kl_annealing_epochs=20,
            loss_calc="sum",
        )

        # Train both models on the same data for 10 times
        for i in range(10):
            # Simulate data
            sim = simulation.MSess_HMM_MVN(
                n_samples=1000,
                trans_prob="sequence",
                session_means="zero",
                session_covariances="random",
                n_states=5,
                n_channels=40,
                n_covariances_act=1,
                n_sessions=n_sessions,
                embeddings_dim=2,
                spatial_embeddings_dim=2,
                embeddings_scale=0.005,
                n_groups=3,
                between_group_scale=0.5,
                stay_prob=0.9,
                observation_error=0.5,
            )
            sim.standardize()
            training_data = data.Data([mtc for mtc in sim.time_series])
            training_data.add_session_labels(
                "session_id", np.arange(n_sessions), "categorical"
            )
            print(f"Training for {n_sessions} sessions, run {i + 1}/10")
            hmm_model, hive_model = build_and_train(
                hmm_config, hive_config, training_data
            )
            hmm_acc, hive_acc = get_acc(hmm_model, hive_model, sim, training_data)
            hmm_dict[n_sessions].append(hmm_acc)
            hive_dict[n_sessions].append(hive_acc)

    # Save the results
    pickle.dump((hmm_dict, hive_dict), open(f"{model_dir}/results.pkl", "wb"))
else:
    hmm_dict, hive_dict = pickle.load(open(f"{model_dir}/results.pkl", "rb"))

# Build dataframe
df = build_df(hmm_dict, hive_dict)

# Fit regression model
betas, t_stats, p_values = fit_regression(df)
print("features: Intercept, Model, Sessions, Interaction")
print("effect sizes:", betas)
print("t_stats:", t_stats)
print("p_values:", p_values)

# Plot results
plot_results(df, figures_dir)
