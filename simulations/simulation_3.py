import os
from collections import defaultdict

import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from osl_dynamics import simulation, data
from osl_dynamics.inference import metrics, modes, tf_ops
from osl_dynamics.models import hmm, hive

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


def plot_results(hmm_dict, hive_dict, plot_dir):
    """Plot the results.

    Parameters
    ----------
    hmm_dict : dict
        Dictionary of HMM results.
    hive_dict : dict
        Dictionary of HIVE results.
    plot_dir : str
        Directory to save the plots.
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

    ax = sns.boxplot(
        data=df,
        x="n_sessions",
        y="session_covs_acc",
        hue="model",
        palette="colorblind",
        linewidth=1,
        showfliers=False,
        medianprops={"color": "r", "linewidth": 2},
        notch=True,
    )
    ax.set_xlabel("Number of sessions", fontsize=20)
    ax.set_ylabel("Accuracy", fontsize=20)
    ax.tick_params(labelsize=15)
    ax.legend()
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    # save figure
    plt.savefig(f"{plot_dir}/acc_increase_with_nsessions.png", dpi=300)


train = True
n_sessions_list = [5, 10, 50, 100]

# Output directories
model_dir = "model/acc_increase_with_nsessions"
figures_dir = "figures/acc_increase_with_nsessions"

os.makedirs(model_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

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
        )
        hive_config = hive.Config(
            n_states=5,
            n_channels=40,
            n_sessions=n_sessions,
            embeddings_dim=2,
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
        )

        # Simulate data
        sim = simulation.MArr_HMM_MVN(
            n_samples=3000,
            trans_prob="sequence",
            session_means="zero",
            session_covariances="random",
            n_states=5,
            n_channels=40,
            n_covariances_act=5,
            n_sessions=n_sessions,
            embeddings_dim=2,
            spatial_embeddings_dim=2,
            embeddings_scale=0.005,
            n_groups=3,
            between_group_scale=0.5,
            stay_prob=0.9,
            random_seed=1234,
        )
        sim.standardize()
        training_data = data.Data([mtc for mtc in sim.time_series])

        # Train both models on the same data for 10 times
        for i in range(10):
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

# Plot the results
plot_results(hmm_dict, hive_dict, figures_dir)

# Clean up
training_data.delete_dir()
