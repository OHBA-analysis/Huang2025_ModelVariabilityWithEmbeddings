import os
from collections import defaultdict

import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

from osl_dynamics import simulation, data
from osl_dynamics.inference import metrics, modes, tf_ops
from osl_dynamics.models import hmm, sehmm

# GPU settings
tf_ops.gpu_growth()


def build_and_train(hmm_config, sehmm_config, training_data):
    # Build and train models
    hmm_model = hmm.Model(hmm_config)
    hmm_model.set_regularizers(training_data)
    hmm_model.random_state_time_course_initialization(
        training_data, n_epochs=5, n_init=5, take=1
    )
    hmm_model.fit(training_data)

    init_model = hmm.Model(hmm_config)
    init_model.set_regularizers(training_data)
    init_model.random_state_time_course_initialization(
        training_data, n_epochs=5, n_init=5, take=1
    )
    _, init_dual_covs = init_model.dual_estimation(training_data)
    init_dual_covs = np.reshape(init_dual_covs, (sehmm_config.n_subjects, -1))
    pca = PCA(n_components=sehmm_config.subject_embeddings_dim)

    sehmm_config.initial_covariances = init_model.get_covariances()
    sehmm_config.initial_trans_prob = init_model.get_trans_prob()
    sehmm_model = sehmm.Model(sehmm_config)
    sehmm_model.set_regularizers(training_data)
    sehmm_model.set_dev_parameters_initializer(training_data)
    sehmm_model.set_subject_embeddings_initializer(pca.fit_transform(init_dual_covs))
    with sehmm_model.set_trainable("subject_embeddings", False):
        sehmm_model.random_state_time_course_initialization(
            training_data, n_epochs=5, n_init=5, take=1
        )
    sehmm_model.fit(training_data)

    return hmm_model, sehmm_model


def get_acc(hmm_model, sehmm_model, sim, training_data):
    # Get the orders of the modes
    sim_alp = np.concatenate(sim.mode_time_course)
    hmm_alp = hmm_model.get_alpha(training_data, concatenate=True)
    sehmm_alp = sehmm_model.get_alpha(training_data, concatenate=True)
    hmm_order = modes.match_modes(sim_alp, hmm_alp, return_order=True)[1]
    sehmm_order = modes.match_modes(sim_alp, sehmm_alp, return_order=True)[1]

    # Get and order the covariances
    sim_subject_covs = sim.subject_covariances
    hmm_subject_covs = hmm_model.dual_estimation(training_data)[1]
    hmm_subject_covs = hmm_subject_covs[:, hmm_order, :, :]
    sehmm_subject_covs = sehmm_model.get_subject_means_covariances()[1]
    sehmm_subject_covs = sehmm_subject_covs[:, sehmm_order, :, :]

    # Accuracy on subject covariances
    n_states = sehmm_model.config.n_states
    hmm_acc = np.empty((n_subjects, n_states))
    sehmm_acc = np.empty((n_subjects, n_states))
    for subj in range(n_subjects):
        hmm_acc[subj] = metrics.alpha_correlation(
            hmm_subject_covs[subj].reshape(n_states, -1).T,
            sim_subject_covs[subj].reshape(n_states, -1).T,
        )
        sehmm_acc[subj] = metrics.alpha_correlation(
            sehmm_subject_covs[subj].reshape(n_states, -1).T,
            sim_subject_covs[subj].reshape(n_states, -1).T,
        )
    return hmm_acc, sehmm_acc


def plot_results(hmm_dict, sehmm_dict, plot_dir):
    df = pd.DataFrame(
        columns=[
            "model",
            "n_subjects",
            "subject_covs_acc",
        ]
    )
    for n_subjects in n_subjects_list:
        acc_dict = {
            "model": ["hmm", "sehmm"],
            "n_subjects": [n_subjects] * 2,
            "subject_covs_acc": [
                np.concatenate(hmm_dict[n_subjects]),
                np.concatenate(sehmm_dict[n_subjects]),
            ],
        }
        df = pd.concat([df, pd.DataFrame(acc_dict)], ignore_index=True)

    df = df.explode(["subject_covs_acc"])
    df["subject_covs_acc"] = df["subject_covs_acc"].astype(np.float32)

    fig, axes = plt.subplots(1, 1, figsize=(10, 5))
    sns.violinplot(
        data=df,
        x="n_subjects",
        y="subject_covs_acc",
        hue="model",
        ax=axes,
        split=True,
        scale="count",
        cut=0,
    )
    axes.get_legend().remove()
    axes.set_title("Accuracy of subject covariances", fontsize=25)
    axes.set_xlabel("Number of subjects", fontsize=20)
    axes.set_ylabel("Accuracy", fontsize=20)
    axes.set_ylim(0, 1)
    plt.yticks(fontsize=13)
    plt.xticks(fontsize=13)
    fig.legend(fontsize=15, loc="lower left")
    fig.savefig(f"{plot_dir}/subject_covs_acc.png", dpi=300)
    plt.close(fig)


train = True
n_subjects_list = [5, 10, 50, 100]

# Output directories
model_dir = "model/acc_increase_with_nsubjects"
figures_dir = "figures/acc_increase_with_nsubjects"

os.makedirs(model_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

if train:
    hmm_dict = defaultdict(list)
    sehmm_dict = defaultdict(list)
    for n_subjects in n_subjects_list:
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
        sehmm_config = sehmm.Config(
            n_states=5,
            n_channels=40,
            n_subjects=n_subjects,
            subject_embeddings_dim=2,
            mode_embeddings_dim=2,
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
        sim = simulation.MSubj_HMM_MVN(
            n_samples=3000,
            trans_prob="sequence",
            subject_means="zero",
            subject_covariances="random",
            n_states=5,
            n_channels=40,
            n_covariances_act=5,
            n_subjects=n_subjects,
            n_subject_embedding_dim=2,
            n_mode_embedding_dim=2,
            subject_embedding_scale=0.005,
            n_groups=3,
            between_group_scale=0.5,
            stay_prob=0.9,
            random_seed=1234,
        )
        sim.standardize()
        training_data = data.Data([mtc for mtc in sim.time_series])

        # Train both models on the same data for 10 times
        for i in range(10):
            print(f"Training for {n_subjects} subjects, run {i + 1}/10")
            hmm_model, sehmm_model = build_and_train(
                hmm_config, sehmm_config, training_data
            )
            hmm_acc, sehmm_acc = get_acc(hmm_model, sehmm_model, sim, training_data)
            hmm_dict[n_subjects].append(hmm_acc)
            sehmm_dict[n_subjects].append(sehmm_acc)

    # Save the results
    pickle.dump((hmm_dict, sehmm_dict), open(f"{model_dir}/results.pkl", "wb"))
else:
    hmm_dict, sehmm_dict = pickle.load(open(f"{model_dir}/results.pkl", "rb"))

# Plot the results
plot_results(hmm_dict, sehmm_dict, figures_dir)

# Clean up
training_data.delete_dir()
