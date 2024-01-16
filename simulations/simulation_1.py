import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pickle

from osl_dynamics import data, simulation, array_ops
from osl_dynamics.inference import metrics, modes, tf_ops
from osl_dynamics.models import hmm, hive, obs_mod
from osl_dynamics.utils import plotting

# GPU settings
tf_ops.gpu_growth()


# Helper functions
def simulate_data(
    n_states,
    n_channels,
    n_samples,
    n_sessions,
    dev_state=0,
    dev_size=5,
):
    """This function simulates data.

    Parameters
    ----------
    n_states : int
        Number of states.
    n_channels : int
        Number of channels.
    n_samples : int
        Number of samples per session.
    n_sessions : int
        Number of sessions. Must be at least 3.
    dev_state : int, optional
        State with deviation, by default 0.
    dev_size : int, optional
        Deviation size, by default 5.

    Returns
    -------
    time_series : list of array
        List of data for each session.
        Each element has shape (n_samples, n_channels).
    sim_session_covs : np.ndarray
        Simulated session specific covariances.
        Shape is (n_sessions, n_states, n_channels, n_channels).
    stc : list of array
        List of simulated state time courses for each session.
        Each element has shape (n_samples, n_states).
    sim_group_covs : np.ndarray
        Simulated group level covariances.
        Shape is (n_states, n_channels, n_channels).
    """
    if n_sessions < 3:
        raise ValueError("n_sessions must be at least 3")

    # channels that have deviation
    dev_channels = [0, 1]

    # sessions with up and down deviations
    up_session = 0
    down_session = 1

    # Simulate group level covariances
    obs_model = simulation.MVN(
        means="zero",
        covariances="random",
        n_modes=n_states,
        n_channels=n_channels,
        random_seed=123,
    )
    group_covs = obs_model.covariances
    group_corrs = array_ops.cov2corr(group_covs)
    group_stds = array_ops.cov2std(group_covs)

    # Add deviation to covariance matrices
    up_stds = group_stds.copy()
    up_stds[dev_state, dev_channels[0]] *= dev_size
    up_stds[dev_state, dev_channels[1]] /= dev_size
    up_stds = np.array([np.diag(stds) for stds in up_stds])
    up_covs = up_stds @ group_corrs @ up_stds

    down_stds = group_stds.copy()
    down_stds[dev_state, dev_channels[0]] /= dev_size
    down_stds[dev_state, dev_channels[1]] *= dev_size
    down_stds = np.array([np.diag(stds) for stds in down_stds])
    down_covs = down_stds @ group_corrs @ down_stds

    # Simulate data
    time_series = []
    stc = []
    sim_session_covs = []
    for i in range(n_sessions):
        if i == up_session:
            sim_covs = up_covs
        elif i == down_session:
            sim_covs = down_covs
        else:
            sim_covs = group_covs

        sim = simulation.HMM_MVN(
            n_samples=n_samples,
            n_states=n_states,
            n_channels=n_channels,
            trans_prob="sequence",
            stay_prob=0.9,
            means="zero",
            covariances=sim_covs,
            random_seed=123 + i,
        )
        sim.standardize()
        sim_session_covs.append(sim.covariances)
        stc.append(sim.mode_time_course)
        time_series.append(sim.time_series)

    sim_session_covs = np.array(sim_session_covs)
    return time_series, sim_session_covs, stc, sim.covariances


def get_prior_dev(model):
    """Get the prior deviation from the generative model.

    Here mean of the exponential distribution is used as
    the deviation magnitude.

    Parameters
    ----------
    model : hive.Model
        HIVE model.

    Returns
    -------
    prior_session_devs : np.ndarray
        Prior session deviations.
        Shape is (n_sessions, n_states, n_channels, n_channels).
    """
    # Get the group covs
    group_covs = model.get_group_covariances()

    # Get the normalised deviation map
    dev_map = obs_mod.get_dev_map(model.model, "covs")

    # Get the deviation magnitude
    concat_embeddings = obs_mod.get_concatenated_embeddings(model.model, "covs")
    covs_dev_decoder_layer = model.get_layer("covs_dev_decoder")
    dev_mag_mod_layer = model.get_layer("covs_dev_mag_mod_beta")
    dev_mag_mod = 1 / dev_mag_mod_layer(covs_dev_decoder_layer(concat_embeddings))

    # Get the prior deviation
    dev_layer = model.get_layer("covs_dev")
    dev = dev_layer([dev_mag_mod, dev_map])

    # Get the prior session covs
    covs_layer = model.get_layer("array_covs")
    covs = np.squeeze(covs_layer([group_covs, dev]).numpy())

    prior_session_devs = covs - group_covs[None, ...]
    return prior_session_devs


# Set parameters
train = True
n_states = 5
n_channels = 11
n_samples = 25600
n_sessions = 10

# Set output directory
figures_dir = "figures/multivariate"
model_dir = "model/multivariate"

os.makedirs(figures_dir, exist_ok=True)
os.makedirs(f"{model_dir}/hive", exist_ok=True)
os.makedirs(f"{model_dir}/hmm", exist_ok=True)

time_series, sim_session_covs, stc, sim_group_covs = simulate_data(
    n_states=n_states,
    n_channels=n_channels,
    n_samples=n_samples,
    n_sessions=n_sessions,
)
# Build osl-dynamics data object
training_data = data.Data(time_series)
stc = np.concatenate(stc)

# Model settings
hmm_config = hmm.Config(
    n_states=n_states,
    n_channels=n_channels,
    sequence_length=200,
    learn_means=False,
    learn_covariances=True,
    batch_size=32,
    learning_rate=0.01,
    lr_decay=0.05,
    n_epochs=40,
)
hive_config = hive.Config(
    n_states=n_states,
    n_channels=n_channels,
    sequence_length=200,
    n_arrays=n_sessions,
    embeddings_dim=2,
    spatial_embeddings_dim=2,
    dev_n_layers=5,
    dev_n_units=32,
    dev_activation="tanh",
    dev_normalization="layer",
    dev_regularizer="l1",
    dev_regularizer_factor=10,
    learn_means=False,
    learn_covariances=True,
    batch_size=32,
    learning_rate=0.005,
    lr_decay=0.05,
    n_epochs=40,
    do_kl_annealing=True,
    kl_annealing_curve="tanh",
    kl_annealing_sharpness=10,
    n_kl_annealing_epochs=20,
)

if train:
    # Build and train HMM model
    hmm_model = hmm.Model(hmm_config)
    hmm_model.summary()
    hmm_model.set_regularizers(training_data)
    hmm_model.random_state_time_course_initialization(
        training_data, n_epochs=5, n_init=10, take=1
    )
    hmm_history = hmm_model.fit(training_data)
    hmm_model.save(f"{model_dir}/hmm")
    pickle.dump(hmm_history, open(f"{model_dir}/hmm/history.pkl", "wb"))

    # Build HIVE model
    hive_model = hive.Model(hive_config)
    hive_model.summary()

    # Set regularizers
    hive_model.set_regularizers(training_data)

    # Set deviation initializer
    hive_model.set_dev_parameters_initializer(training_data)

    # Initialisation
    hive_model.random_state_time_course_initialization(
        training_data, n_epochs=5, n_init=10, take=1
    )

    # Train model
    hive_history = hive_model.fit(training_data)

    hive_model.save(f"{model_dir}/hive")
    pickle.dump(hive_history, open(f"{model_dir}/hive/history.pkl", "wb"))
else:
    hmm_model = hmm.Model.load(f"{model_dir}/hmm")
    hmm_history = pickle.load(open(f"{model_dir}/hmm/history.pkl", "rb"))
    hive_model = hive.Model.load(f"{model_dir}/hive")
    hive_history = pickle.load(open(f"{model_dir}/hive/history.pkl", "rb"))

# Order the states
hive_inf_stc = hive_model.get_alpha(training_data, concatenate=True)
_, hive_order = modes.match_modes(stc, hive_inf_stc, return_order=True)
hive_inf_stc = hive_inf_stc[:, hive_order]
print("Dice score:", metrics.dice_coefficient(stc, hive_inf_stc))

hmm_inf_stc = hmm_model.get_alpha(training_data, concatenate=True)
_, hmm_order = modes.match_modes(stc, hmm_inf_stc, return_order=True)

# Plot training history
plotting.plot_line(
    [range(1, len(hive_history["loss"]) + 1)],
    [hive_history["loss"]],
    x_label="Epoch",
    y_label="Loss",
    title="Training history",
    filename=f"{figures_dir}/loss.png",
)

# Plot session embeddings
session_embeddings = hive_model.get_embeddings()
session_embeddings -= np.mean(session_embeddings, axis=0)
session_embeddings /= np.std(session_embeddings, axis=0)

fig, ax = plotting.plot_scatter(
    [session_embeddings[:, 0]],
    [session_embeddings[:, 1]],
    annotate=[[1, 2] + [""] * (n_sessions - 2)],
    title="Session embeddings",
)
fig.savefig(f"{figures_dir}/session_embeddings.png", dpi=300)

# Plot simulated and inferred group covariances
hive_inf_group_covs = hive_model.get_group_covariances()[hive_order]
hmm_inf_group_covs = hmm_model.get_covariances()[hmm_order]

vmin = np.min([sim_group_covs, hive_inf_group_covs, hmm_inf_group_covs])
vmax = np.max([sim_group_covs, hive_inf_group_covs, hmm_inf_group_covs])

fig, axes = plt.subplots(3, 5, figsize=(15, 9), squeeze=False)
for i in range(n_states):
    axes[0, i].matshow(sim_group_covs[i], cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0, i].set_title(f"State {i + 1}", fontsize=16)
    axes[1, i].matshow(hmm_inf_group_covs[i], cmap="viridis", vmin=vmin, vmax=vmax)
    im = axes[2, i].matshow(
        hive_inf_group_covs[i], cmap="viridis", vmin=vmin, vmax=vmax
    )

axes[0, 0].set_ylabel("Simulated", fontsize=16)
axes[1, 0].set_ylabel("HMM-DE inferred", fontsize=16)
axes[2, 0].set_ylabel("HIVE inferred", fontsize=16)
fig.tight_layout()
fig.subplots_adjust(right=0.8)
color_bar_axis = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=color_bar_axis)
fig.savefig(f"{figures_dir}/group_covs.png", dpi=300)

# Get the prior deviation
prior_session_devs = get_prior_dev(hive_model)[:, hive_order]
sim_session_devs = sim_session_covs - sim_group_covs[None, ...]

# Plot simulated and inferred session deviations
vmin = np.min([sim_session_devs[[0, 1, 2], 0], prior_session_devs[[0, 1, 2], 0]])
vmax = np.max([sim_session_devs[[0, 1, 2], 0], prior_session_devs[[0, 1, 2], 0]])

fig, axes = plt.subplots(2, 2, figsize=(6, 6), squeeze=False)
axes[0, 0].matshow(sim_session_devs[0, 0], cmap="viridis", vmin=vmin, vmax=vmax)
axes[0, 0].set_ylabel("Simulated", fontsize=16)
axes[0, 0].set_title("Session 1", fontsize=16)
axes[0, 1].matshow(sim_session_devs[1, 0], cmap="viridis", vmin=vmin, vmax=vmax)
axes[0, 1].set_title("Session 2", fontsize=16)
axes[1, 0].matshow(prior_session_devs[0, 0], cmap="viridis", vmin=vmin, vmax=vmax)
axes[1, 0].set_ylabel("Inferred", fontsize=16)
im = axes[1, 1].matshow(prior_session_devs[1, 0], cmap="viridis", vmin=vmin, vmax=vmax)
fig.tight_layout()
fig.subplots_adjust(right=0.8)
color_bar_axis = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=color_bar_axis)
fig.savefig(f"{figures_dir}/devs.png", dpi=300)
plt.close(fig)

# Clean up directory
training_data.delete_dir()
