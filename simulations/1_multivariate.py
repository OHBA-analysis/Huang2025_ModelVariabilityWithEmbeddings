import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pickle

from osl_dynamics import data, simulation, array_ops
from osl_dynamics.inference import metrics, modes, tf_ops
from osl_dynamics.models import hmm, sehmm, obs_mod
from osl_dynamics.utils import plotting

# GPU settings
tf_ops.gpu_growth()


# Helper functions
def simulate_data(
    n_states,
    n_channels,
    n_samples,
    n_subjects,
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
        Number of samples per subject.
    n_subjects : int
        Number of subjects. Must be at least 3.
    dev_state : int, optional
        State with deviation, by default 0.
    dev_size : int, optional
        Deviation size, by default 5.

    Returns
    -------
    time_series : list of array
        List of data for each subject.
        Each element has shape (n_samples, n_channels).
    sim_subject_covs : np.ndarray
        Simulated subject specific covariances.
        Shape is (n_subjects, n_states, n_channels, n_channels).
    stc : list of array
        List of simulated state time courses for each subject.
        Each element has shape (n_samples, n_states).
    sim_group_covs : np.ndarray
        Simulated group level covariances.
        Shape is (n_states, n_channels, n_channels).
    """
    if n_subjects < 3:
        raise ValueError("n_subjects must be at least 3")

    # channels that have deviation
    dev_channels = [0, 1]

    # subjects with up and down deviations
    up_subject = 0
    down_subject = 1

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
    sim_subject_covs = []
    for i in range(n_subjects):
        if i == up_subject:
            sim_covs = up_covs
        elif i == down_subject:
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
        sim_subject_covs.append(sim.covariances)
        stc.append(sim.mode_time_course)
        time_series.append(sim.time_series)

    sim_subject_covs = np.array(sim_subject_covs)
    return time_series, sim_subject_covs, stc, sim.covariances


def get_prior_dev(model):
    """Get the prior deviation from the generative model.

    Here mean of the exponential distribution is used as
    the deviation magnitude.

    Parameters
    ----------
    model : sehmm.Model
        SE-HMM model.

    Returns
    -------
    prior_subject_devs : np.ndarray
        Prior subject deviations.
        Shape is (n_subjects, n_states, n_channels, n_channels).
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

    # Get the prior subject covs
    covs_layer = model.get_layer("subject_covs")
    covs = covs_layer([group_covs, dev]).numpy()

    prior_subject_devs = covs - group_covs[None, ...]
    return prior_subject_devs


# Set parameters
train = True
n_states = 5
n_channels = 11
n_samples = 25600
n_subjects = 10

# Set output directory
figures_dir = "figures/multivariate"
model_dir = "models/multivariate"

os.makedirs(figures_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

time_series, sim_subject_covs, stc, sim_group_covs = simulate_data(
    n_states=n_states,
    n_channels=n_channels,
    n_samples=n_samples,
    n_subjects=n_subjects,
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
    n_epochs=3,
)
sehmm_config = sehmm.Config(
    n_states=n_states,
    n_channels=n_channels,
    sequence_length=200,
    n_subjects=n_subjects,
    subject_embeddings_dim=2,
    mode_embeddings_dim=2,
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
    # Initialize parameters with HMM
    init_model = hmm.Model(hmm_config)
    init_model.set_regularizers(training_data)
    init_model.random_state_time_course_initialization(
        training_data, n_epochs=3, n_init=5, take=1
    )
    _, init_dual_covs = init_model.dual_estimation(training_data)
    init_dual_covs = np.reshape(init_dual_covs, (n_subjects, -1))
    pca = PCA(n_components=sehmm_config.subject_embeddings_dim)

    sehmm_config.initial_covariances = init_model.get_covariances()
    sehmm_config.initial_trans_prob = init_model.get_trans_prob()

    # Build SE-HMM model
    model = sehmm.Model(sehmm_config)
    model.summary()

    # Set regularizers
    model.set_regularizers(training_data)

    # Set deviation initializer
    model.set_dev_parameters_initializer(training_data)

    # Set subject embeddings initializer
    model.set_subject_embeddings_initializer(pca.fit_transform(init_dual_covs))

    # Initialize model with subject embeddings fixed
    with model.set_trainable("subject_embeddings", False):
        model.random_state_time_course_initialization(
            training_data, n_epochs=5, n_init=10, take=1
        )

    # Train model
    history = model.fit(training_data)
    model.save(model_dir)
    pickle.dump(history, open(f"{model_dir}/history.pkl", "wb"))
else:
    model = sehmm.Model.load(model_dir)
    history = pickle.load(open(f"{model_dir}/history.pkl", "rb"))

# Order the states
inf_stc = model.get_alpha(training_data, concatenate=True)
_, order = modes.match_modes(stc, inf_stc, return_order=True)
inf_stc = inf_stc[:, order]
print("Dice score:", metrics.dice_coefficient(stc, inf_stc))

# Plot training history
plotting.plot_line(
    [range(1, len(history["loss"]) + 1)],
    [history["loss"]],
    x_label="Epoch",
    y_label="Loss",
    title="Training history",
    filename=f"{figures_dir}/loss.png",
)

# Plot subject embeddings
subject_embeddings = model.get_subject_embeddings()
subject_embeddings -= np.mean(subject_embeddings, axis=0)
subject_embeddings /= np.std(subject_embeddings, axis=0)

plotting.plot_scatter(
    [subject_embeddings[:, 0]],
    [subject_embeddings[:, 1]],
    annotate=[[str(i + 1) for i in range(n_subjects)]],
    title="Subject embeddings",
    filename=f"{figures_dir}/subject_embeddings.png",
)

# Plot simulated and inferred group covariances
inf_group_covs = model.get_group_covariances()[order]
plotting.plot_matrices(
    sim_group_covs,
    main_title="Simulated group covariances",
    filename=f"{figures_dir}/sim_group_covs.png",
)
plotting.plot_matrices(
    inf_group_covs,
    main_title="Inferred group covariances",
    filename=f"{figures_dir}/inf_group_covs.png",
)

# Get the prior deviation
prior_subject_devs = get_prior_dev(model)[:, order]
sim_subject_devs = sim_subject_covs - sim_group_covs[None, ...]

# Plot simulated and inferred subject deviations
vmin = np.min([sim_subject_devs, prior_subject_devs])
vmax = np.max([sim_subject_devs, prior_subject_devs])

fig, axes = plt.subplots(3, 5, figsize=(15, 9))
for i in range(5):
    axes[0, i].matshow(sim_subject_devs[0, i], cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1, i].matshow(sim_subject_devs[1, i], cmap="viridis", vmin=vmin, vmax=vmax)
    axes[2, i].matshow(sim_subject_devs[2, i], cmap="viridis", vmin=vmin, vmax=vmax)

axes[0, 0].set_ylabel("Subject 1", fontsize=16)
axes[1, 0].set_ylabel("Subject 2", fontsize=16)
axes[2, 0].set_ylabel("Other subjects", fontsize=16)
for i in range(5):
    axes[0, i].set_title(f"State {i}", fontsize=16)

fig.tight_layout()
fig.savefig(f"{figures_dir}/sim_covs_dev.png", dpi=300)
plt.close(fig)

fig, axes = plt.subplots(3, 5, figsize=(15, 9))
for i in range(5):
    axes[0, i].matshow(prior_subject_devs[0, i], cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1, i].matshow(prior_subject_devs[1, i], cmap="viridis", vmin=vmin, vmax=vmax)
    axes[2, i].matshow(prior_subject_devs[2, i], cmap="viridis", vmin=vmin, vmax=vmax)

axes[0, 0].set_ylabel("Subject 1", fontsize=16)
axes[1, 0].set_ylabel("Subject 2", fontsize=16)
axes[2, 0].set_ylabel("Other subjects", fontsize=16)
for i in range(5):
    axes[0, i].set_title(f"State {i}", fontsize=16)

fig.tight_layout()
fig.savefig(f"{figures_dir}/prior_covs_dev.png", dpi=300)
plt.close(fig)

# Clean up directory
training_data.delete_dir()
