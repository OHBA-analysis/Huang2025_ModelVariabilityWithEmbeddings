import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from osl_dynamics import data, simulation
from osl_dynamics.inference import tf_ops, metrics, modes
from osl_dynamics.models import hmm, hive
from osl_dynamics.utils import plotting, set_random_seed


train_hmm = True
train_hive = True
dual_estimation = True

# Output directories
model_dir = "model/recover_structure"
figures_dir = "figures/recover_structure"

os.makedirs(model_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

# GPU settings
tf_ops.gpu_growth()
set_random_seed(0)

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
    n_sessions=100,
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
    learn_trans_prob=True,
    batch_size=128,
    learning_rate=0.005,
    lr_decay=0.05,
    n_epochs=40,
    do_kl_annealing=True,
    kl_annealing_curve="tanh",
    kl_annealing_sharpness=10,
    n_kl_annealing_epochs=20,
)

# Simulate data
print("Simulating data")
sim = simulation.MSess_HMM_MVN(
    n_samples=3000,
    trans_prob="sequence",
    session_means="zero",
    session_covariances="random",
    n_states=hive_config.n_states,
    n_channels=hive_config.n_channels,
    n_covariances_act=3,
    n_sessions=hive_config.n_sessions,
    embeddings_dim=2,
    spatial_embeddings_dim=2,
    embeddings_scale=0.002,
    n_groups=3,
    between_group_scale=0.2,
    stay_prob=0.9,
    observation_error=0.2,
)
sim.standardize()
training_data = data.Data([mtc for mtc in sim.time_series])
training_data.add_session_labels(
    "session_id", np.arange(hive_config.n_sessions), "categorical"
)

# Build and train models
if train_hmm:
    hmm_model = hmm.Model(hmm_config)
    hmm_model.set_regularizers(training_data)
    print("Training HMM")
    # Initialisation
    hmm_model.random_state_time_course_initialization(
        training_data, n_init=5, n_epochs=3, take=1, use_tqdm=True
    )
    # Full training
    hmm_history = hmm_model.fit(training_data, use_tqdm=True)

    pickle.dump(hmm_history, open(f"{model_dir}/hmm_history.pkl", "wb"))
    hmm_model.save(f"{model_dir}/hmm")
else:
    print("Loading HMM")
    hmm_model = hmm.Model.load(f"{model_dir}/hmm")
    hmm_config = hmm_model.config
    hmm_history = pickle.load(open(f"{model_dir}/hmm_history.pkl", "rb"))


if train_hive:
    hive_model = hive.Model(hive_config)
    hive_model.set_regularizers(training_data)
    # Initialise deviation parameters
    hive_model.set_dev_parameters_initializer(training_data)
    print("Training HIVE")
    # Initialisation
    hive_model.random_state_time_course_initialization(
        training_data,
        n_init=10,
        n_epochs=5,
        take=1,
        use_tqdm=True,
    )
    # Full training
    hive_history = hive_model.fit(training_data, use_tqdm=True)

    pickle.dump(hive_history, open(f"{model_dir}/hive_history.pkl", "wb"))
    hive_model.save(f"{model_dir}/hive")
else:
    print("Loading HIVE")
    hive_model = hive.Model.load(f"{model_dir}/hive")
    hive_config = hive_model.config
    hive_history = pickle.load(open(f"{model_dir}/hive_history.pkl", "rb"))

if dual_estimation:
    hmm_session_covs = hmm_model.dual_estimation(training_data)[1]
    with open(f"{model_dir}/hmm_session_covs.pkl", "wb") as file:
        pickle.dump(hmm_session_covs, file)
else:
    with open(f"{model_dir}/hmm_session_covs.pkl", "rb") as file:
        hmm_session_covs = pickle.load(file)

plotting.plot_line(
    [range(hmm_config.n_epochs), range(hive_config.n_epochs)],
    [hmm_history["loss"], hive_history["loss"]],
    labels=["HMM", "HIVE"],
    x_label="Epoch",
    y_label="Loss",
    title="Training loss",
    filename=f"{figures_dir}/training_loss.png",
)

# Get and order the mode time courses
sim_alp = np.concatenate(sim.mode_time_course)
hmm_alp = hmm_model.get_alpha(training_data, concatenate=True)
hive_alp = hive_model.get_alpha(training_data, concatenate=True)

hmm_order = modes.match_modes(sim_alp, hmm_alp, return_order=True)[1]
hive_order = modes.match_modes(sim_alp, hive_alp, return_order=True)[1]

hmm_alp = hmm_alp[:, hmm_order]
hive_alp = hive_alp[:, hive_order]

# Get and order the session covariances
sim_session_covs = sim.session_covariances

hmm_session_covs = hmm_session_covs[:, hmm_order, :, :]
hive_session_covs = hive_model.get_session_means_covariances()[1]
hive_session_covs = hive_session_covs[:, hive_order, :, :]

# Plot the simulated and inferred session embeddings
sim_se = sim.embeddings
inf_se = hive_model.get_summed_embeddings()
inf_se = LinearDiscriminantAnalysis(n_components=2).fit_transform(
    inf_se, sim.assigned_groups
)
group_masks = [sim.assigned_groups == i for i in range(sim.n_groups)]

print(f"HMM DICE: {metrics.dice_coefficient(sim_alp, hmm_alp):.2f}")
print(f"HIVE DICE: {metrics.dice_coefficient(sim_alp, hive_alp):.2f}")

markers = ["o", "x", "v"]
fig, ax = plotting.create_figure(figsize=(8, 6))
for i in range(sim.n_groups):
    sns.scatterplot(
        x=sim_se[group_masks[i], 0],
        y=sim_se[group_masks[i], 1],
        ax=ax,
        marker=markers[i],
        label=f"Group {i + 1}",
        s=80,
    )
for i in range(sim.n_sessions):
    ax.annotate(
        str(i + 1),
        (sim_se[i, 0], sim_se[i, 1]),
        fontsize=12,
        color="black",
    )
ax.set_xlabel("Dimension 1", fontsize=16)
ax.set_ylabel("Dimension 2", fontsize=16)
ax.set_title("Simulated session embedding vectors", fontsize=20)
fig.tight_layout()
fig.savefig(f"{figures_dir}/simulated_session_embeddings.png")

fig, ax = plotting.create_figure(figsize=(8, 6))
for i in range(sim.n_groups):
    sns.scatterplot(
        x=inf_se[group_masks[i], 0],
        y=inf_se[group_masks[i], 1],
        ax=ax,
        marker=markers[i],
        label=f"Group {i + 1}",
        s=80,
    )
for i in range(sim.n_sessions):
    ax.annotate(
        str(i + 1),
        (inf_se[i, 0], inf_se[i, 1]),
        fontsize=12,
        color="black",
    )
ax.set_xlabel("Dimension 1", fontsize=16)
ax.set_ylabel("Dimension 2", fontsize=16)
ax.set_title("Inferred session embedding vectors", fontsize=20)
fig.tight_layout()
fig.savefig(f"{figures_dir}/inferred_session_embeddings.png")

# Plot the mode time courses
fig, ax = plotting.plot_alpha(
    sim_alp,
    hmm_alp,
    hive_alp,
    n_samples=1000,
    cmap="tab20",
)
ax[0].set_ylabel("Simulated", fontsize=16)
ax[1].set_ylabel("HMM-DE", fontsize=16)
ax[2].set_ylabel("HIVE", fontsize=16)
ax[0].set_title("First 1000 samples of session 1", fontsize=20)
ax[2].set_xlabel("Sample", fontsize=16)
fig.tight_layout()
fig.savefig(f"{figures_dir}/state_time_courses.png")

# Pairwise cosine distances
sim_pw_cos = np.empty((sim.n_states, sim.n_sessions, sim.n_sessions))
hmm_pw_cos = np.empty((sim.n_states, sim.n_sessions, sim.n_sessions))
hive_pw_cos = np.empty((sim.n_states, sim.n_sessions, sim.n_sessions))
for j in range(sim.n_states):
    sim_pw_cos[j] = 1 - metrics.pairwise_congruence_coefficient(sim_session_covs[:, j])
    hmm_pw_cos[j] = 1 - metrics.pairwise_congruence_coefficient(hmm_session_covs[:, j])
    hive_pw_cos[j] = 1 - metrics.pairwise_congruence_coefficient(
        hive_session_covs[:, j]
    )

# Linear regression on pairwise cosine distances
m, n = np.tril_indices(sim.n_sessions, k=-1)
lr_hmm = LinearRegression().fit(
    sim_pw_cos[:, m, n].reshape(-1, 1),
    hmm_pw_cos[:, m, n].flatten(),
)
r2_hmm = lr_hmm.score(
    sim_pw_cos[:, m, n].reshape(-1, 1),
    hmm_pw_cos[:, m, n].flatten(),
)
lr_hive = LinearRegression().fit(
    sim_pw_cos[:, m, n].reshape(-1, 1),
    hive_pw_cos[:, m, n].flatten(),
)
r2_hive = lr_hive.score(
    sim_pw_cos[:, m, n].reshape(-1, 1),
    hive_pw_cos[:, m, n].flatten(),
)

# Plot the pairwise cosine distances
fig, ax = plotting.create_figure(figsize=(8, 6))
sns.set_palette("colorblind", n_colors=sim.n_states)
for i in range(sim.n_states):
    sns.scatterplot(
        x=sim_pw_cos[i, m, n],
        y=hmm_pw_cos[i, m, n],
        ax=ax,
        label=f"State {i + 1}",
    )
ax.axline((0, 0), slope=1, color="black")
ax.axline((0, lr_hmm.intercept_), slope=lr_hmm.coef_[0], color="red")
ax.set_xlabel("Simulated pairwise cosine distance", fontsize=16)
ax.set_ylabel("Inferred pairwise cosine distance", fontsize=16)
ax.set_title("HMM-DE " + r"$r^2 = $" + f"{r2_hmm:.2f}", fontsize=20)
fig.tight_layout()
fig.savefig(f"{figures_dir}/pairwise_cosine_distances_hmm.png")

fig, ax = plotting.create_figure(figsize=(8, 6))
sns.set_palette("colorblind", n_colors=sim.n_states)
for i in range(sim.n_states):
    sns.scatterplot(
        x=sim_pw_cos[i, m, n],
        y=hive_pw_cos[i, m, n],
        ax=ax,
        label=f"State {i + 1}",
    )
ax.axline((0, 0), slope=1, color="black")
ax.axline((0, lr_hive.intercept_), slope=lr_hive.coef_[0], color="red")
ax.set_xlabel("Simulated pairwise cosine distance", fontsize=16)
ax.set_ylabel("Inferred pairwise cosine distance", fontsize=16)
ax.set_title("HIVE " + r"$r^2 = $" + f"{r2_hive:.2f}", fontsize=20)
fig.tight_layout()
fig.savefig(f"{figures_dir}/pairwise_cosine_distances_hive.png")
