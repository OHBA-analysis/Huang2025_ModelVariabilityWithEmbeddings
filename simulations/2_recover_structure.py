import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

from osl_dynamics import data, simulation
from osl_dynamics.inference import tf_ops, metrics, modes
from osl_dynamics.models import hmm, sehmm
from osl_dynamics.utils import plotting


train_hmm = True
train_sehmm = True
dual_estimation = True

# Output directories
model_dir = "model/recover_structure"
figures_dir = "figures/recover_structure"

os.makedirs(model_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

# GPU settings
tf_ops.gpu_growth()

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
    n_subjects=100,
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
sim = simulation.MSubj_HMM_MVN(
    n_samples=3000,
    trans_prob="sequence",
    subject_means="zero",
    subject_covariances="random",
    n_states=sehmm_config.n_states,
    n_channels=sehmm_config.n_channels,
    n_covariances_act=5,
    n_subjects=sehmm_config.n_subjects,
    n_subject_embedding_dim=2,
    n_mode_embedding_dim=2,
    subject_embedding_scale=0.002,
    n_groups=3,
    between_group_scale=0.2,
    stay_prob=0.9,
    random_seed=1234,
    observation_error=0.2,
)
sim.standardize()
training_data = data.Data(
    [mtc for mtc in sim.time_series], use_tfrecord=True, buffer_size=2000
)

# Build and train models
if train_hmm:
    hmm_model = hmm.Model(hmm_config)
    hmm_model.set_regularizers(training_data)
    print("Training HMM")
    # Initialisation
    hmm_model.random_state_time_course_initialization(
        training_data, n_init=5, n_epochs=3, take=1
    )
    # Full training
    hmm_history = hmm_model.fit(training_data)

    pickle.dump(hmm_history, open(f"{model_dir}/hmm_history.pkl", "wb"))
    hmm_model.save(f"{model_dir}/hmm")
else:
    print("Loading HMM")
    hmm_model = hmm.Model.load(f"{model_dir}/hmm")
    hmm_config = hmm_model.config
    hmm_history = pickle.load(open(f"{model_dir}/hmm_history.pkl", "rb"))


if train_sehmm:
    sehmm_model = sehmm.Model(sehmm_config)
    sehmm_model.set_regularizers(training_data)
    # Initialise deviation parameters
    sehmm_model.set_dev_parameters_initializer(training_data)
    print("Training SE-HMM")
    # Initialisation
    sehmm_model.random_state_time_course_initialization(
        training_data,
        n_init=10,
        n_epochs=5,
        take=1,
    )
    # Full training
    sehmm_history = sehmm_model.fit(training_data)

    pickle.dump(sehmm_history, open(f"{model_dir}/sehmm_history.pkl", "wb"))
    sehmm_model.save(f"{model_dir}/sehmm")
else:
    print("Loading SE-HMM")
    sehmm_model = sehmm.Model.load(f"{model_dir}/sehmm")
    sehmm_config = sehmm_model.config
    sehmm_history = pickle.load(open(f"{model_dir}/sehmm_history.pkl", "rb"))

if dual_estimation:
    hmm_subject_covs = hmm_model.dual_estimation(training_data)[1]
    with open(f"{model_dir}/hmm_subject_covs.pkl", "wb") as file:
        pickle.dump(hmm_subject_covs, file)
else:
    with open(f"{model_dir}/hmm_subject_covs.pkl", "rb") as file:
        hmm_subject_covs = pickle.load(file)

# Plot the training histories
plotting.plot_line(
    [range(hmm_config.n_epochs), range(sehmm_config.n_epochs)],
    [hmm_history["loss"], sehmm_history["loss"]],
    labels=["HMM", "SE-HMM"],
    x_label="Epoch",
    y_label="Loss",
    filename=f"{figures_dir}/loss.png",
)

# Get the free energy
hmm_free_energy = hmm_model.free_energy(training_data)
sehmm_free_energy = sehmm_model.free_energy(training_data, verbose=0)

# Get and order the mode time courses
sim_alp = np.concatenate(sim.mode_time_course)
hmm_alp = hmm_model.get_alpha(training_data, concatenate=True)
sehmm_alp = sehmm_model.get_alpha(training_data, concatenate=True)

hmm_order = modes.match_modes(sim_alp, hmm_alp, return_order=True)[1]
sehmm_order = modes.match_modes(sim_alp, sehmm_alp, return_order=True)[1]

hmm_alp = hmm_alp[:, hmm_order]
sehmm_alp = sehmm_alp[:, sehmm_order]

hmm_trans_prob = hmm_model.get_trans_prob()
sehmm_trans_prob = sehmm_model.get_trans_prob()
hmm_trans_prob = hmm_trans_prob[np.ix_(hmm_order, hmm_order)]
sehmm_trans_prob = sehmm_trans_prob[np.ix_(sehmm_order, sehmm_order)]

# Plot the transition probabilities
plotting.plot_matrices(
    [hmm_trans_prob, sehmm_trans_prob],
    filename=f"{figures_dir}/trans_prob.png",
)

# Get and order the subject covariances
sim_subject_covs = sim.subject_covariances

hmm_subject_covs = hmm_subject_covs[:, hmm_order, :, :]
sehmm_subject_covs = sehmm_model.get_subject_means_covariances()[1]
sehmm_subject_covs = sehmm_subject_covs[:, sehmm_order, :, :]

# Plot the simulated and inferred subject embeddings
sim_se = sim.subject_embeddings
inf_se = sehmm_model.get_subject_embeddings()
inf_se -= np.mean(inf_se, axis=0)
inf_se /= np.std(inf_se, axis=0)
group_masks = [sim.assigned_groups == i for i in range(sim.n_groups)]

plotting.plot_scatter(
    [sim_se[group_mask, 0] for group_mask in group_masks],
    [sim_se[group_mask, 1] for group_mask in group_masks],
    labels=[f"Group {i + 1}" for i in range(sim.n_groups)],
    annotate=[
        np.array([str(i) for i in range(sim.n_subjects)])[group_mask]
        for group_mask in group_masks
    ],
    legend_loc="lower right",
    markers=["o", "x", "v"],
    x_label="dim1",
    y_label="dim2",
    filename=f"{figures_dir}/sim_se.png",
)
plotting.plot_scatter(
    [inf_se[group_mask, 0] for group_mask in group_masks],
    [inf_se[group_mask, 1] for group_mask in group_masks],
    labels=[f"Group {i + 1}" for i in range(sim.n_groups)],
    annotate=[
        np.array([str(i) for i in range(sim.n_subjects)])[group_mask]
        for group_mask in group_masks
    ],
    legend_loc="lower right",
    markers=["o", "x", "v"],
    x_label="dim1",
    y_label="dim2",
    filename=f"{figures_dir}/inf_se.png",
)

# Plot the mode time courses
plotting.plot_alpha(
    sim_alp,
    hmm_alp,
    sehmm_alp,
    n_samples=1000,
    y_labels=["Simulated", "HMM", "SE-HMM"],
    filename=f"{figures_dir}/mtc.png",
)

# Pairwise cosine distances
sim_pw_cos = np.empty((sim.n_states, sim.n_subjects, sim.n_subjects))
hmm_pw_cos = np.empty((sim.n_states, sim.n_subjects, sim.n_subjects))
sehmm_pw_cos = np.empty((sim.n_states, sim.n_subjects, sim.n_subjects))
for j in range(sim.n_states):
    sim_pw_cos[j] = 1 - metrics.pairwise_congruence_coefficient(sim_subject_covs[:, j])
    hmm_pw_cos[j] = 1 - metrics.pairwise_congruence_coefficient(hmm_subject_covs[:, j])
    sehmm_pw_cos[j] = 1 - metrics.pairwise_congruence_coefficient(
        sehmm_subject_covs[:, j]
    )

# Linear regression on pairwise cosine distances
m, n = np.tril_indices(sim.n_subjects, k=-1)
lr_hmm = LinearRegression().fit(
    sim_pw_cos[:, m, n].reshape(-1, 1),
    hmm_pw_cos[:, m, n].flatten(),
)
r2_hmm = lr_hmm.score(
    sim_pw_cos[:, m, n].reshape(-1, 1),
    hmm_pw_cos[:, m, n].flatten(),
)
lr_sehmm = LinearRegression().fit(
    sim_pw_cos[:, m, n].reshape(-1, 1),
    sehmm_pw_cos[:, m, n].flatten(),
)
r2_sehmm = lr_sehmm.score(
    sim_pw_cos[:, m, n].reshape(-1, 1),
    sehmm_pw_cos[:, m, n].flatten(),
)

# Plot the pairwise cosine distances
fig, ax = plotting.plot_scatter(
    sim_pw_cos[:, m, n],
    hmm_pw_cos[:, m, n],
    labels=[f"State {i + 1}" for i in range(hmm_config.n_states)],
    x_label="Simulated PW Cos",
    y_label="Inferred PW Cos",
    title=r"$r^2 = $" + f"{r2_hmm:.2f}",
)
ax.axline((0, 0), slope=1, color="black")
ax.axline((0, lr_hmm.intercept_), slope=lr_hmm.coef_[0], color="red")
fig.savefig(f"{figures_dir}/pw_cos_hmm.png")
plt.close(fig)

fig, ax = plotting.plot_scatter(
    sim_pw_cos[:, m, n],
    sehmm_pw_cos[:, m, n],
    labels=[f"State {i + 1}" for i in range(sehmm_config.n_states)],
    x_label="Simulated PW Cos",
    y_label="Inferred PW Cos",
    title=r"$r^2 = $" + f"{r2_sehmm:.2f}",
)
ax.axline((0, 0), slope=1, color="black")
ax.axline((0, lr_sehmm.intercept_), slope=lr_sehmm.coef_[0], color="red")
fig.savefig(f"{figures_dir}/pw_cos_sehmm.png")
plt.close(fig)

plotting.plot_matrices(
    [np.mean(sim_pw_cos, 0), np.mean(hmm_pw_cos, 0), np.mean(sehmm_pw_cos, 0)],
    titles=["Simulated", "HMM", "SE-HMM"],
    filename=f"{figures_dir}/pw_cos.png",
)

print("Free energy")
print(f"HMM: {hmm_free_energy:.2f}")
print(f"SE-HMM: {sehmm_free_energy:.2f}")
print(f"HMM DICE: {metrics.dice_coefficient(sim_alp, hmm_alp):.2f}")
print(f"SE-HMM DICE: {metrics.dice_coefficient(sim_alp, sehmm_alp):.2f}")

# Clean up
training_data.delete_dir()
