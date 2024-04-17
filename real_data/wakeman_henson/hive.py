from sys import argv

if len(argv) != 3:
    print(f"Please pass the run id and embedding dimension, e.g. python {argv[0]} 1 10")
    exit()
id = argv[1]
embeddings_dim = argv[2]

from glob import glob

from osl_dynamics import run_pipeline
from osl_dynamics.data import Data
from osl_dynamics.inference import tf_ops


# GPU settings
tf_ops.gpu_growth()


# Helper functions
def load_data(data_dir, store_dir, use_tfrecord=True, buffer_size=2000, n_jobs=16):
    """Load the data."""

    data_paths = sorted(glob(f"{data_dir}/sub*_run*/sflip_parc-raw.fif"))
    training_data = Data(
        data_paths,
        sampling_frequency=250,
        picks="misc",
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


# Set directories
output_dir = f"results/hive_{embeddings_dim}/run{id}"
tmp_dir = f"tmp/hive_{embeddings_dim}/run{id}"
data_dir = "/well/woolrich/projects/wakeman_henson/spring23/src"

config = f"""
    train_hive:
        config_kwargs:
            n_states: 6
            learn_means: False
            learn_covariances: True
            embeddings_dim: {embeddings_dim}
"""
training_data = load_data(data_dir, tmp_dir, n_jobs=16)
run_pipeline(
    config,
    output_dir=output_dir,
    data=training_data,
)
