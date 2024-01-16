from sys import argv

if len(argv) != 2:
    print(f"Need to pass the run id, e.g. python {argv[0]} 1")
    exit()
id = argv[1]

from glob import glob

from osl_dynamics import run_pipeline
from osl_dynamics.data import Data
from osl_dynamics.inference import tf_ops

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


# Set directories
output_dir = f"results/hmm/run{id}"
tmp_dir = f"tmp/hmm/run{id}"
data_dir = "/well/woolrich/projects/camcan/winter23/src"

config = """
    train_hmm:
        config_kwargs:
            n_states: 8
            learn_means: False
            learn_covariances: True
            batch_size: 32
            n_epochs: 10
            learning_rate: 0.001
    dual_estimation:
        n_jobs: 16
"""

training_data = load_data(data_dir, tmp_dir)
run_pipeline(
    config,
    output_dir=output_dir,
    data=training_data,
)
