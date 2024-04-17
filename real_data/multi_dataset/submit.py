import os


def write_hive_job_script(run, embeddings_dim):
    with open("job.sh", "w") as file:
        name = f"hive-{embeddings_dim}-{run}"
        file.write("#!/bin/bash\n")
        file.write(f"#SBATCH -J {name}\n")
        file.write(f"#SBATCH -o outputs/{name}.out\n")
        file.write("#SBATCH -p gpu_short\n")
        file.write(f"#SBATCH --gres gpu:1\n")
        file.write("source activate osld\n")
        file.write(f"python hive.py {run} {embeddings_dim}")


def write_hmm_job_script(run):
    with open("job.sh", "w") as file:
        name = f"hmm-{run}"
        file.write("#!/bin/bash\n")
        file.write(f"#SBATCH -J {name}\n")
        file.write(f"#SBATCH -o outputs/{name}.out\n")
        file.write("#SBATCH -p gpu_short\n")
        file.write(f"#SBATCH --gres gpu:1\n")
        file.write("source activate osld\n")
        file.write(f"python hmm.py {run}")


os.makedirs("outputs", exist_ok=True)

for dim in [3, 5, 10, 20, 30]:
    for run in range(1, 11):
        write_hive_job_script(run, dim)
        os.system("sbatch job.sh")
        os.system("rm job.sh")

for run in range(1, 11):
    write_hmm_job_script(run)
    os.system("sbatch job.sh")
    os.system("rm job.sh")
