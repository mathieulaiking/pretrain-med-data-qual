from IPython.display import clear_output
import subprocess
import os
import time
from datetime import datetime
import glob
from threading import Event
import signal


def display_slurm_queue(name=None, timestep=5):
    """
    IDRIS function
    Interactive Slurm Queue for Ipython (Jupyter Notebook)
    Refreshed display from squeue command. Loop turns until queue is empty.
    If you want break the loop before the end, you have to stop the cell (no impact on the slurm Queue)

    Args:
        name (string): Filter job list by name, default None (no filter)
        timestep (int): refresh time step, default 5 seconds
    """

    def quit(signo, _frame):
        print("Key interrupt")
        exit.set()

    for sig in ("TERM", "HUP", "INT"):
        signal.signal(getattr(signal, "SIG" + sig), quit)
    exit = Event()
    if name:
        sq = (
            subprocess.run(
                f'squeue -u {os.environ["USER"]} -n {name}'.split(), capture_output=True
            )
            .stdout.decode("utf8")
            .splitlines()
        )
    else:
        sq = (
            subprocess.run(
                f'squeue -u {os.environ["USER"]}'.split(), capture_output=True
            )
            .stdout.decode("utf8")
            .splitlines()
        )
    while len(sq) >= 2 and not exit.is_set():
        clear_output(wait=True)
        for l in sq:
            print(l)
        time.sleep(timestep)
        if name:
            sq = (
                subprocess.run(
                    f'squeue -u {os.environ["USER"]} -n {name}'.split(),
                    capture_output=True,
                )
                .stdout.decode("utf8")
                .splitlines()
            )
        else:
            sq = (
                subprocess.run(
                    f'squeue -u {os.environ["USER"]}'.split(), capture_output=True
                )
                .stdout.decode("utf8")
                .splitlines()
            )
        exit.wait(1)
    if not exit.is_set():
        print("\n Done!")


def display_slurm_queue_jupyter(name=None, timestep=5):
    """
    IDRIS function
    Interactive Slurm Queue for Ipython (Jupyter Notebook)
    Refreshed display from squeue command. Loop turns until queue is empty.
    If you want break the loop before the end, you have to stop the cell (no impact on the slurm Queue)

    Args:
        name (string): Filter job list by name, default None (no filter)
        timestep (int): refresh time step, default 5 seconds
    """

    def quit(signo, _frame):
        print("Key interrupt")
        exit.set()

    for sig in ("TERM", "HUP", "INT"):
        signal.signal(getattr(signal, "SIG" + sig), quit)
    exit = Event()
    if name:
        sq = (
            subprocess.run(
                f'squeue -u {os.environ["USER"]} -n {name}'.split(), capture_output=True
            )
            .stdout.decode("utf8")
            .splitlines()
        )
    else:
        sq = (
            subprocess.run(
                f'squeue -u {os.environ["USER"]}'.split(), capture_output=True
            )
            .stdout.decode("utf8")
            .splitlines()
        )
    while len(sq) >= 2 and not exit.is_set():
        clear_output(wait=True)
        for l in sq:
            print(l)
        time.sleep(timestep)
        if name:
            sq = (
                subprocess.run(
                    f'squeue -u {os.environ["USER"]} -n {name}'.split(),
                    capture_output=True,
                )
                .stdout.decode("utf8")
                .splitlines()
            )
        else:
            sq = (
                subprocess.run(
                    f'squeue -u {os.environ["USER"]}'.split(), capture_output=True
                )
                .stdout.decode("utf8")
                .splitlines()
            )
        exit.wait(1)
    if not exit.is_set():
        print("\n Done!")


def slurm_file_creator(
    srun_command,
    n_nodes=1,
    n_tasks_per_node=1,
    n_gpu_per_node=0,
    module=None,
    name=None,
    singularity=None,
    time="02:00:00",
    qos=None,
    partition=None,
    constraint=None,
    cpus_per_task=None,
    exclusive=False,
    account=None,
    verbose=0,
    email=None,
    slurm_addon=None,
    script_addon=None,
):
    """
    IDRIS function
    Create Slurm File for sbatch execution
    Adapted for gpu job and cpu job, with module or singularity container.

    Example: sfile, stdout, stderr = slurm_file_creator('my_script.py --batch_size 128 -e 10 -lr .0001',
                                                        2, 4, 4, module=MODULE, name=name, time='10:00:00',
                                                        constraint='v100-32g', account=xxx@gpu, verbose=1,
                                                        script_addon="idrenv \necho $SLURM_SUBMIT_DIR")

    Args:
        srun_command (string): srun command to exec, if string begin by .py file add 'python -u' at the begining
                               ex: 'my_script.py --batch_size 128 -e 10 -lr .0001'
        n_nodes (int): number of nodes to set
        n_tasks_per_node (int): number of tasks per node to set
        n_gpu_per_node (int): number of gpu per node to set, default is 0 for CPU jobs
        module (string): module name to load
        name (string): name of job
        singularity (string): SIF image name
        time (string): max job time, default 02:00:00
        qos (string): gos to use, see idrdoc command to get the good qos names
        partition (string): partition to use, see idrdoc command to get the good partition names
        constraint (string): GPU constraint to use, see idrdoc command to get the good partition names
        cpus_per_task (int): number of CPUs per task to set, default is 1 for CPU job, 10 for gpu_p13, 3 for gpu_p2
        exclusive (bool): exclusive node jobs
        account (string): account, have to be set if user have several account
        verbose (int): 0 no debug add, 1 debug tracks are add
        email (string): send job state to 'email'
        slurm_addon (string): Add slurm option in header, for example '#SBATCH --reservation=toto'
        script_addon (string): Add some lines at the beginning of batch script, for example openMP export or unset proxy

    Return:
        slurm file path (string): path of slurm file.
    """

    os.makedirs("slurm", exist_ok=True)
    os.makedirs("slurm/log", exist_ok=True)

    if not name:
        name = srun_command.split(".")[0].split(" ")[-1]

    assert not " " in name, "Please don't use space character in name !!"

    if partition:
        assert (
            partition in ["gpu_p2", "gpu_p2l", "gpu_p2s", "gpu_p4"]
        ), "set partition args to gpu_p2, gpu_p2l, gpu_p2s or gpu_p4, if you want to use default partition (gpu_p13 for V100 or gpu_p5 for A100) just set nothing"

    A100_AMD = False
    if account and account.split("@")[-1] == "a100":
        A100_AMD = True

    if not cpus_per_task:
        if n_gpu_per_node:
            if A100_AMD:
                cpus_per_task = 8
            elif partition == "gpu_p4":
                cpus_per_task = 6
            elif partition:
                cpus_per_task = 3
            else:
                cpus_per_task = 10
        else:
            cpus_per_task = 1

    if singularity:
        assert (
            not module
        ), "If you use singularity container, please don't specify a module name !!"
        assert singularity[-4:] == ".sif", "SIF Image name has to finish by '.sif' !!"
        module = "singularity"

    assert module, "Please specify a module name, or use singularity"

    dtime = ".".join("_".join(str(datetime.now()).split(".")[0].split()).split(":"))

    with open("slurm/" + name + ".slurm", "w") as f:
        file = f"""#!/bin/bash
#SBATCH --job-name={name}
#SBATCH --output=slurm/log/{name}@JZ_%j_{n_nodes}nodes_{n_tasks_per_node}tasks_{dtime}.out 
#SBATCH --error=slurm/log/{name}@JZ_%j_{n_nodes}nodes_{n_tasks_per_node}tasks_{dtime}.err
"""
        if n_gpu_per_node > 0:
            file += f"#SBATCH --gres=gpu:{n_gpu_per_node}\n"
        file += f"""#SBATCH --nodes={n_nodes}
#SBATCH --ntasks-per-node={n_tasks_per_node}
#SBATCH --hint=nomultithread 
#SBATCH --time={time}
"""

        if qos:
            file += f"#SBATCH --qos={qos}\n"
        if exclusive:
            file += "#SBATCH --exclusive\n"

        file += f"#SBATCH --cpus-per-task={cpus_per_task}\n"

        if A100_AMD:
            constraint = "a100"
        if not partition and constraint:
            file += f"#SBATCH -C {constraint}\n"
        if partition:
            file += f"#SBATCH --partition={partition}\n"

        if account:
            file += f"#SBATCH --account={account}\n"

        if email:
            file += f"#SBATCH --mail-user={email}\n"

        if slurm_addon:
            file += slurm_addon + "\n"

        if verbose:
            file += """

export NCCL_INFO=DEBUG
export NCCL_DEBUG_SUBSYS=INIT,COLL
"""

        if module and A100_AMD:
            file += f"""

## load module 
module purge
module load cpuarch/amd
module load {module}
"""
        elif module:
            file += f"""

## load module 
module purge
module load {module}
"""

        if script_addon:
            file += "\n" + script_addon + "\n"

        if ".py" in srun_command.split()[0]:
            srun_command = "python -u " + srun_command

        if singularity:
            srun_command = (
                f"singularity exec --nv $SINGULARITY_ALLOWED_DIR/{singularity} "
                + srun_command
            )

        file += f"""

## launch script on every task 
set -x
time srun {srun_command}
date
"""
        f.write(file)

    return "slurm/" + name + ".slurm"


def gpu_jobs_submitter(
    srun_commands,
    n_gpu=1,
    module=None,
    singularity=None,
    name=None,
    n_gpu_per_task=1,
    time_max="02:00:00",
    qos=None,
    partition=None,
    constraint=None,
    cpus_per_task=None,
    exclusive=False,
    account=None,
    verbose=0,
    email=None,
    slurm_addon=None,
    script_addon=None,
):
    """
    IDRIS IA function
    Submit sbatch in queue for python jobs with distributed GPU use

    Example: paths = gpu_jobs_runner('python -u my_script.py --batch_size 128 -e 10 -lr .0001',
                                     n_gpu=[1, 2, 4, 8, 16, 32],
                                     module='pytorch-gpu/py3/1.7.1')

    Args:
        srun_commands: (string): srun command to exec, if string begin by .py file add 'python -u' at the begining
                               ex: 'my_script.py --batch_size 128 -e 10 -lr .0001',
                       or (string list) list of srun commands to exec, 1 job is launch by commands
        n_gpu: (int) number of distributed GPU to use, default is 1
               or (int list) list of number of distributed GPU to use, 1 job is launch by number of GPU
        module (string): module name to load
        singularity (string): SIF image name
        name (string): name of jobs
        n_gpu_per_task(int): number of GPUs associated per task,
                            if you use model parallelism or Tensorflow strategy for distribution,
                            you need more than 1 GPU per task.
        time_max (string): max job time, default 02:00:00
        qos (string): gos to use, see idrdoc command to get the good qos names
        partition (string): partition to use, see idrdoc command to get the good partition names
        constraint (string): GPU constraint to use, see idrdoc command to get the good partition names
        cpus_per_task (int): number of CPUs per task to set, default is 1 for CPU job, 10 for gpu_p13, 3 for gpu_p2
        exclusive (bool): exclusive node jobs
        account (string): account, have to be set if user have several account
        verbose (int): 0 no debug add, 1 debug tracks are add
        email (string): send job state to 'email'
        slurm_addon (string): Add slurm option in header, for example '#SBATCH --reservation=toto'
        script_addon (string): Add some lines at the beginning of the batch script, for example openMP export or unset proxy

    Return:
        jobids (list): list of submitted job ids
    """

    if type(n_gpu) == int:
        n_gpu = [n_gpu]
    if type(srun_commands) == str:
        srun_commands = [srun_commands]
    log_paths = {"stdout": [], "stderr": []}
    jobids = []

    A100_AMD = False
    if account and account.split("@")[-1] == "a100":
        A100_AMD = True
    node_capacity = 8 if partition or A100_AMD else 4

    for i, n in enumerate(n_gpu):
        assert (
            n // node_capacity == 0 or n % node_capacity == 0
        ), f"Over {node_capacity} GPUs, total GPU number has to be a multiple of {node_capacity} - You ask {n} GPUS !!"
        n_nodes = n // node_capacity if n // node_capacity > 0 else 1
        n_gpu_per_node = node_capacity if n // node_capacity > 0 else n

        assert (
            not n_gpu_per_node % n_gpu_per_task
        ), f"Number of GPUs have to be a multiple of n_gpu_per_task, you ask {n} GPUs and {n_gpu_per_task} GPUs per task !!"
        n_tasks_per_node = n_gpu_per_node // n_gpu_per_task

        if not cpus_per_task:
            if A100_AMD:
                cpus_per_task = 8 * n_gpu_per_task
            elif partition == "gpu_p4":
                cpus_per_task = 6 * n_gpu_per_task
            elif partition:
                cpus_per_task = 3 * n_gpu_per_task
            else:
                cpus_per_task = 10 * n_gpu_per_task

        print(
            f"batch job {i*len(srun_commands)}: {n} GPUs distributed on {n_nodes} nodes with {n_tasks_per_node} tasks / {n_gpu_per_node} gpus per node and {cpus_per_task} cpus per task"
        )

        for com in srun_commands:
            sfile = slurm_file_creator(
                com,
                n_nodes,
                n_tasks_per_node,
                n_gpu_per_node,
                module=module,
                name=name,
                time=time_max,
                qos=qos,
                partition=partition,
                constraint=constraint,
                exclusive=exclusive,
                account=account,
                verbose=verbose,
                email=email,
                cpus_per_task=cpus_per_task,
                singularity=singularity,
                slurm_addon=slurm_addon,
                script_addon=script_addon,
            )
            time.sleep(2)
            proc = subprocess.run(f"sbatch {sfile}".split(), capture_output=True)
            if proc.returncode == 0:
                for l in proc.stdout.decode("utf8").splitlines():
                    print(l)
                    jobids.append(l.split()[-1])
            else:
                for l in proc.stderr.decode("utf8").splitlines():
                    print(l)

    return jobids


def search_log(name="*", contains="", with_err=False):
    """
    IDRIS function
    Search and sort the log paths available in the log folder

    Example: paths = search_log('my_job', contains='2021-02-12_22:*1node)

    Args:
        name (string): Filter list by name, default is '*' (no filter)
        contains (string): Add filter after'@' by date or by number of tasks/nodes or jobid,
                           '*' for every string between 2 filters
        with_err (bool): if True return a dictionary with paths of out files and paths of err files,
                         if False return a list of out files only, default is False

    Return:
        paths: sorted list of .out files,
               or if with_err set, 2 keys dict -> paths['stdout']:sorted list of .out files,
                                              and paths['stderr']:sorted list of .err files.
    """

    if not with_err:
        paths = sorted(glob.glob(f"./slurm/log/{name}@*{contains}*.out"))

    else:
        paths = {}
        paths["stdout"] = sorted(glob.glob(f"./slurm/log/{name}@*{contains}*.out"))
        paths["stderr"] = sorted(glob.glob(f"./slurm/log/{name}@*{contains}*.err"))

    return paths
