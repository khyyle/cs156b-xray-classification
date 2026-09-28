from __future__ import annotations

import logging
import shlex
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from radiology_cls.settings import CLUSTER_ACCOUNT, CLUSTER_GPU_PARTITION
from radiology_cls.utils import PROJECT_ROOT

logger = logging.getLogger(__name__)

JOB_ENV_SCRIPT = PROJECT_ROOT / "scripts" / "env.sh"


@dataclass(frozen=True)
class JobResources:
    """
    Resources for a single-node, single-GPU job.

    Parameters:
    -----------
    gpu_type: str
        GPU model for `--gres` (e.g. `h100`).
    cpus: int
        CPUs per task.
    mem: str
        System RAM, in Slurm syntax (e.g. `64G`).
    time: str
        Wall time limit (`HH:MM:SS`).
    """
    gpu_type: str
    cpus: int
    mem: str
    time: str


def build_slurm_flags(job_name: str, resources: JobResources) -> list[str]:
    """
    Build the resource flags shared by `srun` and `#SBATCH` headers.

    Parameters:
    -----------
    job_name: str
        Name shown in `squeue`.
    resources: JobResources
        Resources to request.

    Returns:
    --------
    list[str]
        Flags such as `--gres=gpu:h100:1`.
    """
    return [
        f"--account={CLUSTER_ACCOUNT}",
        f"--partition={CLUSTER_GPU_PARTITION}",
        "--nodes=1",
        "--ntasks=1",
        f"--gres=gpu:{resources.gpu_type}:1",
        f"--cpus-per-task={resources.cpus}",
        f"--mem={resources.mem}",
        f"--time={resources.time}",
        f"--job-name={job_name}",
    ]


def _job_preamble() -> str:
    return f"cd {shlex.quote(str(PROJECT_ROOT))} && source {shlex.quote(str(JOB_ENV_SCRIPT))}"


def run_foreground(command: list[str], job_name: str, resources: JobResources) -> int:
    """
    Run `command` on a compute node via `srun`, streaming output to the terminal.

    Parameters:
    -----------
    command: list[str]
        Command and arguments to run from the project root.
    job_name: str
        Name shown in `squeue`.
    resources: JobResources
        Resources to request.

    Returns:
    --------
    int
        Exit code of the `srun` process.
    """
    srun_command = [
        "srun", *build_slurm_flags(job_name, resources),
        "bash", "-c", f"{_job_preamble()} && {shlex.join(command)}",
    ]
    return subprocess.run(srun_command).returncode


def submit_batch(
    command: list[str],
    job_name: str,
    resources: JobResources,
    log_path: Path,
    script_path: Path,
    email: str | None = None,
) -> str:
    """
    Write an sbatch script that runs `command` from the project root and submit it.

    Parameters:
    -----------
    command: list[str]
        Command and arguments to run on the compute node.
    job_name: str
        Name shown in `squeue`.
    resources: JobResources
        Resources to request.
    log_path: Path
        Where Slurm writes the job's stdout and stderr.
    script_path: Path
        Preferred location for the generated script. Falls back to a temp
        file when not writable (e.g. a run dir owned by a teammate).
    email: str | None, Default=None
        Address for begin/end/fail notifications.

    Returns:
    --------
    str
        sbatch's confirmation line, e.g. `Submitted batch job 12345`.

    Raises:
    -------
    RuntimeError
        If sbatch rejects the job.
    """
    header = [f"#SBATCH {flag}" for flag in build_slurm_flags(job_name, resources)]
    header.append(f"#SBATCH --output={log_path}")
    if email:
        header += [f"#SBATCH --mail-user={email}", "#SBATCH --mail-type=BEGIN,END,FAIL"]

    script = "\n".join([
        "#!/bin/bash",
        *header,
        "",
        f"{_job_preamble()} || exit 1",
        f"srun {shlex.join(command)}",
        "",
    ])
    written_path = _write_script(script, script_path)

    submission = subprocess.run(["sbatch", str(written_path)], capture_output=True, text=True)
    if submission.returncode != 0:
        raise RuntimeError(f"sbatch failed: {submission.stderr.strip()}")
    return submission.stdout.strip()


def _write_script(script: str, preferred_path: Path) -> Path:
    try:
        preferred_path.write_text(script)
        return preferred_path
    except PermissionError:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".sh", prefix=f"{preferred_path.stem}_", delete=False
        ) as handle:
            handle.write(script)
        logger.info("%s not writable; sbatch script written to %s", preferred_path, handle.name)
        return Path(handle.name)
