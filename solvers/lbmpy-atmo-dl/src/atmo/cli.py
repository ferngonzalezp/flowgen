import json
import yaml
import rich_click as click
from rich import print
from pathlib import Path

from .run_dir_tools import (
    JOB_LOG_FILE,
    JobDirectory,
    find_job_directories,
    find_latest_job_directory,
    find_job_chain,
    join_temporals,
)


click.rich_click.USE_RICH_MARKUP = True


@click.group()
def cli():
    pass


@cli.command()
def jobs():
    """View all jobs in current run directory, with start / end iteration"""
    jobdirs = sorted(find_job_directories(), key=lambda job: job.iteration_interval[0])
    if not jobdirs:
        warn("No jobs found in current directory")
        return
    info("Start it -   End it :: R :: Nsol :: Name")
    info("---------------------------------------------------------")
    messages = []
    for job in jobdirs:
        start, end = job.iteration_interval
        res = "Y" if job.is_restartable else " "
        msg = (
            f"{start: >8} - {'ø' if start == end else end: >8} :: {res} :: "
            f"{len(job.solutions): >4} :: {job.name}"
        )
        messages.append((end, msg))
    for _, msg in sorted(messages, key=lambda pair: pair[0], reverse=True):
        info(msg)


@cli.command()
@click.option(
    "-j",
    "--jobname",
    type=click.Path(),
    help="Unique job name to start from (default = latest)",
)
def jobchain(jobname):
    """View job chain leading up to latest / target"""
    jobdir = (
        JobDirectory(Path(jobname))
        if jobname is not None
        else find_latest_job_directory(Path.cwd())
    )
    if jobdir is None:
        warn("No job found")
        return

    chain = find_job_chain(jobdir.path)
    if len(chain) == 1:
        warn(f"No chain found from {jobdir.path.name}")
        return

    info("Job chain (most recent first)")
    info(chain[0].name)
    for j in chain[1:]:
        info("  ^ " + j.name)


@cli.command()
@click.option(
    "-r",
    "--restartable/--no-restartable",
    default=False,
    help="Only look for restartable files",
)
def latestjob(restartable):
    """View latest (restartable) job [bold]in terms of iteration advancement[/]"""
    latest = find_latest_job_directory(Path.cwd(), restartable=restartable)
    if latest:
        info(
            f"Latest{' restartable' if restartable else ''} "
            f"job directory:  {latest.name}"
        )
    else:
        info(f"[red]No{' restartable' if restartable else ''} job directory found[/]")


@cli.command()
@click.option(
    "-j",
    "--jobname",
    type=click.Path(),
    help="Unique job name to start from (default = latest)",
)
@click.option(
    "-o",
    "--overwrite/--no-overwrite",
    default=False,
    help="Overwrite if already exists",
)
def jointemp(jobname, overwrite):
    """Join temporals"""
    jobdir = (
        JobDirectory(Path(jobname))
        if jobname is not None
        else find_latest_job_directory(Path.cwd())
    )
    if jobdir is None:
        warn("No job found")
        return

    info("Joining files")
    joined_dict = join_temporals(jobdir.path.name)
    for joined in sorted(joined_dict):
        path = Path.cwd() / joined
        if path.is_file() and not overwrite:
            warn(f"    {joined} already exists. Use --overwrite to force write")
        else:
            info(f"    {joined}")
            with open(path, "w") as fh:
                fh.write(joined_dict[joined])


@cli.command()
@click.option(
    "-j",
    "--json-file",
    type=click.Path(),
    help="JSON serialized log file to read",
)
@click.option(
    "-l",
    "--as-regular-log/--no-as-regular-log",
    default=False,
    help="View as if from non-serialized log",
)
def json_log(json_file, as_regular_log):
    """View JSON serialized log more easily"""
    if json_file is None:
        json_file = Path.cwd() / JOB_LOG_FILE

    if not json_file.is_file():
        warn(f"{json_file} doesn't exist")
        return

    log = [json.loads(line) for line in open(json_file, "r")]
    for line in log:
        if as_regular_log:
            # msg = (
            #    f"{line['time']:YYYY-MM-DD HH:mm:ss.SSS} | "
            #    f"{line['level']: <8} | "
            #    f"{line['name']}:{line['function']}:{line['line']} | "
            #    f"{line['message']}"
            # )
            # info(msg)
            info(line["text"].strip())
        else:
            info(yaml.dump(line))


def info(msg):
    print(f"[colour(0)]{msg}[/]")


def warn(msg):
    print(f"[red bold]{msg}[/]")
