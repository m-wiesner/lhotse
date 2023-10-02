from typing import Optional, Tuple, List, Sequence, Union

import click

from lhotse.bin.modes import prepare
from lhotse.recipes.stm_parallel import prepare_stm_parallel
from lhotse.utils import Pathlike

__all__ = ["stm_parallel"]

@prepare.command(context_settings=dict(show_default=True))
@click.argument("stms")
@click.argument("output_dir", type=click.Path(dir_okay=True))
@click.option(
    "-l",
    "--langs",
    type=str,
    default=1,
    help="The src-tgt language pairs in a comma-separated list"
)
@click.option(
    "-n",
    "--name",
    type=str,
    default="stm",
    help="The dataset name that will be a prefix for the manifest filenames."
)
@click.option(
    "-j",
    "--num-jobs",
    type=int,
    default=4,
    help="The number of parallel threads to use for data preparation"
)
def stm_parallel(
    stms: str,
    output_dir: Pathlike,
    langs: Optional[List[Tuple]] = None,
    name: str = "stm",
    num_jobs: int = 4, 
):
    """STM data preparation"""
    stm_files = stms.split(',')
    src_tgt_langs = [tuple(l.split('-')) for l in langs.split(',')]
    prepare_stm_parallel(
        stm_files,
        output_dir=output_dir,
        src_tgt_langs=src_tgt_langs,
        name=name,
        num_jobs=num_jobs,
    )


