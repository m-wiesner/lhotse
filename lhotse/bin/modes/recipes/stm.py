from typing import Optional, Tuple, List, Sequence, Union

import click

from lhotse.bin.modes import prepare
from lhotse.recipes.stm import prepare_stm
from lhotse.utils import Pathlike

#__all__ = ["stm"]

@prepare.command(context_settings=dict(show_default=True))
@click.argument("stms")
@click.argument("output_dir", type=click.Path())
@click.option(
    "-l",
    "--langs",
    type=str,
    default=1,
    help="The src-tgt language pairs in a comma-separated list"
)
def stm(
    stms: str,
    output_dir: Pathlike,
    langs: Optional[List[Tuple]],
):
    """STM data preparation"""
    stm_files = stms.split(',')
    src_tgt_langs = [tuple(l.split('-')) for l in langs.split(',')]
    prepare_stm(stm_files, output_dir=output_dir, src_tgt_langs=src_tgt_langs)


