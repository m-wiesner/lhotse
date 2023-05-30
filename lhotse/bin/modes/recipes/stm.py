from typing import Optional, Tuple, List, Sequence, Union

import click

from lhotse.bin.modes import prepare
from lhotse.recipes.mtedx import prepare_stm
from lhotse.utils import Pathlike

@prepare.command(context_settings=dict(show_default=True))
@click.argument("stms", help="comma separated list of stms")
@click.argument("output_dir", type=click.Path())
@click.option(
    "-l",
    "--langs",
    type=str,
    default=1,
    help="specifies the source-target language pairs",
)
def stm(
    stms: str,
    output_dir: Pathlike,
    langs: Optional[List[Tuple]],
):
    stm_files = stms.split(',')
    src_tgt_langs = [tuple(l.split('-')) for l in langs.split(',')]
    prepare_stm(stm_files, output_dir=output_idr, src_tgt_langs=src_tgt_langs)


