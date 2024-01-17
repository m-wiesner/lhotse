import json
import logging
from pathlib import Path
from typing import Dict, Optional, Union, NamedTuple, Tuple, List
from tqdm import tqdm
from lhotse.audio import Recording, RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike
import lhotse
import re

from functools import partial

from lhotse.parallel import parallel_map
from lhotse.audio import set_ffmpeg_torchaudio_info_enabled
import torchaudio
import torch


torchaudio.set_audio_backend("soundfile")
set_ffmpeg_torchaudio_info_enabled(False)


def _make_reco_and_sups_from_file(sf, msd=0.5):
    corpus_dir = sf.parents[2]
    audio_dir = corpus_dir / 'recos'
    fname = sf.with_suffix('.flac').stem
    # E.g. 2023_10_01_09h_02m_54s_dur30_ZnpbY9Zx_lat3.17_long113.04
    chunk_idx = int(sf.parent.suffix.strip('.'))
    reco_file = audio_dir / f'recos.{chunk_idx}' / f'{fname}.flac'
    reco = Recording.from_file(reco_file, recording_id=fname)
    reco.channel_ids = [0]
    sups = []
    total = 0
    with open(sf) as f:
        segments = json.load(f)
    lat, lon = re.search(r'lat[^_]+_long[^_]+', Path(sf).stem).group(0).split("_")
    lat = float(lat.replace('lat', ''))
    lon = float(lon.replace('long', ''))
    station = re.search(r's_dur[0-9]+_(.*)_lat[^_]+_long[^_]+', fname).groups()[0] #.split('_')
    fname_vals = fname.split('_')
    date = [int(i.strip('hms')) for i in fname_vals[0:6]] # YY MM DD hh mm ss
    for seg in segments:
        start, end = float(seg[1]), float(seg[2])
        dur = end - start
        if seg[0] in ('male', 'female') and dur > msd:
            sups.append(
                SupervisionSegment(
                    id=f'{fname}_{int(100*start):04}',
                    recording_id=fname,
                    start=start,
                    duration=round(dur, 4),
                    channel=0,
                    custom={
                        'date': date,
                        'lat': lat,
                        'lon': lon,
                        'station': station,
                        'est_gender': seg[0],
                    },
                )
            )
    return sups, reco


def prepare_radio(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    min_segment_duration: float = 0.5,
    num_jobs: int = 4,
) -> Dict[str, Union[RecordingSet, SupervisionSet]]:
    """
    Return the manifests which consist of recordings and supervisions
    :param corpus_dir: Path to the collected radio samples
    :param output_dir: Pathlike, the path where manifests are written
    :return: A Dict whose key is the dataset part and the value is a Dict with
        keys 'recordings' and 'supervisions'.  
    """
    corpus_dir = Path(corpus_dir)
    segment_files = corpus_dir.rglob('segs/*/*.json')
    supervisions, recordings = [], []
    fun = partial(_make_reco_and_sups_from_file, msd=min_segment_duration)
    output_dir = Path(output_dir) if output_dir is not None else None
    output_dir.mkdir(mode=511, parents=True, exist_ok=True)
    with RecordingSet.open_writer(output_dir / "radio_recordings.jsonl.gz") as rec_writer:
        with SupervisionSet.open_writer(output_dir / "radio_supervisions.jsonl.gz") as sup_writer:
            for sups, reco in tqdm(
                parallel_map(
                    fun,
                    segment_files,
                    num_jobs=num_jobs,
                ),
                desc=f'Making recordings and supervisions',
            ):
                rec_writer.write(reco)
                for sup in sups:
                    sup_writer.write(sup)

            manifests = {
                "recordings": RecordingSet.from_jsonl_lazy(rec_writer.path),
                "supervisions": SupervisionSet.from_jsonl_lazy(sup_writer.path),
            }

    return manifests
