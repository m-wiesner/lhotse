"""
The SEAME corpora of Singaporean Codeswitched English and Mandarin.

This corpus comes defined with a training split and two development splits:

train -- A mix of codeswitched, Mandarin and Singaporean English
dev_sge -- A set of primarily Singaporean English though there is codeswitching  
dev_man -- A set of primarily Mandarin though there is also some codeswitching

From these dev sets we separate the sentences with purely English, purely
Mandarin sentences, and mixes of the two to form new sets called:

dev_eng -- English only sentences (at least more so than dev_sge)
dev_cmn -- Mandarin only sentences (at least more so than dev_man)
dev_csw -- Codeswitched only sentences (at least in theory)

All audio files (found in audio in the directory shown in the directory tree
below), are sampled at 16kHz and stored in the .flac format.
 
The directory structure of the corpus is

/LDC2015S04/
├── data
│   ├── conversation
│   │   ├── audio
│   │   └── transcript
│   │       ├── phaseI
│   │       └── phaseII
│   └── interview
│       ├── audio
│       └── transcript
│           ├── phaseI
│           └── phaseII
├── docs
├── original
│   ├── data
│   │   ├── conversation
│   │   │   ├── audio
│   │   │   └── transcript
│   │   └── interview
│   │       ├── audio
│   │       └── transcript
│   └── docs
└── partitions
    ├── dev_man
    ├── dev_sge
    └── train
"""

import logging
import os
import shutil
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Union
import soundfile as sf

from lhotse import AudioSource, validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, urlretrieve_progress
from lhotse.qa import (
    remove_missing_recordings_and_supervisions,
    trim_supervisions_to_recordings,
)
from lhotse.utils import Pathlike
from lhotse.manipulation import combine


def prepare_seame(
    corpus_dir: Pathlike, output_dir: Optional[Pathlike] = None
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests of Recordings and Supervisions
    :param corpus_dir: Pathlike, the path of the data dir.
    :param output_dir: Pathlike, the path where to write the manifests.
    :return: a Dict whose key is the dataset part, and whose values are Dicts
        with keys 'recordings', and 'supervisions'
    """
    corpus_dir = Path(corpus_dir)
    recos = id2recos(corpus_dir)
    # For now we are just preparing the test sets. We actually have a kaldi
    # data directory for them. We parse the segments file and wav.scp to 
    # get the manifests
    dataset_parts = ["dev_man", "dev_sge"]
    manifests = {}
    for part in dataset_parts:
        lang = 'Mandarin' if part == 'dev_man' else 'English'
        supervisions, recordings = [], {}
        text_file = corpus_dir / 'partitions' / part / 'text'
        segments_file = corpus_dir / 'partitions' / part / 'segments'
        with open(text_file, 'r', encoding='utf-8') as f:
            text = load_text(f)
        
        with open(segments_file, 'r', encoding='utf-8') as f:
            for l in f:
                uttid, recoid, start, end = l.strip().split()
                spk = uttid.split('-')[0]
                duration = round(float(end) - float(start), ndigits=8)
                audio_file = recos[recoid] 
                supervisions.append(
                    SupervisionSegment(
                        id=uttid,
                        recording_id=recoid,
                        start=float(start),
                        duration=duration,
                        channel=0,
                        text=text[uttid],
                        language=lang,
                        speaker=spk,
                    )
                )
                if recoid not in recordings:
                    audio_sf = sf.SoundFile(str(recos[recoid]))
                    recordings[recoid] = Recording(
                        id=recoid,
                        sources=[
                            AudioSource(
                                type="file",
                                channels=[0],
                                source=str(recos[recoid]),
                            ),
                        ],
                        sampling_rate=audio_sf.samplerate,
                        num_samples=audio_sf.frames,
                        duration=audio_sf.frames / audio_sf.samplerate,
                    )
            supervisions = SupervisionSet.from_segments(supervisions)
            recordings = RecordingSet.from_recordings(recordings.values())
            recordings, supervisions = remove_missing_recordings_and_supervisions(
                recordings, supervisions,
            )
            supervisions = trim_supervisions_to_recordings(recordings, supervisions)
            validate_recordings_and_supervisions(recordings, supervisions)
            manifests[part] = {'recordings': recordings, 'supervisions': supervisions}
    
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            recordings.to_file(output_dir / f"recordings_{part}.json")
            supervisions.to_file(output_dir / f"supervisions_{part}.json")
    return manifests
                       

def load_text(f):
    text = {}
    for l in f:
        uttid, utt_text = l.strip().split(None, 1)
        text[uttid] = utt_text
    return text


def id2recos(path):
    recos = {}
    for p in path.glob("data/*/audio/*.flac"):
        recoid = p.stem.lower()
        recos[recoid] = p.resolve(strict=False)
    return recos
