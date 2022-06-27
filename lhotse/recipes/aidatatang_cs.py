"""
About the aidatatang_cs corpus

This corpus was used during the ASRU2019 codeswitching challenge. There are 4
parts to this corpus.

1. 500 hrs Monolingual Mandarin training set
2. 200 hrs Codeswitching Mandarin training set
3. ASRU TEST
4. ASRU DEV
"""

from pathlib import Path
from typing import Dict, Optional, Union
from lhotse.audio import Recording, RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet
import unicodedata
import re
from tqdm.auto import tqdm
import sys
from lhotse.utils import is_module_available

if sys.version_info >= (3, 10, 0):
    from itertools import pairwise
else:
    if not is_module_available("more_itertools"):
        raise ImportError(
            "more-itertools not found. Please install ... (pip install more-itertools)"
        )
    else:
        from more_itertools import pairwise 

from lhotse import (
    Recording,
    AudioSource,
    RecordingSet,
    SupervisionSegment,
    SupervisionSet,
    validate_recordings_and_supervisions,
)
from lhotse.qa import (
    remove_missing_recordings_and_supervisions,
    trim_supervisions_to_recordings,
)

from lhotse.utils import Pathlike, is_module_available, urlretrieve_progress
import soundfile as sf

VALID_CATEGORIES = ("Mc", "Mn", "Ll", "Lm", "Lo", "Lt", "Lu", "Nd", "Zs")


def prepare_aidatatang_cs(
   corpus_dir: Pathlike, output_dir: Optional[Pathlike] = None,
   parts: list = ["ZH+EN-200h", "ZH-500h", "ASRU-TEST", "ASRU-DEV"]
) -> Dict[str, Dict[str, Union[RecordingSet,SupervisionSet]]]:
    corpus_dir = Path(corpus_dir)
    manifests = {}
    for p in parts:
        print(p)
        recordings = {}
        supervisions = []
        part_dir = corpus_dir / p if p == "ASRU-TEST" else corpus_dir / p / "data"
        if p in ("ZH+EN-200h", "ZH-500h"):
            files = part_dir.glob("category*/*/*/*.txt")
        else:
            files = part_dir.glob("*.txt")
        for f in tqdm(list(files)):
            uttid = f.stem
            audio_sf = sf.SoundFile(str(f.with_suffix('.wav'))) 
            duration = audio_sf.frames / audio_sf.samplerate
            with open(f, 'r', encoding='utf-8') as fh:
                for l in fh:
                    text = prepare_text(l.strip())
                    spkid = uttid[5:10] 
                    sup = SupervisionSegment(
                        id=uttid,
                        recording_id=uttid,
                        start=0,
                        duration=round(duration, ndigits=8),
                        text=text,
                        speaker=spkid,
                    )
                    rec = Recording(
                        id=uttid,
                        sources=[
                            AudioSource(
                                type="file",
                                channels=[0],
                                source=str(f.with_suffix('.wav')),
                            ),
                        ],
                        sampling_rate=audio_sf.samplerate,
                        num_samples=audio_sf.frames,
                        duration=round(duration, ndigits=8),
                    )
                    supervisions.append(sup)
                    recordings[uttid] = rec
        supervisions = SupervisionSet.from_segments(supervisions)
        recordings = RecordingSet.from_recordings(recordings.values())
        recordings, supervisions = remove_missing_recordings_and_supervisions(
            recordings, supervisions,
        )
        supervisions = trim_supervisions_to_recordings(recordings, supervisions)
        validate_recordings_and_supervisions(recordings, supervisions)
        
        manifests[p] = {
            "recordings": recordings,
            "supervisions": supervisions,
        }
        if output_dir is not None:
            if isinstance(output_dir, str):
                output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            recordings.to_file(output_dir / f"recordings_{p}.json")
            supervisions.to_file(output_dir / f"supervisions_{p}.json")

    return manifests


def cond(s): 
    if (
     "\u2e80" <= s <= "\u2fd5" or
     "\u3190" <= s <= "\u319f" or
     "\u3400" <= s <= "\u4dbf" or
     "\u4e00" <= s <= "\u9fcc" or
     "\uf900" <= s <= "\ufaad"
    ):
        return True
    return False


def _map(pair):
    is_first_mandarin = cond(pair[0])
    is_second_mandarin = cond(pair[1])
    if is_first_mandarin and is_second_mandarin:
        return pair[0] + " "
    elif not is_first_mandarin and not is_second_mandarin:
        return pair[0].lower() if unicodedata.category(pair[0]) in VALID_CATEGORIES else ""
    elif not is_first_mandarin and is_second_mandarin:
        return pair[0].lower() + " " if unicodedata.category(pair[0]) in VALID_CATEGORIES else " "
    else:
        return ""


def prepare_text(line):
    # remove the [S] and [N] symbols for instance
    line = re.sub(r'\[[A-Z]\]', '', line)
    line_no_punct_space = ''.join(map(_map, pairwise(line + " "))).strip()
    return re.sub("  +", " ", line_no_punct_space)
