import json
import logging
from pathlib import Path
from typing import Dict, Optional, Union, NamedTuple, Tuple, List
from tqdm import tqdm
from lhotse import validate_recordings_and_supervisions, fix_manifests
from lhotse.audio import Recording, RecordingSet, AudioSource, sph_info
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, Seconds, is_module_available
from itertools import groupby
import soundfile as sf


def stm_to_supervisions_and_recordings(fname, src=None, tgt=None):
    with open(str(fname), 'r', encoding='utf-8') as f:
        stm_entries = []
        for l in f:
            # Sometimes there are empty transcripts / translations.
            #
            try:
                wav, chn, spk, beg, end, lbl, txt = l.strip().split(None, 6)  
            except ValueError:
                wav, chn, spk, beg, end, lbl = l.strip().split(None, 5) 
                txt = ""

            # Retrieve channel information and convert A/B phone conversations to
            # integers. A --> 0, B --> 1 
            if chn == "A":
                chn = 0
            elif chn == "B":
                chn = 1
            else:
                chn = int(chn)
            beg, end = float(beg), float(end)
            stm_entries.append(
                {
                    'wav': wav,
                    'chn': chn,
                    'beg': beg,
                    'end': end,
                    'lbl': lbl,
                    'txt': txt,
                }
            )
    
    group_fun = lambda x: x['wav']
    recordings = RecordingSet.from_recordings(
        Recording.from_file(k) for k, g in groupby(stm_entries, group_fun):
    ) 
    
    supervisions = []
    for utt in stm_entries:
        recoid = Path(utt['wav']).stem
        beg = utt['beg']
        begstr = format(int(format(beg, '0.3f').replace('.', '')), '010d')
        if end - beg <= 0.01:
            raise ValueError(f"The duration of {utt} in stm file {fname} is too short")
        uttid = '_'.join([utt['spk'], str(utt['chn']), recoid, beg_str])
        supervisions.append(
            SupervisionSegment(
                id=uttid,
                recording_id=recoid,
                start=beg,
                duration=round(end - beg, 4),
                channel=utt['chn'],
                language=language,
                speaker=utt['spk'],
                text=utt['txt'],
                custom={'src_lang': src, 'tgt_lang': tgt},
            )
    supervision_set = SupervisionSet.from_segments(supervisions) 
    recording_set, supervision_set = fix_manifests(recording_set, supervision_set)
    validate_recordings_and_supervisions(recording_set, supervision_set)
    return recording_set, supervision_set


def pepare_stm(
    stm_files: list,
    output_dir: Optional[Pathlike] = None,
    src_tgt_langs: Optional[List[Tuple]] = None,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
   
    sups = SupervisionSet.from_segments([])
    recos = RecordingSet.from_recordings([])
    for i, stm_file in enumerate(stm_files):
        stm_file = Path(stm_file)
        if not stm_file.is_file():
            raise ValueError(f"{stm_file} does not exist")
        
        if src_tgt_langs is not None:
            src, tgt = src_tgt_langs[i]
        else:
            src, tgt = None, None
        sups_i, recos_i = stm_to_supervisions_and_recordings(
            stm_file,
            src=src,
            tgt=tgt
        )
        sups = sups + sups_i
        recos = recos + recos_i

    # Clean up supervisions and recordings
    recording_set, supervision_set = fix_manifests(recos, sups)
    validate_recordings_and_supervisions(recording_set, supervision_set)
    manifests = {
        "recordings": recordings,
        "supervisions": supervisions,
    }
    
    # Dump manifests to output dir if it is specified
    if output_dir is not None:
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        recordings.to_file(output_dir / f"recordings.jsonl.gz")
        supervisions.to_file(output_dir / f"supervisions.jsonl.gz")
    
    return dict(manifests)

    
