import json
import logging
from pathlib import Path
from typing import Dict, Optional, Union, NamedTuple, Tuple, List
from tqdm import tqdm
from lhotse import validate_recordings_and_supervisions, fix_manifests
from lhotse.audio import Recording, RecordingSet, AudioSource, sph_info
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, Seconds, is_module_available, fastcopy
from itertools import groupby
import soundfile as sf
import lhotse

from lhotse.audio import set_ffmpeg_torchaudio_info_enabled
import torchaudio
import torch

torchaudio.set_audio_backend("soundfile")
set_ffmpeg_torchaudio_info_enabled(False)

# Torch's multithreaded behavior needs to be disabled or
# it wastes a lot of CPU and slow things down.
# Do this outside of main() in case it needs to take effect
# even when we are not invoking the main (e.g. when spawning subprocesses).


def stm_to_supervisions_and_recordings(fname, src=None, tgt=None, permissive=True):
    with open(str(fname), 'r', encoding='utf-8') as f:
        stm_entries = []
        for l in tqdm(f, desc="Parsing the stm ..."):
            # Sometimes there are empty transcripts / translations.
            #
            try:
                wav, chn, spk, beg, end, lbl, txt = l.strip().split(None, 6)  
            except ValueError:
                wav, chn, spk, beg, end, lbl = l.strip().split(None, 5) 
                txt = ""
                assert len(lbl.split()) == 1
            
            # Retrieve channel information and convert A/B phone conversations to
            # integers. A --> 0, B --> 1 
            if chn == "A":
                chn = 0
            elif chn == "B":
                chn = 1
            else:
                chn = int(chn)
            beg, end = float(beg), float(end)
            if end - beg < 0.01: 
                print(f"The utterance is too short")
                print(f"{l}")
                continue;
            stm_entries.append(
                {
                    'wav': wav,
                    'chn': chn,
                    'spk': spk,
                    'beg': beg,
                    'end': end,
                    'lbl': lbl,
                    'txt': txt,
                }
            )
   
    group_fun = lambda x: (x['wav'], x['chn'])
    group_iter = list(groupby(sorted(stm_entries, key=group_fun), group_fun))
    recordings = []
    for k, g in tqdm(group_iter, desc=f"Making recordings from {fname}"):
        recording_id = str(Path(k[0]).with_suffix("")).replace("/", "_")[1:] + f"_{k[1]}"
        reco = Recording.from_file(k[0], recording_id=recording_id)
        # We want to treat these as mono recordings (not multicut recordings).
        # So, we create an extra recording as if there were extra audiofiles, for
        # each channel.
        for s in reco.sources:
            s.channels = [k[1]]
            reco.channel_ids = [k[1]]
            #reco_ = fastcopy(reco)
            #reco_.sources[s].channels = [k[1]]
            #reco_.channel_ids = [k[1]]
            #recordings.append(reco_)
        recordings.append(reco)
    recording_set = RecordingSet.from_recordings(recordings) 

    supervisions = []
    utts = set()
    for utt in stm_entries:
        recoid = str(Path(utt['wav']).with_suffix("")).replace("/", "_")[1:] + f"_{utt['chn']}"
        beg, end = utt['beg'], utt['end']
        beg_str = format(int(format(beg, '0.3f').replace('.', '')), '010d')
        duration = end - beg
        if duration <= 0.01:
            if permissive:
                print(f"The duration of {utt} in stm file {fname} is too short") 
                continue
            else:
                raise ValueError(f"The duration of {utt} in stm file {fname} is too short")
        
        uttid = '_'.join([utt['spk'], str(utt['chn']), recoid, tgt, beg_str])
        if uttid in utts:
            if permissive:
                print(f"Duplicate utterance id {uttid}")
                continue
            else:
                raise ValueError(f"Detected duplicate utterance id {uttid}")
        utts.add(uttid)
        supervisions.append(
            SupervisionSegment(
                id=uttid,
                recording_id=recoid,
                start=beg,
                duration=round(end - beg, 4),
                channel=utt['chn'],
                language=tgt,
                speaker=utt['spk'],
                text=utt['txt'],
                custom={'src_lang': src, 'tgt_lang': tgt},
            )
        )
    supervision_set = SupervisionSet.from_segments(supervisions)
    recording_set, supervision_set = fix_manifests(recording_set, supervision_set)
    validate_recordings_and_supervisions(recording_set, supervision_set)
    return recording_set, supervision_set


def prepare_stm(
    stm_files: list,
    output_dir: Optional[Pathlike] = None,
    src_tgt_langs: Optional[List[Tuple]] = None,
    permissive: bool = True,
    name: str = "stm",
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
   
    sups = SupervisionSet.from_segments([])
    recos = RecordingSet.from_recordings([])
    recoids = set()
    for i, stm_file in tqdm(enumerate(stm_files), desc="Making supervisions"):
        stm_file = Path(stm_file)
        if not stm_file.is_file():
            raise ValueError(f"{stm_file} does not exist")
        
        if src_tgt_langs is not None:
            src, tgt = src_tgt_langs[i]
        else:
            src, tgt = None, None
        recos_i, sups_i = stm_to_supervisions_and_recordings(
            stm_file,
            src=src,
            tgt=tgt
        )
        sups = sups + sups_i
        recos = recos + recos_i.filter(lambda r: r.id not in recoids)
        for r in recos_i:
            recoids.add(r.id)

    # Clean up supervisions and recordings
    recording_set, supervision_set = fix_manifests(recos, sups)
    validate_recordings_and_supervisions(recording_set, supervision_set)
    manifests = {
        "recordings": recording_set,
        "supervisions": supervision_set,
    }
    
    # Dump manifests to output dir if it is specified
    if output_dir is not None:
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        recording_set.to_file(output_dir / f"recordings_{name}.jsonl.gz")
        supervision_set.to_file(output_dir / f"supervisions_{name}.jsonl.gz")
    
    return dict(manifests)

    
