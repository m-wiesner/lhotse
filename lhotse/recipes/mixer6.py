import json
import logging
from pathlib import Path
from typing import Dict, Optional, Union, NamedTuple, Tuple
from tqdm import tqdm
from lhotse import validate_recordings_and_supervisions, fix_manifests
from lhotse.audio import Recording, RecordingSet, AudioSource, sph_info
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, Seconds, is_module_available
import soundfile as sf

dataset_parts = [
    "train_intv",
    "train_call",
    "dev_a",
    "dev_b",
    "test",
]

def prepare_mixer6(
    corpus_dir: Pathlike,
    transcript_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    part: str = "train_intv",
    channels: list = list(range(13)),
    multicut: bool = True,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    corpus_dir = Path(corpus_dir)
    transcript_dir = Path(transcript_dir)

    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"
    assert part in dataset_parts

    # Get session ids
    session_id_fname = transcript_dir / "splits" / f"{part}.list"
    sessions = {}
    with open(str(session_id_fname), 'r', encoding='utf-8') as f:
        for l in f:
            session_id, speakers, room = l.strip().split('\t')
            subj, interviewer = speakers.split(',')
            sessions[session_id] = {'subj': subj, 'intv': interviewer, 'room': room}

    # Prepare recordings
    pcm_files = corpus_dir / "data" / "pcm_flac"
    recordings = []
    
    def chnid2name(i):
        return f"CH{format(i+1, '02d')}"

    for session_id in tqdm(sessions):
        paths = [(i, pcm_files / chnid2name(i) / f"{session_id}_{chnid2name(i)}.flac") for i in channels] 
        audio_sf = sf.SoundFile(paths[0][1])
        reco = Recording(
            id=session_id,
            sources = [
                AudioSource(type="file", channels=[c], source=str(p)) for c, p in paths if p.is_file()
            ],
            sampling_rate=int(audio_sf.samplerate),
            num_samples=int(audio_sf.frames),
            duration=float(audio_sf.frames) / audio_sf.samplerate,
        )
        recordings.append(reco)
    
    # Prepare supervisions
    transcript_dir = transcript_dir / "splits" / part
    transcripts = transcript_dir.rglob("*.json") 
    supervisions = []
    for fname in tqdm(list(transcripts)):
        session_id = fname.stem
        with open(str(fname), 'r', encoding='utf-8') as f:
            transcript = json.load(f)
        for seg in transcript: 
            spkid, text = seg['speaker'], seg['words']
            start, end = float(seg['start_time']), float(seg['end_time'])
            begstr = format(int(format(start, '0.3f').replace('.', '')), '08d')
            if end - start <= 0.01:
                raise ValueError("The duration is too short or negative. This "
                    "indicates a problem in the data")
            for chn in channels:
                filepath = pcm_files / chnid2name(chn) / f"{session_id}_{chnid2name(chn)}.flac"
                if filepath.is_file():
                    supervisions.append(
                        SupervisionSegment(
                            id=f"{spkid}-{session_id}-{chnid2name(chn)}-{begstr}",
                            recording_id=session_id,
                            start=start,
                            duration=round(end - start, 4),
                            channel=chn,
                            language="English",
                            speaker=spkid,
                            text=text,
                        )
                    )
    
    recording_set = RecordingSet.from_recordings(recordings)
    supervision_set = SupervisionSet.from_segments(supervisions)
    recording_set, supervision_set = fix_manifests(recording_set, supervision_set)
    validate_recordings_and_supervisions(recording_set, supervision_set)

    if output_dir is not None:
        output_dir = Path(output_dir)
        recording_set.to_file(str(output_dir / f"recordings_{part}.jsonl.gz"))
        supervision_set.to_file(str(output_dir / f"supervisions_{part}.jsonl.gz"))

    manifest = {"recordings": recording_set, "supervisions": supervision_set}
    return manifest
