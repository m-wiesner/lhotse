"""
South African corpus of Multilingual speech. (From Soap Operas)

Described in https://aclanthology.org/L18-1451.pdf

"""
import logging
import re
from collections import defaultdict
from functools import partial
import glob
from lxml import etree

from pathlib import Path
from typing import Dict, Optional, Sequence, Union
import unicodedata
import zipfile

from tqdm.auto import tqdm

from lhotse import (
    Recording,
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

# Keep Markings such as vowel signs, all letters, and decimal numbers
VALID_CATEGORIES = ("Mc", "Mn", "Ll", "Lm", "Lo", "Lt", "Lu", "Nd", "Zs")
KEEP_LIST = ["\u2019"]


###############################################################################
#                             Download and Untar
###############################################################################

lang_shortname = {
    "xhosa": "xho",
    "zulu": "zul",
    "sesotho": "sot",
    "setswana": "tsn",
}

def download_soapies(
    target_dir: Pathlike = ".",
    languages: Optional[Union[str, Sequence[str]]] = "all",
) -> Path:

    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    # Download the splits.
    completed_detector = target_dir / ".splits.completed"
    if completed_detector.is_file():
        logging.info(f"Skipping downloading splits because {completed_detector} exists.")
    else:
        zip_path = target_dir / "splits.zip"
        urlretrieve_progress(
            f"https://repo.sadilar.org/bitstream/handle/20.500.12185/545/soapies_dev_and_test_set_utterance_ids.zip",
            filename=zip_path,
            desc=f"Downloading soapies splits",
        )

        with zipfile.ZipFile(zip_path, "r") as f:
            f.extractall(target_dir / "splits")

        completed_detector.touch()


    langs_list = list(lang_shortname.keys())
    # If for some reason languages = None, assume this also means 'all'
    if isinstance(languages, str) and languages != "all":
        langs_list = [languages]
    elif isinstance(languages, list) or isinstance(languages, tuple):
        langs_list = languages

    for lang in tqdm(langs_list, "Downloading soapy languages"):
        zip_path = target_dir / f"{lang}.zip"
        xml_path = target_dir / f"{lang}.xml"
        completed_detector = target_dir / f".{lang}.completed"
        if completed_detector.is_file():
            logging.info(f"Skipping {lang} because {completed_detector} exists.")
            continue

        urlretrieve_progress(
            f"https://repo.sadilar.org/bitstream/handle/20.500.12185/545/balanced_eng{lang_shortname[lang]}.zip",
            filename=zip_path,
            desc=f"Downloading soapies/{lang} (zip)",
        )

        with zipfile.ZipFile(zip_path, "r") as f:
            f.extractall(target_dir / f"{lang}")

        urlretrieve_progress(
            f"https://repo.sadilar.org/bitstream/handle/20.500.12185/545/balanced_eng{lang_shortname[lang]}.xml",
            filename=xml_path,
            desc=f"Downloading soapies/{lang} (xml)",
        )
        completed_detector.touch()

    return target_dir


def prepare_soapies(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = Path("./"),
    languages: Optional[Union[str, Sequence[str]]] = "all",
) -> Dict[str, Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]]:

    # Resolve corpus_dir type
    if isinstance(corpus_dir, str):
        corpus_dir = Path(corpus_dir)

    # Resolve output_dir type
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)

    langs_list = list(lang_shortname.keys())
    # If for some reason languages = None, assume this also means 'all'
    if isinstance(languages, str) and languages != "all":
        langs_list = [languages]
    elif isinstance(languages, list) or isinstance(languages, tuple):
        if languages[0] != "all":
            langs_list = languages

    manifests = defaultdict(dict)
    for lang in langs_list:
        corpus_dir_lang = corpus_dir
        output_dir_lang = output_dir / f"{lang}"
        if corpus_dir_lang.is_dir():
            manifests[lang] = prepare_single_soapy_language(
                corpus_dir_lang,
                output_dir_lang,
                language=lang,
            )

    return dict(manifests)


###############################################################################
# All remaining functions are just helper functions, mainly for text
# normalization and parsing the vtt files that come with the mtedx corpus
###############################################################################

# Prepare data for a single language
def prepare_single_soapy_language(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    language: str = "language",
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:

    if isinstance(corpus_dir, str):
        corpus_dir = Path(corpus_dir)
    manifests = defaultdict(dict)

    wavdir = corpus_dir / language / "audio"
    uttids = set()
    for fname in wavdir.glob("*wav"):
        uttids.add(fname.stem)

    sname = lang_shortname[language]
    dev_list = corpus_dir / "splits" / f"cs_eng{sname}_balanced" / "transcriptions" / f"eng{sname}_dev_set_utterance_ids.txt"
    dev_uttids = set()
    with open(dev_list, "r") as f:
        for line in f:
            uttid = line.strip()
            if uttid in uttids:
                dev_uttids.add(uttid)

    sname = lang_shortname[language]
    test_list = corpus_dir / "splits" / f"cs_eng{sname}_balanced" / "transcriptions" / f"eng{sname}_tst_set_utterance_ids.txt"
    test_uttids = set()
    with open(test_list, "r") as f:
        for line in f:
            uttid = line.strip()
            if uttid in uttids:
                test_uttids.add(uttid)

    train_uttids = set()
    for uttid in uttids:
        if uttid in dev_uttids  or uttid in test_uttids:
            continue
        train_uttids.add(uttid)

    manifests = defaultdict(dict)

    xml_file = corpus_dir / f"{language}.xml"
    audio_dir = corpus_dir / language / "audio"
    with open(xml_file, "r") as f:
        tree = etree.parse(f)
        for split, uttids in zip(("train", "dev", "test"),
                                 (train_uttids, dev_uttids, test_uttids)):
            recordings = []
            segments = []
            for utterance in tree.xpath("//utterance"):
                speaker = utterance.xpath('./speaker_id/text()')[0].replace(' ', '_')
                wav = utterance.xpath('./audio/text()')[0]
                text = ' '.join(utterance.xpath('.//transcription/text()')).upper().replace('!', "'")
                uttid = Path(wav).stem
                if uttid in uttids:
                    recordings.append(Recording.from_file(str(audio_dir / wav)))
                    segments.append(
                        SupervisionSegment(
                            id=uttid,
                            recording_id=uttid,
                            start=0,
                            duration=round(recordings[-1].duration, ndigits=8),
                            channel=0,
                            text=text,
                            language=language,
                            speaker=speaker,
                        )
                    )
            recordings = RecordingSet.from_recordings(recordings)
            supervisions = SupervisionSet.from_segments(segments)

            if len(recordings) == 0:
                logging.warning(f"No .wav files found in {audio_dir}")


            recordings, supervisions = remove_missing_recordings_and_supervisions(
                recordings, supervisions
            )
            supervisions = trim_supervisions_to_recordings(recordings, supervisions)
            validate_recordings_and_supervisions(recordings, supervisions)

            manifests[split] = {
                "recordings": recordings,
                "supervisions": supervisions,
            }

            if output_dir is not None:
                if isinstance(output_dir, str):
                    output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                recordings.to_file(
                    output_dir / f"soapies-{language}_recordings_{split}.jsonl.gz"
                )
                supervisions.to_file(
                    output_dir / f"soapies-{language}_supervisions_{split}.jsonl.gz"
                )

    return dict(manifests)

