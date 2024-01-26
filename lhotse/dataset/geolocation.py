import math
import warnings
from typing import Any, Dict, Union, List, Optional, Tuple, Callable

import torch
from torch.utils.data import IterableDataset

from lhotse import RecordingSet, Seconds, compute_num_samples, validate
from lhotse.audio import suppress_audio_loading_errors
from lhotse.augmentation import AugmentFn
from lhotse.cut import CutSet
from lhotse.dataset.collation import collate_audio, collate_features, collate_matrices
from lhotse.dataset.input_strategies import BatchIO, PrecomputedFeatures, AudioSamples
from lhotse.features import FeatureExtractor
from lhotse.features.kaldi.layers import _get_strided_batch_streaming
from lhotse.workarounds import Hdf5MemoryIssueFix
from lhotse.utils import compute_num_frames, ifnone


class GeolocationDataset(torch.utils.data.Dataset):
    """
    Dataset that contains no supervision - it only provides the features extracted from recordings.

    .. code-block::

        {
            'features': (B x T x F) tensor
            'features_lens': (B, ) tensor
        }
    """

    def __init__(
        self, 
        return_cuts: bool = False,
        use_feats: bool = False,
        cut_transforms: List[Callable[[CutSet], CutSet]] = None,
        input_transforms: List[Callable[[torch.Tensor], torch.Tensor]] = None,
        input_strategy: BatchIO = AudioSamples(),
    ):
        super().__init__()
        self.return_cuts = return_cuts
        self.use_feats = use_feats
        self.cut_transforms = ifnone(cut_transforms, [])
        self.input_transforms = ifnone(input_transforms, [])
        self.input_strategy = input_strategy
        self.hdf5_fix = Hdf5MemoryIssueFix(reset_interval=50)

    def __getitem__(self, cuts: CutSet) -> Dict[str, Union[torch.Tensor, List[str]]]:
        self._validate(cuts)
        self.hdf5_fix.update()
        cuts = cuts.sort_by_duration(ascending=False)
        for tnfm in self.cut_transforms:
            cuts = tnfm(cuts)
        cuts = cuts.sort_by_duration(ascending=False)
        inputs, input_lens = collate_audio(cuts)
        # Get a tensor with batched feature matrices, shape (B, T, F)
        # Collation performs auto-padding, if necessary.
        supervision_intervals = self.input_strategy.supervision_intervals(cuts)
        
        segments = torch.stack(list(supervision_intervals.values()), dim=1)
        for tnfm in self.input_transforms:
            inputs = tnfm(inputs, supervision_segments=segments)

        if cuts[0].supervisions[0].custom is not None:
            targets = torch.Tensor(
                [
                    [s.custom['lat'], s.custom['lon']]
                    for c in cuts for s in c.supervisions
                ]
            )
        else:
            targets = None
      
        batch = {
            "supervisions": {
                "targets": targets,
                "language": [c.supervisions[0].language for c in cuts]
            },
            "ids": [c.id for c in cuts],
            "inputs": inputs,
            "features_lens": input_lens,
        }
        return batch

    def _validate(self, cuts: CutSet) -> None:
        validate(cuts)

