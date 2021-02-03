# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
import logging
import numpy as np
import sys

import torch
import torch.nn.functional as F

from .. import FairseqDataset

import copy

# Audiomentations
from audiomentations import (
    Compose, 
    AddGaussianNoise, 
    AddGaussianSNR, 
    AddShortNoises, 
    ClippingDistortion, 
    Gain, 
    PitchShift, 
    Shift, 
    TimeStretch,
    FrequencyMask,
    TimeMask
)

# WavAugment
import augment

logger = logging.getLogger(__name__)


class RawAudioDataset(FairseqDataset):
    def __init__(
        self,
        sample_rate,
        max_sample_size=None,
        min_sample_size=None,
        shuffle=True,
        min_length=0,
        pad=False,
        normalize=False,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.sizes = []
        self.max_sample_size = (
            max_sample_size if max_sample_size is not None else sys.maxsize
        )
        self.min_sample_size = min_sample_size
        self.min_length = min_length
        self.pad = pad
        self.shuffle = shuffle
        self.normalize = normalize

    def __getitem__(self, index):
        raise NotImplementedError()

    def __len__(self):
        return len(self.sizes)

    def postprocess(self, feats, curr_sample_rate):
        if feats.dim() == 2:
            feats = feats.mean(-1)

        if curr_sample_rate != self.sample_rate:
            raise Exception(f"sample rate: {curr_sample_rate}, need {self.sample_rate}")

        assert feats.dim() == 1, feats.dim()

        if self.normalize:
            with torch.no_grad():
                feats = F.layer_norm(feats, feats.shape)
        return feats

    def crop_to_max_size(self, wav, target_size):
        size = len(wav)
        diff = size - target_size
        if diff <= 0:
            return wav

        start = np.random.randint(0, diff + 1)
        end = size - diff + start
        return wav[start:end]

    def collater(self, samples):
        samples = [
            s
            for s in samples
            if s["source"] is not None
        ]
        if len(samples) == 0:
            return {}

        sources = [s["source"] for s in samples]
        sizes = [len(s) for s in sources]

        if self.pad:
            target_size = min(max(sizes), self.max_sample_size)
        else:
            target_size = min(min(sizes), self.max_sample_size)

        collated_sources = sources[0].new_zeros(len(sources), target_size)
        padding_mask = (
            torch.BoolTensor(collated_sources.shape).fill_(False) if self.pad else None
        )
        for i, (source, size) in enumerate(zip(sources, sizes)):
            diff = size - target_size
            if diff == 0:
                collated_sources[i] = source
            elif diff < 0:
                assert self.pad
                collated_sources[i] = torch.cat(
                    [source, source.new_full((-diff,), 0.0)]
                )
                padding_mask[i, diff:] = True
            else:
                collated_sources[i] = self.crop_to_max_size(source, target_size)

        input = {"source": collated_sources}
        if self.pad:
            input["padding_mask"] = padding_mask
        return {"id": torch.LongTensor([s["id"] for s in samples]), "net_input": input}

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        if self.pad:
            return self.sizes[index]
        return min(self.sizes[index], self.max_sample_size)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""

        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]

        order.append(self.sizes)
        return np.lexsort(order)[::-1]


class FileAudioDataset(RawAudioDataset):
    def __init__(
        self,
        manifest_path,
        sample_rate,
        max_sample_size=None,
        min_sample_size=None,
        shuffle=True,
        min_length=0,
        pad=False,
        normalize=False,
    ):
        super().__init__(
            sample_rate=sample_rate,
            max_sample_size=max_sample_size,
            min_sample_size=min_sample_size,
            shuffle=shuffle,
            min_length=min_length,
            pad=pad,
            normalize=normalize,
        )

        self.fnames = []

        skipped = 0
        with open(manifest_path, "r") as f:
            self.root_dir = f.readline().strip()
            for line in f:
                items = line.strip().split("\t")
                assert len(items) == 2, line
                sz = int(items[1])
                if min_length is not None and sz < min_length:
                    skipped += 1
                    continue
                self.fnames.append(items[0])
                self.sizes.append(sz)
        logger.info(f"loaded {len(self.fnames)}, skipped {skipped} samples")

    def __getitem__(self, index):
        import soundfile as sf

        fname = os.path.join(self.root_dir, self.fnames[index])
        wav, curr_sample_rate = sf.read(fname)
        #print("wav", wav.shape)
        feats = torch.from_numpy(wav).float()
        feats = self.postprocess(feats, curr_sample_rate)
        #print("postprocess", feats.size())
        return {"id": index, "source": feats}
    
    
class KDAudioDataset(RawAudioDataset):
    def __init__(
        self,
        manifest_path,
        sample_rate,
        max_sample_size=None,
        min_sample_size=None,
        shuffle=True,
        min_length=0,
        pad=False,
        normalize=False,
        feat_extension='flac',
    ):
        super().__init__(
            sample_rate=sample_rate,
            max_sample_size=max_sample_size,
            min_sample_size=min_sample_size,
            shuffle=shuffle,
            min_length=min_length,
            pad=pad,
            normalize=normalize,
        )

        self.fnames = []
        self.tnames = []

        skipped = 0
        with open(manifest_path, "r") as f:
            self.root_dir = f.readline().strip()
            for line in f:
                items = line.strip().split("\t")
                assert len(items) == 2, line
                sz = int(items[1])
                if min_length is not None and sz < min_length:
                    skipped += 1
                    continue
                self.fnames.append(items[0])
                self.tnames.append(items[0].replace('.'+feat_extension, '.ctcoutput'))
                self.sizes.append(sz)
        logger.info(f"loaded {len(self.fnames)}, skipped {skipped} samples")
        
    def crop_to_max_size(self, wav, target_size):
        size = len(wav)
        diff = size - target_size
        if diff <= 0:
            return wav

        start = np.random.randint(0, diff + 1)
        end = size - diff + start
        
        #if end-start != target_size:
        #    return wav[start:end-1]
        return wav[start:end]
        
    def collater(self, samples):
        samples = [
            s
            for s in samples
            if s["source"] is not None
        ]
        if len(samples) == 0:
            return {}

        sources = [s["source"] for s in samples]
        targets = [s["target"] for s in samples]
        sizes = [len(s) for s in sources]
        t_sizes = [s["target"].size(0) for s in samples]

        if self.pad:
            target_size = min(max(sizes), self.max_sample_size)
        else:
            target_size = min(min(sizes), self.max_sample_size)
            
        target_t_size = max(t_sizes)
        #print(target_t_size)
        target_dim = targets[0].size(-1)

        collated_sources = sources[0].new_zeros(len(sources), target_size)
        collated_targets = targets[0].new_zeros(len(targets), target_t_size, target_dim)
        padding_mask = (
            torch.BoolTensor(collated_sources.shape).fill_(False) if self.pad else None
        )
        for i, (source, target, size, t_size) in enumerate(zip(sources, targets, sizes, t_sizes)):
            diff = size - target_size
            t_diff = t_size - target_t_size
            if diff == 0:
                collated_sources[i] = source
            elif diff < 0:
                assert self.pad
                collated_sources[i] = torch.cat(
                    [source, source.new_full((-diff,), 0.0)]
                )
                padding_mask[i, diff:] = True
            else:
                collated_sources[i] = self.crop_to_max_size(source, target_size)
                
            #print(t_diff, t_size, target_t_size)
                
            try:
                if t_diff == 0:
                    collated_targets[i] = target
                elif t_diff < 0:
                    assert self.pad
                    
                    #print(target.size())
                    #print(target.new_full((-t_diff, target_dim), 0.0).size())
                    #print(torch.cat(
                    #    [target, target.new_full((-diff, target_dim), 0.0)]
                    #).size())
                    collated_targets[i] = torch.cat(
                        [target, target.new_full((-t_diff, target_dim), 0.0)]
                    )
                else:
                    collated_targets[i] = self.crop_to_max_size(target, target_t_size)[:target_t_size]
                    
            except:
                print(target_t_size, len(target))
                

        input = {"source": collated_sources}
        if self.pad:
            input["padding_mask"] = padding_mask
        else:
            input["padding_mask"] = None
        return {"id": torch.LongTensor([s["id"] for s in samples]), "net_input": input, "target": collated_targets}
        
    def __getitem__(self, index):
        import soundfile as sf

        fname = os.path.join(self.root_dir, self.fnames[index])
        wav, curr_sample_rate = sf.read(fname)
        #print("wav", wav.shape)
        feats = torch.from_numpy(wav).float()
        feats = self.postprocess(feats, curr_sample_rate)
        tname = os.path.join(self.root_dir, self.tnames[index])
        targets = torch.load(tname).squeeze()
        #print("postprocess", feats.size())
        return {"id": index, "source": feats, "target": targets}
    
    
class AugmentedFileAudioDataset(FileAudioDataset):
    def __init__(
        self,
        manifest_path,
        sample_rate,
        max_sample_size=None,
        min_sample_size=None,
        shuffle=True,
        min_length=0,
        pad=False,
        normalize=False,
    ):
        super(AugmentedFileAudioDataset, self).__init__(
            manifest_path=manifest_path,
            sample_rate=sample_rate,
            max_sample_size=max_sample_size,
            min_sample_size=min_sample_size,
            shuffle=shuffle,
            min_length=min_length,
            pad=pad,
            normalize=normalize,
        )
        
        self.pre_transform = Compose([
            #AddGaussianNoise(min_amplitude=1e-3, max_amplitude=5e-2, p=0.8),
            #PitchShift(min_semitones=-4, max_semitones=4, p=0.8),
            FrequencyMask(min_frequency_band=0.0, max_frequency_band=0.05, p=0.5),
            TimeMask(min_band_part=0.0, max_band_part=0.05, p=0.5)
            #ClippingDistortion(min_percentile_threshold=10, max_percentile_threshold=40, p=0.2),
        ])
        
        random_reverb = RandomReverb()
        random_clip = RandomClip()
        random_time_dropout = RandomTimeDropout()
        self.post_transform = augment.EffectChain().reverb(random_reverb).channels(1).clip(random_clip)#.time_dropout(200)
        
    def collater(self, samples):
        samples = [
            s
            for s in samples
            if s["source"] is not None
        ]
        if len(samples) == 0:
            return {}

        sources = [s["source"] for s in samples]
        sizes0 = [len(s[0]) for s in sources]
        sizes1 = [len(s[1]) for s in sources]
        sizes = sizes0 + sizes1
        
        if self.pad:
            target_size = min(max(sizes), self.max_sample_size)
        else:
            target_size = min(min(sizes), self.max_sample_size)

        collated_sources0 = sources[0][0].new_zeros(len(sources), target_size)
        collated_sources1 = sources[0][1].new_zeros(len(sources), target_size)
        padding_mask0 = (
            torch.BoolTensor(collated_sources0.shape).fill_(False) if self.pad else None
        )
        padding_mask1 = (
            torch.BoolTensor(collated_sources1.shape).fill_(False) if self.pad else None
        )
        for i, (source, size0, size1) in enumerate(zip(sources, sizes0, sizes1)):
            diff0 = size0 - target_size
            diff1 = size1 - target_size
            if diff0 == 0:
                collated_sources0[i] = source[0]
            elif diff0 < 0:
                assert self.pad
                collated_sources0[i] = torch.cat(
                    [source[0], source[0].new_full((-diff0,), 0.0)]
                )
                padding_mask0[i, diff0:] = True
            else:
                collated_sources0[i] = self.crop_to_max_size(source[0], target_size)
                
            if diff1 == 0:
                collated_sources1[i] = source[1]
            elif diff1 < 0:
                assert self.pad
                collated_sources1[i] = torch.cat(
                    [source[1], source[1].new_full((-diff1,), 0.0)]
                )
                padding_mask1[i, diff1:] = True
            else:
                collated_sources1[i] = self.crop_to_max_size(source[1], target_size)

        input = {"source": [collated_sources0, collated_sources1]}
        if self.pad:
            input["padding_mask"] = [padding_mask0, padding_mask1]
        return {"id": torch.LongTensor([s["id"] for s in samples]), "net_input": input}
        
    def __getitem__(self, index):
        import soundfile as sf

        fname = os.path.join(self.root_dir, self.fnames[index])
        wav0, curr_sample_rate = sf.read(fname)
        wav1 = copy.deepcopy(wav0)
        #print("wav", wav0.shape)
        wav0 = self.pre_transform(samples=wav0, sample_rate=curr_sample_rate)
        wav1 = self.pre_transform(samples=wav1, sample_rate=curr_sample_rate)
        src_info = {'rate': curr_sample_rate}
        tgt_info = {'channels': 1,
                    'rate': curr_sample_rate
                   }
        feats0 = torch.from_numpy(wav0).float()
        feats1 = torch.from_numpy(wav1).float()
        #print("preprocess", feats0.size())
        #print("preprocess", feats1.size())
        #feats0 = self.post_transform.apply(feats0, src_info=src_info, target_info=tgt_info).squeeze()
        #feats1 = self.post_transform.apply(feats1, src_info=src_info, target_info=tgt_info).squeeze()
        #print("postprocess", feats0.size())
        #print("postprocess", feats1.size())
        feats0 = self.postprocess(feats0, curr_sample_rate)
        feats1 = self.postprocess(feats1, curr_sample_rate)
        #print(feats0.size())
        return {"id": index, "source": [feats0, feats1]}
    
    
class RandomReverb:
    def __init__(self):
        self.reverberance_min: int = 0
        self.reverberance_max: int = 100
        self.damping_min: int = 0
        self.damping_max: int = 100
        self.room_scale_min: int = 0
        self.room_scale_max: int = 100
        self.p = 0.8

    def __call__(self):
        prob = np.random.random_sample()
        
        if prob < self.p:
            reverberance = np.random.randint(self.reverberance_min, self.reverberance_max + 1)
            damping = np.random.randint(self.damping_min, self.damping_max + 1)
            room_scale = np.random.randint(self.room_scale_min, self.room_scale_min + 1)
        else:
            reverberance = 0
            damping = 0
            room_scale = 0
        return [reverberance, damping, room_scale]

class RandomTimeDropout:
    def __init__(self):
        self.ms_min = 0
        self.ms_max = 200
        self.p = 0.5
    
    def __call__(self):
        prob = np.random.random_sample()
        
        if prob < self.p:
            ms = np.random.randint(self.ms_min, self.ms_max + 1)
        else:
            ms = 0
        
        return ms
    
class RandomClip:
    def __init__(self):
        self.factor_min = 0.0
        self.factor_max = 0.2
        self.p = 0.25
    
    def __call__(self):
        prob = np.random.random_sample()
        
        if prob < self.p:
            ratio = np.random.triangular(self.factor_min, self.factor_max, self.factor_max)
        else:
            ratio = 0.0
        return ratio