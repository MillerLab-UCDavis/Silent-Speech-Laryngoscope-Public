import re
import os
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import defaultdict
import scipy
import json
import copy
import sys
import pickle
import string
import logging
from functools import lru_cache
from PIL import Image
from torchvision.io import read_image
from torchvision.transforms.functional import convert_image_dtype
from copy import copy
from neural_synthesis.transforms import SpecAugment
from torchvision import transforms
from PIL import Image
from tsaug import AddNoise, TimeWarp

import librosa
import soundfile as sf

import torch

from data_utils import load_audio, get_emg_features, FeatureNormalizer, phoneme_inventory, read_phonemes, TextTransform

from absl import flags
FLAGS = flags.FLAGS
# Patient [11 17 13 16 12 14 15 10] => [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 18, 19]
# Patient without emg processing [ 5  9  8  6 14 15 12 18]
# Patient 2 with emg processing [ 3  5  9  4 14 13 18  2]
# Patient mapped to reference_speaker [2, 15, 1, 6]
# flags.DEFINE_list('remove_channels', [0, 1, 2, 3, 4, 7, 10, 11, 13, 16, 17, 19], 'channels to remove') # [ 7, 11,  0,  6, 10, 13,  1,  4], [16, 13, 14, 8, 2, 3, 19], [2, 3, 5, 7, 8, 9, 11, 12], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 18, 19]
flags.DEFINE_list('remove_channels_reference_speaker', [1, 2, 3, 4, 5, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19], 'channels to remove') # [ 7, 11,  0,  6, 10, 13,  1,  4], [16, 13, 14, 8, 2, 3, 19], [0, 1, 5, 6, 7, 8, 10, 11, 13, 14, 16, 19]
flags.DEFINE_list('remove_channels', [0, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19], 'channels to remove') # [ 7, 11,  0,  6, 10, 13,  1,  4], [16, 13, 14, 8, 2, 3, 19], [2, 3, 5, 7, 8, 9, 11, 12], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 18, 19]
# flags.DEFINE_list('remove_channels', [0, 1, 6, 7, 8, 10, 11, 12, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31], 'channels to remove') # [ 7, 11,  0,  6, 10, 13,  1,  4], [16, 13, 14, 8, 2, 3, 19], [2, 3, 5, 7, 8, 9, 11, 12], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 18, 19]
flags.DEFINE_list('silent_data_directories', ['./patient_data/closed_vocab/silent'], 'silent data locations')
flags.DEFINE_list('voiced_data_directories', ['./patient_data/closed_vocab/voiced'], 'voiced data locations')
# flags.DEFINE_list('silent_data_directories', ['./emg_data/closed_vocab/silent'], 'silent data locations')
# flags.DEFINE_list('voiced_data_directories', ['./emg_data/closed_vocab/voiced'], 'voiced data locations')
# flags.DEFINE_list('silent_data_directories', ['./emg_data/silent_parallel_data'], 'silent data locations')
# flags.DEFINE_list('voiced_data_directories', ['./emg_data/voiced_parallel_data', './emg_data/nonparallel_data'], 'voiced data locations')
flags.DEFINE_string('testset_file', 'reference_speaker_closed_vocab.json', 'file with testset indices')
flags.DEFINE_string('text_align_directory', 'text_alignments', 'directory with alignment files')

def remove_drift(signal, fs):
    b, a = scipy.signal.butter(3, 2, 'highpass', fs=fs)
    return scipy.signal.filtfilt(b, a, signal)

def notch(signal, freq, sample_frequency):
    b, a = scipy.signal.iirnotch(freq, 30, sample_frequency)
    return scipy.signal.filtfilt(b, a, signal)

def notch_harmonics(signal, freq, sample_frequency):
    for harmonic in range(1,8):
        signal = notch(signal, freq*harmonic, sample_frequency)
    return signal

def subsample(signal, new_freq, old_freq):
    times = np.arange(len(signal))/old_freq
    sample_times = np.arange(0, times[-1], 1/new_freq)
    result = np.interp(sample_times, times, signal)
    return result

def apply_to_all(function, signal_array, *args, **kwargs):
    results = []
    for i in range(signal_array.shape[1]):
        results.append(function(signal_array[:,i], *args, **kwargs))
    return np.stack(results, 1)


def load_video_frames(video_dir):
    # print(video_dir)
    frame_files = sorted(os.listdir(video_dir), key=lambda x: int(x.split('_')[1].split('.')[0]))  # Sort files by frame number
    frames = []
    for frame_file in frame_files:
        frame_path = os.path.join(video_dir, frame_file)
        frame = read_image(frame_path)  # Load image as a Torch tensor
        frame = convert_image_dtype(frame, dtype=torch.float)  # Convert image to float tensor (values between 0 and 1)
        frames.append(frame)
        # print(frame.shape)

    frames = torch.stack(frames, dim=0)  # Convert list of tensors to 4D tensor (T, C, H, W)
    return frames

def load_utterance(base_dir, index, limit_length=False, debug=False, text_align_directory=None, is_silent=None, augmenter=None):
    index = int(index)
    raw_emg = np.load(os.path.join(base_dir, f'{index}_emg.npy'))
    # before = os.path.join(base_dir, f'{index-1}_emg.npy')
    # after = os.path.join(base_dir, f'{index+1}_emg.npy')
    sess = int(base_dir[-1])
    if(sess == 0):
        raw_emg = raw_emg.T ## For patient data
    raw_emg = apply_to_all(subsample, raw_emg, 1000, 5000)

    # if os.path.exists(before):
    #     raw_emg_before = np.load(before)
    #     raw_emg_before = apply_to_all(subsample, raw_emg_before, 1000, 5000)
    # else:
    #     raw_emg_before = np.zeros([0,raw_emg.shape[1]])
    # if os.path.exists(after):
    #     raw_emg_after = np.load(after)
    #     raw_emg_after = apply_to_all(subsample, raw_emg_after, 1000, 5000)
    # else:
    #     raw_emg_after = np.zeros([0,raw_emg.shape[1]])

    # x = np.concatenate([raw_emg_before, raw_emg, raw_emg_after], 0)
    x = raw_emg
    x = apply_to_all(notch_harmonics, x, 60, 1000)
    x = apply_to_all(remove_drift, x, 1000)
    # x = x[raw_emg_before.shape[0]:x.shape[0]-raw_emg_after.shape[0],:]
    emg = x

    if(augmenter):
        for aug in augmenter:
            emg = aug.augment(emg)

    if emg.shape[1] == 20:
        if sess == 0:
            keep_channels = [i for i in range(emg.shape[1]) if i not in FLAGS.remove_channels]
        else:
            keep_channels = [i for i in range(emg.shape[1]) if i not in FLAGS.remove_channels_reference_speaker]
        emg = emg[:, keep_channels]
        # for c in FLAGS.remove_channels:
        #     emg[:, int(c)] = 0

    if is_silent is not None:
        # base_dir = is_silent
        if(sess == 0):
            audio = load_audio(os.path.join(is_silent, f'{index}_audio.flac'))
        else:
            audio = load_audio(os.path.join(is_silent, f'{index}_audio_clean.flac'))
    else:
        if(sess == 0):
            audio = load_audio(os.path.join(base_dir, f'{index}_audio.flac'))
        else:
            audio = load_audio(os.path.join(base_dir, f'{index}_audio_clean.flac'))


    with open(os.path.join(base_dir, f'{index}_info.txt'), 'r') as file:
        info = file.read()

    sess = os.path.basename(base_dir)
    tg_fname = f'{text_align_directory}/{sess}/{sess}_{index}_audio.TextGrid'

    if os.path.exists(tg_fname):
        phonemes = read_phonemes(tg_fname)
    else:
        phonemes = np.zeros(0)

    video_dir = os.path.join(base_dir, f'{index}_video')
    if os.path.isdir(video_dir):
        video_frames = load_video_frames(video_dir)
    else:
        video_frames = None  # or appropriate default value

    return audio, emg, info, phonemes, video_frames
    

class EMGDirectory(object):
    def __init__(self, session_index, directory, silent, exclude_from_testset=False):
        self.session_index = session_index
        self.directory = directory
        self.silent = silent
        self.exclude_from_testset = exclude_from_testset

    def __lt__(self, other):
        return self.session_index < other.session_index

    def __repr__(self):
        return self.directory

class SizeAwareSampler(torch.utils.data.Sampler):
    def __init__(self, emg_dataset, max_len):
        self.dataset = emg_dataset
        self.max_len = max_len

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        batch = []
        batch_length = 0
        for i, idx in enumerate(indices):
            # print(batch)
            
            directory_info, file_idx = self.dataset.example_indices[idx]
            with open(os.path.join(directory_info.directory, f'{file_idx}_info.txt')) as f:
                info = f.read()
            
            if not np.any([l in string.ascii_letters for l in info]):
                continue
            length = np.load(os.path.join(directory_info.directory, f'{file_idx}_emg.npy')).shape[1]
            # if length > self.max_len:
            #     logging.warning(f'Warning: example {idx} cannot fit within desired batch length')
            if length + batch_length > self.max_len or i == len(indices)-1:
                if(len(batch) != 0):
                    yield batch
                batch = []
                batch_length = 0
            batch.append(idx)
            batch_length += length
        # dropping last incomplete batch

class EMGDataset(torch.utils.data.Dataset):
    def __init__(self, base_dir=None, limit_length=False, dev=False, test=False, no_testset=False, no_normalizers=False):

        self.text_align_directory = FLAGS.text_align_directory


        if no_testset:
            # devset = []
            testset = []
        else:
            with open(FLAGS.testset_file) as f:
                testset_json = json.load(f)
                testset = testset_json['test']

        directories = []
        if base_dir is not None:
            directories.append(EMGDirectory(0, base_dir, False))
        else:
            # for sd in FLAGS.silent_data_directories:
            #     for session_dir in sorted(os.listdir(sd)):
            #         directories.append(EMGDirectory(len(directories), os.path.join(sd, session_dir), True))

            # has_silent = len(FLAGS.silent_data_directories) > 0
            for vd in FLAGS.voiced_data_directories:
                for session_dir in sorted(os.listdir(vd)):
                    directories.append(EMGDirectory(len(directories), os.path.join(vd, session_dir), False, exclude_from_testset=True))

            for sd in FLAGS.silent_data_directories:
                for session_dir in sorted(os.listdir(sd)):
                    # print(session_dir)
                    directories.append(EMGDirectory(len(directories), os.path.join(sd, session_dir), True))


        self.example_indices = []
        self.voiced_data_locations = {} # map from book/sentence_index to directory_info/index

        for directory_info in directories:
            sess = int(directory_info.directory[-1]) ## change this line to capture session index
            for fname in os.listdir(directory_info.directory):
                m = re.match(r'(\d+)_info.txt', fname)
                if m is not None:
                    trial = int(m.group(1))
                    location_in_testset = [sess, trial] in testset

                    # directory_info.silent and 
                    if directory_info.silent and ((sess, trial) in self.voiced_data_locations) and ((test and location_in_testset and not directory_info.exclude_from_testset) \
                        or (not test and not location_in_testset)):
                            # print(directory_info.silent)
                            # print(sess, trial)
                            self.example_indices.append((directory_info, trial))
                            

                    if not directory_info.silent:
                        location = (directory_info, trial)
                        # print(directory_info, trial)
                        self.voiced_data_locations[(sess, trial)] = location

        self.example_indices.sort()
        random.seed(0)
        random.shuffle(self.example_indices)

        self.no_normalizers = no_normalizers
        if not self.no_normalizers:
            self.mfcc_norm, self.emg_norm = pickle.load(open(FLAGS.normalizers_file,'rb'))
        
        # audio, emg, _, _ = load_utterance(voiced[0], voiced[1])
        self.limit_length = limit_length
        self.num_sessions = len(directories)

        self.text_transform = TextTransform()
        self.spec_augment = SpecAugment()
        self.test = test
        # self.my_augmenter = AddNoise(scale=0.3)
        self.noise_augmentor = AddNoise(scale=0.1)
        self.timewarp_augmentor = TimeWarp(n_speed_change=3, max_speed_ratio=2.0)

        
    def silent_subset(self):
        result = copy(self)
        silent_indices = []
        for example in self.example_indices:
            if example[0].silent:
                silent_indices.append(example)
        result.example_indices = silent_indices
        return result

    def subset(self, fraction):
        result = copy(self)
        result.example_indices = self.example_indices[:int(fraction*len(self.example_indices))]
        return result

    def __len__(self):
        return len(self.example_indices)

    @lru_cache(maxsize=None)
    def __getitem__(self, i):
        directory_info, idx = self.example_indices[i]

        if directory_info.silent:
            voiced_directory, _ = self.voiced_data_locations[(int(directory_info.directory[-1]), idx)]
            audio, emg, text, phonemes, video_frames = load_utterance(directory_info.directory, idx, self.limit_length, text_align_directory=self.text_align_directory, is_silent=voiced_directory.directory, augmenter=[self.timewarp_augmentor])
            
        else:
            audio, emg, text, phonemes, video_frames = load_utterance(directory_info.directory, idx, self.limit_length, text_align_directory=self.text_align_directory, is_silent=None, augmenter=[self.timewarp_augmentor])
        
        
        audio_file = f'{directory_info.directory}/{idx}_audio.flac'

        output_file_path = '/'.join(directory_info.directory.split('/')[-2:])
        result = {'text':text, 'file_label':idx, 'silent':directory_info.silent}


        result['phonemes'] = torch.from_numpy(phonemes).pin_memory() # either from this example if vocalized or aligned example if silent
        result['audio_file'] = audio_file
        result['audio'] = torch.from_numpy(audio).pin_memory()
        result['emg'] = torch.from_numpy(emg).pin_memory()
        result['output_file_path'] = output_file_path
        result['video_frames'] = video_frames
        result['idx'] = i

        return result

    @staticmethod
    def collate_raw(batch):

        phonemes = [ex['phonemes'] for ex in batch]
        emg = [ex['emg'] for ex in batch]
        lengths = [ex['emg'].shape[0] for ex in batch]
        audio_lengths = [ex['audio'].shape[0] for ex in batch]
        silent = [ex['silent'] for ex in batch]
        audio = [ex['audio'] for ex in batch]
        output_file_path = [ex['output_file_path'] for ex in batch]
        file_label = [ex['file_label'] for ex in batch]
        text = [ex['text'] for ex in batch]
        video_frames = [ex['video_frames'] for ex in batch]
        idx = [ex['idx'] for ex in batch]

        result = {'emg':emg,
                  'phonemes':phonemes,
                  'lengths':lengths,
                  'silent':silent,
                  'audio': audio,
                  'audio_lengths': audio_lengths,
                  'output_file_path': output_file_path,
                  'file_label': file_label,
                  'video_frames': video_frames,
                  'text': text,
                  'idx': idx}  
        
        return result

def make_normalizers():
    dataset = EMGDataset(no_normalizers=True)
    mfcc_samples = []
    emg_samples = []
    for d in dataset:
        mfcc_samples.append(d['audio_features'])
        emg_samples.append(d['emg'])
        if len(emg_samples) > 50:
            break
    mfcc_norm = FeatureNormalizer(mfcc_samples, share_scale=True)
    emg_norm = FeatureNormalizer(emg_samples, share_scale=False)
    pickle.dump((mfcc_norm, emg_norm), open(FLAGS.normalizers_file, 'wb'))