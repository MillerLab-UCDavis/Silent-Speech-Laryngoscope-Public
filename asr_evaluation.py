import os
import logging
import sys
import deepspeech
import jiwer
import soundfile as sf
import numpy as np
from unidecode import unidecode
import librosa
from read_emg import EMGDataset
from tensorboardX import SummaryWriter
from absl import flags
FLAGS = flags.FLAGS
writer = SummaryWriter()

def evaluate(testset, audio_directory):
    model = deepspeech.Model('deepspeech-0.7.0-models.pbmm')
    model.enableExternalScorer('deepspeech-0.7.0-models.scorer')
    predictions = []
    targets = []
    for i, datapoint in enumerate(testset):
        file_label = datapoint['file_label']
        filepath = os.path.join(audio_directory,f'{file_label}_audio.wav')
        # audio, rate = sf.read(filepath)
        # print(os.path.exists(filepath))
        if(not os.path.exists(filepath)):
            continue
        audio, rate = sf.read(filepath)
        if rate != 16000:
            audio = librosa.resample(audio, orig_sr=rate, target_sr=16000)
        assert model.sampleRate() == 16000, 'wrong sample rate'
        audio_int16 = (audio*(2**15)).astype(np.int16)
        text = model.stt(audio_int16)
        target_text = unidecode(datapoint['text'])
        if(len(text) != 0 and len(target_text) != 0):
            predictions.append(text)
            targets.append(target_text)
    transformation = jiwer.Compose([jiwer.RemovePunctuation(), jiwer.ToLowerCase()])
    targets = transformation(targets)
    predictions = transformation(predictions)
    print(f'targets: {targets}')
    print(f'predictions: {predictions}')
    print(f'wer: {jiwer.wer(targets, predictions)}')


if __name__ == '__main__':
    FLAGS(sys.argv)
    testset = EMGDataset(test=True)
    evaluate(testset, 'output_audios_closed_6/greedy_beam/silent/5-19/')
