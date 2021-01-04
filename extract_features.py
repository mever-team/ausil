import argparse
import numpy as np
import os
import librosa as lib
import subprocess as sp
from tqdm import tqdm
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset

from feature_extraction.network_architectures import weak_mxh64_1024
import feature_extraction.extractor as exm


usegpu = True
n_fft = 1024
hop_length = 512
n_mels = 128
trainType = 'weak_mxh64_1024'
pre_model_path = 'feature_extraction/mx-h64-1024_0d3-1.17.pkl'
featType = ['layer1', 'layer2', 'layer4', 'layer5', 'layer7', 'layer8', 'layer10', 'layer11', 'layer13', 'layer14', 'layer16', 'layer18'] # or layer 19 -  layer19 might not work well
globalpoolfn = F.max_pool2d # can use max also
netwrkgpl = F.avg_pool2d # keep it fixed


def load_model(netx,modpath):
    state_dict = torch.load(modpath, map_location=lambda storage, loc: storage)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module.' in k:
            name = k[7:]
        else:
            name = k
        new_state_dict[name] = v
    netx.load_state_dict(new_state_dict)


def ffmpeg_load_audio(filename, sr=44100, mono=True, normalize=True, in_type=np.int16, out_type=np.float32):
    channels = 1 if mono else 2
    format_strings = {
        np.float64: 'f64le',
        np.float32: 'f32le',
        np.int16: 's16le',
        np.int32: 's32le',
        np.uint32: 'u32le'
    }
    format_string = format_strings[in_type]

    command = [
        'ffmpeg',
        '-i', filename,
        '-f', format_string,
        '-acodec', 'pcm_' + format_string,
        '-ar', str(sr),
        '-ac', str(channels),
        '-']

    p = sp.Popen(command, stdout=sp.PIPE, stderr=sp.PIPE, bufsize=4096)
    # amy-uprint(filename)
    bytes_per_sample = np.dtype(in_type).itemsize
    frame_size = bytes_per_sample * channels
    chunk_size = frame_size * sr  # read in 1-second chunks
    raw = b''
    with p.stdout as stdout:
        while True:
            data = stdout.read(chunk_size)
            if data:
                raw += data
            else:
                break
    audio = np.frombuffer(raw, dtype=in_type).astype(out_type)
    if channels > 1:
        audio = audio.reshape((-1, channels)).transpose()
    if audio.size == 0:
        return audio, sr
    if issubclass(out_type, np.floating):
        if normalize:
            audio -= np.mean(audio)
            peak = np.abs(audio).max()
            if peak > 0:
                audio /= peak
        elif issubclass(in_type, np.integer):
            audio /= np.iinfo(in_type).max

    return audio, sr


def wav_load_audio(f):
    audio, fs = lib.load(f)
    audio = lib.resample(audio, fs, 44100)
    return audio


def extract_spectrogram(video_path):
    try:
        audio, sr = ffmpeg_load_audio(video_path)

        audio = lib.resample(audio, sr, 44100)
        audio = lib.util.normalize(audio)

        spectgram = lib.power_to_db(
            lib.feature.melspectrogram(y=audio, sr=44100, n_fft=n_fft, hop_length=hop_length, n_mels=128)).T
        spectgram = np.concatenate([spectgram, np.zeros((176 - spectgram.shape[0] % 176, 128))])  # zero padding

        # This part is for time step of 125 ms
        '''
        iterations = int(spectgram.shape[0]/11 -15)

        spectgram2 = np.zeros((iterations, 1, 176, 128))

        for i in range(iterations):
            spectgram2[i] = spectgram[i*11:i*11+176,:].reshape(1,176,128)
        '''
        
        spectgram = np.concatenate([spectgram, np.zeros((87 - spectgram.shape[0] % 87, 128))])  # zero padding
        spectgram = np.reshape(spectgram, (spectgram.shape[0] // 87, 1, 87, 128))  # shape needed from pytorch
        spectgram = np.concatenate([spectgram[:-1], spectgram[1:]], axis=2)  # 1 sec overlap

        return spectgram.astype(np.float32)
    except Exception as e:
        print(video_path)
        print(str(e))


class VideoDataset(Dataset):
    def __init__(self, videos_file):

        with open(videos_file) as f:
            videos = f.readlines()
        self.videos = [x.strip() for x in videos]

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video_id = self.videos[idx].split(' ')[0]
        video_path = self.videos[idx].split(' ')[1]
        video_spec = extract_spectrogram(video_path)
        if video_spec is None:
            video_spec = np.array([])

        return video_spec, video_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--videos_file', type=str, help='File containing the video ids and locations', required=True)
    parser.add_argument('-o', '--output_dir', type=str, help='Directory to save the extracted features', required=True)
    args = parser.parse_args()
    # Load model
    netx = weak_mxh64_1024(527, netwrkgpl)
    load_model(netx, pre_model_path)

    feat_extractor = torch.nn.DataParallel(exm.featExtractor(netx, featType))

    feat_extractor.to(0)
    feat_extractor.eval()

    dataset = VideoDataset(args.videos_file)
    loader = DataLoader(
                dataset,
                batch_size=1,
                num_workers=20,
                shuffle=False)

    pbar = tqdm(loader)
    for video in pbar:

        video_tensor = video[0][0]
        video_id = video[1][0]

        if video_tensor.shape[0] > 1:

            features = []
            for i in range(video_tensor.shape[0]//128 + 1):
                batch = video_tensor[i*128:(i+1)*128]
                if batch.shape[0] > 0:
                    features.append(feat_extractor(batch.to(0)).data.cpu().numpy())
            features = np.concatenate(features, axis=0)

            if not os.path.exists('{}/{}/'.format(args.output_dir, video_id)):
                os.makedirs('{}/{}/'.format(args.output_dir, video_id))
            np.savez_compressed('{}/{}/wlaf'.format(args.output_dir, video_id), features=features)


