import matplotlib.pyplot as plt
import os
import json
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from  scipy.io import wavfile 

import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence

from scipy.io.wavfile import write


def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

# stn_tst = get_text("this is a test of the tts system", hps)
# with torch.no_grad():
#     x_tst = stn_tst.unsqueeze(0)
#     x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
#     audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.float().numpy()
#     #ipd.display(ipd.Audio(audio, rate=hps.data.sampling_rate))


def create_context_response(text, save_path):
    hps = utils.get_hparams_from_file("./configs/ljs_base.json")

    hps_ms = utils.get_hparams_from_file("./configs/vctk_base.json")

    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model)
    _ = net_g.eval()

    net_g_ms = SynthesizerTrn(
        len(symbols),
        hps_ms.data.filter_length // 2 + 1,
        hps_ms.train.segment_size // hps.data.hop_length,
        n_speakers=hps_ms.data.n_speakers,
        **hps_ms.model)
    _ = net_g.eval()

    _ = utils.load_checkpoint("pretrained_vctk.pth", net_g_ms, None)


    _ = utils.load_checkpoint("pretrained_ljs.pth", net_g, None)

    # 16
    # 37 14 17 34 29 28 24 22 16 15 0 100 99
    sid = torch.LongTensor([16]) # speaker identity

    stn_tst = get_text(text, hps_ms)

    with torch.no_grad():
        x_tst = stn_tst.unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
        audio = net_g_ms.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.float().numpy()

    wavfile.write(save_path, hps.data.sampling_rate, audio) # <-- way to save to audio.

