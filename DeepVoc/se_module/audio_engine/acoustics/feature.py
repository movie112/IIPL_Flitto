import os

import librosa
import math
import numpy as np
import torch
import torch.nn as nn


def stft(y, n_fft, hop_length, win_length):
    assert y.dim() == 2
    return torch.stft(
        y,
        n_fft,
        hop_length,
        win_length,
        window=torch.hann_window(n_fft).to(y.device),
        return_complex=True
    )


def istft(features, n_fft, hop_length, win_length, length=None, use_mag_phase=False):
    if use_mag_phase:
        # (mag, phase) or [mag, phase]
        assert isinstance(features, tuple) or isinstance(features, list)
        mag, phase = features
        features = torch.complex(mag * torch.cos(phase), mag * torch.sin(phase))
    else:
        features = torch.complex(features[:, :, :, 0], features[:, :, :, 1])

    return torch.istft(
        features,
        n_fft,
        hop_length,
        win_length,
        window=torch.hann_window(n_fft).to(features.device),
        length=length
    )


def mc_stft(y_s, n_fft, hop_length, win_length):

    assert y_s.dim() == 3
    batch_size, num_channels, num_wav_samples = y_s.size()

    stft_coefficients = torch.stft(
        y_s.reshape(batch_size * num_channels, num_wav_samples),
        n_fft=n_fft,
        hop_length=hop_length,
        window=torch.hann_window(win_length, device=y_s.device),
        win_length=win_length,
        return_complex=True
    )

    return stft_coefficients.reshape(batch_size, num_channels, stft_coefficients.shape[-2], stft_coefficients.shape[-1])


def mag_phase(complex_tensor):
    return torch.abs(complex_tensor), torch.angle(complex_tensor)


def norm_amplitude(y, scalar=None, eps=1e-6):
    if not scalar:
        scalar = np.max(np.abs(y)) + eps

    return y / scalar, scalar


def tailor_dB_FS(y, target_dB_FS=-25, eps=1e-6):
    rms = np.sqrt(np.mean(y ** 2))
    scalar = 10 ** (target_dB_FS / 20) / (rms + eps)
    y *= scalar
    return y, rms, scalar


def is_clipped(y, clipping_threshold=0.999):
    return any(np.abs(y) > clipping_threshold)


def load_wav(file, sr=16000):
    if len(file) == 2:
        return file[-1]
    else:
        return librosa.load(os.path.abspath(os.path.expanduser(file)), mono=False, sr=sr)[0]


def aligned_subsample(data_a, data_b, sub_sample_length):
    
    assert data_a.shape[-1] == data_b.shape[-1], "Inconsistent dataset size."

    if data_a.shape[-1] > sub_sample_length:
        length = data_a.shape[-1]
        start = np.random.randint(length - sub_sample_length + 1)
        end = start + sub_sample_length
        return data_a[..., start:end], data_b[..., start:end]
    elif data_a.shape[-1] < sub_sample_length:
        length = data_a.shape[-1]
        pad_size = sub_sample_length - length
        pad_width = [(0, 0)] * (data_a.ndim - 1) + [(0, pad_size)]
        data_a = np.pad(data_a, pad_width=pad_width, mode="constant", constant_values=0)
        data_b = np.pad(data_b, pad_width=pad_width, mode="constant", constant_values=0)
        return data_a, data_b
    else:
        return data_a, data_b


def subsample(data, sub_sample_length, start_position: int = -1, return_start_position=False):

    assert np.ndim(data) == 1, f"Only support 1D data. The dim is {np.ndim(data)}"
    length = len(data)

    if length > sub_sample_length:
        if start_position < 0:
            start_position = np.random.randint(length - sub_sample_length)
        end = start_position + sub_sample_length
        data = data[start_position:end]
    elif length < sub_sample_length:
        data = np.append(data, np.zeros(sub_sample_length - length, dtype=np.float32))
    else:
        pass

    assert len(data) == sub_sample_length

    if return_start_position:
        return data, start_position
    else:
        return data


def overlap_cat(chunk_list, dim=-1):

    overlap_output = []
    for i, chunk in enumerate(chunk_list):
        first_half, last_half = torch.split(chunk, chunk.size(-1) // 2, dim=dim)
        if i == 0:
            overlap_output += [first_half, last_half]
        else:
            overlap_output[-1] = (overlap_output[-1] + first_half) / 2
            overlap_output.append(last_half)

    overlap_output = torch.cat(overlap_output, dim=dim)
    return overlap_output


def activity_detector(audio, fs=16000, activity_threshold=0.13, target_level=-25, eps=1e-6):

    audio, _, _ = tailor_dB_FS(audio, target_level)
    window_size = 50  # ms
    window_samples = int(fs * window_size / 1000)
    sample_start = 0
    cnt = 0
    prev_energy_prob = 0
    active_frames = 0

    a = -1
    b = 0.2
    alpha_rel = 0.05
    alpha_att = 0.8

    while sample_start < len(audio):
        sample_end = min(sample_start + window_samples, len(audio))
        audio_win = audio[sample_start:sample_end]
        frame_rms = 20 * np.log10(sum(audio_win ** 2) + eps)
        frame_energy_prob = 1. / (1 + np.exp(-(a + b * frame_rms)))

        if frame_energy_prob > prev_energy_prob:
            smoothed_energy_prob = frame_energy_prob * alpha_att + prev_energy_prob * (1 - alpha_att)
        else:
            smoothed_energy_prob = frame_energy_prob * alpha_rel + prev_energy_prob * (1 - alpha_rel)

        if smoothed_energy_prob > activity_threshold:
            active_frames += 1
        prev_energy_prob = frame_energy_prob
        sample_start += window_samples
        cnt += 1

    perc_active = active_frames / cnt
    return perc_active


def drop_band(input, num_groups=2):
    batch_size, _, num_freqs, _ = input.shape
    assert batch_size > num_groups, f"Batch size = {batch_size}, num_groups = {num_groups}. The batch size should larger than the num_groups."

    if num_groups <= 1:
        return input

    if num_freqs % num_groups != 0:
        input = input[..., :(num_freqs - (num_freqs % num_groups)), :]
        num_freqs = input.shape[2]

    output = []
    for group_idx in range(num_groups):
        samples_indices = torch.arange(group_idx, batch_size, num_groups, device=input.device)
        freqs_indices = torch.arange(group_idx, num_freqs, num_groups, device=input.device)

        selected_samples = torch.index_select(input, dim=0, index=samples_indices)
        selected = torch.index_select(selected_samples, dim=2, index=freqs_indices)  # [B, C, F // num_groups, T]

        output.append(selected)

    return torch.cat(output, dim=0)


def init_stft_kernel(frame_len,
                     frame_hop,
                     num_fft=None,
                     window="sqrt_hann"):
    if window != "sqrt_hann":
        raise RuntimeError("Now only support sqrt hanning window in order "
                           "to make signal perfectly reconstructed")
    if not num_fft:
        fft_size = 2 ** math.ceil(math.log2(frame_len))
    else:
        fft_size = num_fft
    window = torch.hann_window(frame_len) ** 0.5
    S_ = 0.5 * (fft_size * fft_size / frame_hop) ** 0.5
    kernel = torch.rfft(torch.eye(fft_size) / S_, 1)[:frame_len]
    kernel = torch.transpose(kernel, 0, 2) * window
    kernel = torch.reshape(kernel, (fft_size + 2, 1, frame_len))
    return kernel


class CustomSTFTBase(nn.Module):

    def __init__(self,
                 frame_len,
                 frame_hop,
                 window="sqrt_hann",
                 num_fft=None):
        super(CustomSTFTBase, self).__init__()
        K = init_stft_kernel(
            frame_len,
            frame_hop,
            num_fft=num_fft,
            window=window)
        self.K = nn.Parameter(K, requires_grad=False)
        self.stride = frame_hop
        self.window = window

    def freeze(self):
        self.K.requires_grad = False

    def unfreeze(self):
        self.K.requires_grad = True

    def check_nan(self):
        num_nan = torch.sum(torch.isnan(self.K))
        if num_nan:
            raise RuntimeError(
                "detect nan in STFT kernels: {:d}".format(num_nan))

    def extra_repr(self):
        return "window={0}, stride={1}, requires_grad={2}, kernel_size={3[0]}x{3[2]}".format(
            self.window, self.stride, self.K.requires_grad, self.K.shape)


class CustomSTFT(CustomSTFTBase):

    def __init__(self, *args, **kwargs):
        super(CustomSTFT, self).__init__(*args, **kwargs)

    def forward(self, x):

        if x.dim() not in [2, 3]:
            raise RuntimeError("Expect 2D/3D tensor, but got {:d}D".format(
                x.dim()))
        self.check_nan()
        if x.dim() == 2:
            x = torch.unsqueeze(x, 1)
        c = torch.nn.functional.conv1d(x, self.K, stride=self.stride, padding=0)
        r, i = torch.chunk(c, 2, dim=1)
        m = (r ** 2 + i ** 2) ** 0.5
        p = torch.atan2(i, r)
        return m, p, r, i


class CustomISTFT(CustomSTFTBase):

    def __init__(self, *args, **kwargs):
        super(CustomISTFT, self).__init__(*args, **kwargs)

    def forward(self, m, p, squeeze=False):

        if p.dim() != m.dim() or p.dim() not in [2, 3]:
            raise RuntimeError("Expect 2D/3D tensor, but got {:d}D".format(
                p.dim()))
        self.check_nan()
        if p.dim() == 2:
            p = torch.unsqueeze(p, 0)
            m = torch.unsqueeze(m, 0)
        r = m * torch.cos(p)
        i = m * torch.sin(p)
        c = torch.cat([r, i], dim=1)
        s = torch.nn.functional.conv_transpose1d(c, self.K, stride=self.stride, padding=0)
        if squeeze:
            s = torch.squeeze(s)
        return s


class ChannelWiseLayerNorm(nn.LayerNorm):

    def __init__(self, *args, **kwargs):
        super(ChannelWiseLayerNorm, self).__init__(*args, **kwargs)

    def forward(self, x):

        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(
                self.__name__))
        x = torch.transpose(x, 1, 2)
        x = super(ChannelWiseLayerNorm, self).forward(x)
        x = torch.transpose(x, 1, 2)
        return x


class DirectionalFeatureComputer(nn.Module):
    def __init__(
            self,
            n_fft,
            win_length,
            hop_length,
            input_features,
            mic_pairs,
            lps_channel,
            use_cos_IPD=True,
            use_sin_IPD=False,
            eps=1e-8
    ):
        super().__init__()
        self.eps = eps
        self.input_features = input_features

        self.stft = CustomSTFT(frame_len=win_length, frame_hop=hop_length, num_fft=n_fft)
        self.num_freqs = n_fft // 2 + 1

        self.mic_pairs = np.array(mic_pairs)
        self.num_mic_pairs = self.mic_pairs.shape[0]
        self.ipd_left = [t[0] for t in mic_pairs]
        self.ipd_right = [t[1] for t in mic_pairs]
        self.use_cos_IPD = use_cos_IPD
        self.use_sin_IPD = use_sin_IPD

        self.lps_channel = lps_channel

        self.directional_feature_dim = 0
        if 'LPS' in self.input_features:
            self.directional_feature_dim += self.num_freqs
            self.lps_layer_norm = ChannelWiseLayerNorm(self.num_freqs)

        if 'IPD' in self.input_features:
            self.directional_feature_dim += self.num_freqs * self.num_mic_pairs
            if self.use_sin_IPD:
                self.directional_feature_dim += self.num_freqs * self.num_mic_pairs

    def compute_ipd(self, phase):

        cos_ipd = torch.cos(phase[:, self.ipd_left] - phase[:, self.ipd_right])
        sin_ipd = torch.sin(phase[:, self.ipd_left] - phase[:, self.ipd_right])
        return cos_ipd, sin_ipd

    def forward(self, y):

        batch_size, num_channels, num_samples = y.shape
        y = y.view(-1, num_samples)
        magnitude, phase, real, imag = self.stft(y)
        _, num_freqs, num_frames = phase.shape

        magnitude = magnitude.view(batch_size, num_channels, num_freqs, num_frames)
        phase = phase.view(batch_size, num_channels, num_freqs, num_frames)
        real = real.view(batch_size, num_channels, num_freqs, num_frames)
        imag = imag.view(batch_size, num_channels, num_freqs, num_frames)

        directional_feature = []
        if "LPS" in self.input_features:
            lps = torch.log(magnitude[:, self.lps_channel, ...] ** 2 + self.eps)
            lps = self.lps_layer_norm(lps)
            directional_feature.append(lps)

        if "IPD" in self.input_features:
            cos_ipd, sin_ipd = self.compute_ipd(phase)
            cos_ipd = cos_ipd.view(batch_size, -1, num_frames)
            sin_ipd = sin_ipd.view(batch_size, -1, num_frames)
            directional_feature.append(cos_ipd)
            if self.use_sin_IPD:
                directional_feature.append(sin_ipd)

        directional_feature = torch.cat(directional_feature, dim=1)

        return directional_feature, magnitude, phase, real, imag


class ChannelDirectionalFeatureComputer(nn.Module):
    def __init__(
            self,
            n_fft,
            win_length,
            hop_length,
            input_features,
            mic_pairs,
            lps_channel,
            use_cos_IPD=True,
            use_sin_IPD=False,
            eps=1e-8
    ):
        super().__init__()
        self.eps = eps
        self.input_features = input_features

        self.stft = CustomSTFT(frame_len=win_length, frame_hop=hop_length, num_fft=n_fft)
        self.num_freqs = n_fft // 2 + 1

        self.mic_pairs = np.array(mic_pairs)
        self.num_mic_pairs = self.mic_pairs.shape[0]
        self.ipd_left = [t[0] for t in mic_pairs]
        self.ipd_right = [t[1] for t in mic_pairs]
        self.use_cos_IPD = use_cos_IPD
        self.use_sin_IPD = use_sin_IPD

        self.lps_channel = lps_channel

        self.directional_feature_dim = 0
        if 'LPS' in self.input_features:
            self.directional_feature_dim += 1

        if 'IPD' in self.input_features:
            self.directional_feature_dim += self.num_mic_pairs
            if self.use_sin_IPD:
                self.directional_feature_dim += self.num_mic_pairs

    def compute_ipd(self, phase):

        cos_ipd = torch.cos(phase[:, self.ipd_left] - phase[:, self.ipd_right])
        sin_ipd = torch.sin(phase[:, self.ipd_left] - phase[:, self.ipd_right])
        return cos_ipd, sin_ipd

    def forward(self, y):

        batch_size, num_channels, num_samples = y.shape
        y = y.view(-1, num_samples)
        magnitude, phase, real, imag = self.stft(y)
        _, num_freqs, num_frames = phase.shape

        magnitude = magnitude.view(batch_size, num_channels, num_freqs, num_frames)
        phase = phase.view(batch_size, num_channels, num_freqs, num_frames)
        real = real.view(batch_size, num_channels, num_freqs, num_frames)
        imag = imag.view(batch_size, num_channels, num_freqs, num_frames)

        directional_feature = []
        if "LPS" in self.input_features:
            lps = torch.log(magnitude[:, self.lps_channel, ...] ** 2 + self.eps)
            lps = lps[:, None, ...]
            directional_feature.append(lps)

        if "IPD" in self.input_features:
            cos_ipd, sin_ipd = self.compute_ipd(phase)
            directional_feature.append(cos_ipd)

            if self.use_sin_IPD:
                directional_feature.append(sin_ipd)

        directional_feature = torch.cat(directional_feature, dim=1)

        return directional_feature, magnitude, phase, real, imag


if __name__ == '__main__':
    ipt = torch.rand(70, 1, 257, 200)
    print(drop_band(ipt, 8).shape)
