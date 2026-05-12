import torch
from librosa.filters import mel as librosa_mel_fn

# cache mel filterbank and hann window per device+fmax
_mel_basis = {}
_hann_window = {}


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    """mel spectrogram with dynamic range compression, same as matcha.utils.audio.mel_spectrogram."""
    global _mel_basis, _hann_window
    key = f"{fmax}_{y.device}"
    if key not in _mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        _mel_basis[key] = torch.from_numpy(mel).float().to(y.device)
        _hann_window[key] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    ).squeeze(1)

    spec = torch.view_as_real(
        torch.stft(
            y,
            n_fft,
            hop_length=hop_size,
            win_length=win_size,
            window=_hann_window[key],
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
    )
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-9)
    spec = torch.matmul(_mel_basis[key], spec)
    spec = torch.log(torch.clamp(spec, min=1e-5))
    return spec
