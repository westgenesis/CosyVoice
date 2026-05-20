"""
FastAPI service for CosyVoice2 TTS.

Supports three inference modes:
  - Pretrained voice (SFT)
  - Zero-shot voice cloning (3-second quick clone)
  - Instruct (natural language control)

Usage:
    uvicorn fastapi_app:app --host 0.0.0.0 --port 50000
"""

import os
import sys
import io
import tempfile

import numpy as np
import torch
import torchaudio
import librosa
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
import uvicorn
from contextlib import asynccontextmanager

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{ROOT_DIR}/third_party/Matcha-TTS')

# CosyVoice imports (must come after third_party path and model download)
# ---------------------------------------------------------------------------
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
from cosyvoice.utils.common import set_all_random_seed

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
max_val = 0.8
prompt_sr = 16000     # prompt audio is resampled to 16 kHz for feature extraction
target_sr = 24000     # final output sample rate

# Global model reference (initialised in lifespan)
cosyvoice: CosyVoice2 | None = None
sft_spk: list[str] = []


# ===================================================================
#  Lifespan
# ===================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global cosyvoice, sft_spk

    load_jit = os.environ.get('jit') == '1'
    load_trt = os.environ.get('trt') == '1'
    fp16 = os.environ.get('fp16') == '1'

    cosyvoice = CosyVoice2(
        f'pretrained_models/CosyVoice2-0.5B',
        load_jit=load_jit,
        load_trt=load_trt,
        fp16=fp16,
    )
    sft_spk = cosyvoice.list_available_spks()
    yield


app = FastAPI(title='CosyVoice2 TTS Service', lifespan=lifespan)


# ===================================================================
#  Helpers
# ===================================================================
def postprocess(speech: torch.Tensor,
                top_db: float = 60,
                hop_length: int = 220,
                win_length: int = 440) -> torch.Tensor:
    """Trim silence and normalise peak amplitude, then append 0.2 s silence."""
    speech, _ = librosa.effects.trim(
        speech, top_db=top_db,
        frame_length=win_length,
        hop_length=hop_length,
    )
    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val
    speech = torch.concat([speech, torch.zeros(1, int(target_sr * 0.2))], dim=1)
    return speech


def _save_upload(wav: UploadFile) -> str:
    """Save an uploaded WAV to a temporary file and return its path."""
    tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    tmp.write(wav.file.read())
    tmp.close()
    return tmp.name


def _validate_prompt_wav(wav_path: str):
    """Validate sample rate and duration of a prompt audio file."""
    info = torchaudio.info(wav_path)
    if info.sample_rate < prompt_sr:
        raise HTTPException(
            status_code=400,
            detail=f'Prompt audio sample rate {info.sample_rate} Hz is below '
                   f'minimum {prompt_sr} Hz',
        )
    duration = info.num_frames / info.sample_rate
    if duration > 10:
        raise HTTPException(
            status_code=400,
            detail=f'Prompt audio duration {duration:.1f}s exceeds maximum 10 s',
        )


def _build_wav_response(tensor: torch.Tensor) -> Response:
    """Convert a 1-D float tensor [1, T] into a WAV HTTP response."""
    # torchaudio.save wants [channels, time]; tensor is already [1, T]
    tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    tmp.close()
    try:
        torchaudio.save(tmp.name, tensor, target_sr)
        with open(tmp.name, 'rb') as f:
            content = f.read()
    finally:
        os.unlink(tmp.name)

    return Response(
        content=content,
        media_type='audio/wav',
        headers={'Content-Disposition': 'attachment; filename="output.wav"'},
    )


# ===================================================================
#  Endpoints
# ===================================================================

@app.get('/health')
async def health():
    """Health check."""
    return {'status': 'ok'}


@app.get('/voices')
async def list_voices():
    """List all available pretrained (SFT) voice names."""
    return {'voices': sft_spk}


# -------------------------------------------------------------------
#  1. Pretrained voice  (SFT)
# -------------------------------------------------------------------
@app.post('/tts/pretrained')
async def tts_pretrained(
    text: str = Form(..., description='Text to synthesise'),
    voice: str = Form(default='', description='Pretrained voice name'),
    seed: int = Form(default=0, ge=0, description='Random seed for reproducibility'),
    speed: float = Form(default=1.0, ge=0.5, le=2.0, description='Speaking speed'),
):
    """Synthesise speech using a built-in pretrained voice."""
    if cosyvoice is None:
        raise HTTPException(status_code=503, detail='Model not loaded')
    voice = voice or (sft_spk[0] if sft_spk else '')

    set_all_random_seed(seed)
    frames = [
        r['tts_speech']
        for r in cosyvoice.inference_sft(text, voice, stream=False, speed=speed)
    ]
    return _build_wav_response(torch.concat(frames, dim=1))


# -------------------------------------------------------------------
#  2. Zero-shot voice cloning  (3s Quick Clone)
# -------------------------------------------------------------------
@app.post('/tts/zero-shot')
async def tts_zero_shot(
    text: str = Form(..., description='Text to synthesise'),
    prompt_text: str = Form(..., description='Transcript of the prompt audio'),
    prompt_wav: UploadFile = File(..., description='Prompt audio file (≤10 s, ≥16 kHz)'),
    seed: int = Form(default=0, ge=0),
    speed: float = Form(default=1.0, ge=0.5, le=2.0),
):
    """Clone a voice from a short prompt audio and its transcript."""
    if cosyvoice is None:
        raise HTTPException(status_code=503, detail='Model not loaded')

    wav_path = _save_upload(prompt_wav)
    try:
        _validate_prompt_wav(wav_path)
        prompt_speech_16k = postprocess(load_wav(wav_path, prompt_sr))
    finally:
        os.unlink(wav_path)

    set_all_random_seed(seed)
    frames = [
        r['tts_speech']
        for r in cosyvoice.inference_zero_shot(
            text, prompt_text, prompt_speech_16k,
            stream=False, speed=speed,
        )
    ]
    return _build_wav_response(torch.concat(frames, dim=1))


# -------------------------------------------------------------------
#  3. Instruct (Natural Language Control)
# -------------------------------------------------------------------
@app.post('/tts/instruct')
async def tts_instruct(
    text: str = Form(..., description='Text to synthesise'),
    instruct_text: str = Form(..., description='Natural-language instruction (e.g. "speak in Sichuan dialect")'),
    prompt_wav: UploadFile = File(..., description='Prompt audio file (≤10 s, ≥16 kHz)'),
    seed: int = Form(default=0, ge=0),
    speed: float = Form(default=1.0, ge=0.5, le=2.0),
):
    """Synthesise speech with natural-language style control."""
    if cosyvoice is None:
        raise HTTPException(status_code=503, detail='Model not loaded')

    wav_path = _save_upload(prompt_wav)
    try:
        _validate_prompt_wav(wav_path)
        prompt_speech_16k = postprocess(load_wav(wav_path, prompt_sr))
    finally:
        os.unlink(wav_path)

    set_all_random_seed(seed)
    frames = [
        r['tts_speech']
        for r in cosyvoice.inference_instruct2(
            text, instruct_text, prompt_speech_16k,
            stream=False, speed=speed,
        )
    ]
    return _build_wav_response(torch.concat(frames, dim=1))


# ===================================================================
#  Entry point
# ===================================================================
if __name__ == '__main__':
    uvicorn.run('fastapi_app:app', host='0.0.0.0', port=50000, reload=False)
