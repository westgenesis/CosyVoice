import functools
import tempfile
import os
import sys
import argparse
import io
import json
import uuid
import random
from contextlib import asynccontextmanager
from typing import Optional, List, Set

import numpy as np
import torch
import torchaudio

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn

# ===================== 路径 =====================
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT_DIR, "third_party", "Matcha-TTS"))

from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.file_utils import logging
from cosyvoice.utils.common import set_all_random_seed

# ===================== 常量 =====================
PROMPT_SR = 16000
CLONED_SPKS_FILE = "cloned_spks.json"

# ===================== 全局模型 =====================
cosyvoice: Optional[AutoModel] = None
sft_spk: List[str] = []
cloned_spks: Set[str] = set()
_model_dir: str = ""


# ===================== 启动时加载模型 =====================

def _load_cloned_spks() -> Set[str]:
    path = os.path.join(_model_dir, CLONED_SPKS_FILE)
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return set(json.load(f))
    return set()


def _save_cloned_spks():
    path = os.path.join(_model_dir, CLONED_SPKS_FILE)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(sorted(cloned_spks), f, ensure_ascii=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global cosyvoice, sft_spk, cloned_spks, _model_dir
    _model_dir = os.environ.get("COSYVOICE_MODEL_DIR", args.model_dir)
    cosyvoice = AutoModel(model_dir=_model_dir)
    sft_spk = cosyvoice.list_available_spks()
    if len(sft_spk) == 0:
        sft_spk = [""]
    cloned_spks = _load_cloned_spks()
    # 清理 spk2info 中可能残留的无效条目（已被手动删除但 spk2info.pt 仍有）
    for spk in list(cloned_spks):
        if spk not in cosyvoice.frontend.spk2info:
            cloned_spks.discard(spk)
    logging.info(f"Model loaded from {_model_dir}, sft speakers: {sft_spk}, cloned: {sorted(cloned_spks)}")
    yield


# ===================== FastAPI 实例 =====================

app = FastAPI(
    title="CosyVoice TTS Service",
    description="基于 CosyVoice 的语音合成 RESTful API",
    version="2.0.0",
    lifespan=lifespan,
)

import hashlib
import uuid
import os
from fastapi import UploadFile

# 全局缓存：MD5 -> 文件路径
file_paths = {}

# 音色克隆缓存：(prompt_wav_md5, prompt_text) -> spk_id
clone_cache = {}


def _save_upload_wav(upload_file: UploadFile) -> str:
    """保存上传的音频文件到临时目录，基于 MD5 缓存避免重复存储"""
    # 1. 确保文件指针在起点
    upload_file.file.seek(0)
    # 2. 读取原始数据（同步方式，UploadFile.file 是 SpooledTemporaryFile）
    data = upload_file.file.read()
    # 3. 计算 MD5 值
    md5_hash = hashlib.md5(data).hexdigest()
    # 4. 检查缓存
    if md5_hash in file_paths:
        cached_path = file_paths[md5_hash]
        return md5_hash,cached_path
    # 5. 未命中缓存：保存新文件
    tmp_dir = os.path.abspath("tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    suffix = os.path.splitext(upload_file.filename or "wav")[1] or ".wav"
    file_path = os.path.join(tmp_dir, str(uuid.uuid4()) + suffix)
    with open(file_path, 'wb') as f:
        f.write(data)
    file_paths[md5_hash] = file_path
    return md5_hash,file_path

def _save_ls_wav(upload_file: UploadFile) -> str:
    suffix = os.path.splitext(upload_file.filename or "wav")[1] or ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(upload_file.file.read())
    tmp.close()
    return tmp.name

def _collect_full_audio(generator):
    chunks = []
    for item in generator:
        chunks.append(item["tts_speech"].numpy().flatten())
    if not chunks:
        return np.array([])
    return np.concatenate(chunks)


def _numpy_to_wav_bytes(audio_np: np.ndarray, sample_rate: int) -> io.BytesIO:
    tensor = torch.from_numpy(audio_np).unsqueeze(0)
    buffer = io.BytesIO()
    torchaudio.save(buffer, tensor, sample_rate, format="wav")
    buffer.seek(0)
    return buffer


@functools.lru_cache(maxsize=128)
def _check_prompt_wav(prompt_wav_path: str):
    """校验 prompt 音频采样率"""
    wav_info = torchaudio.info(prompt_wav_path)
    if wav_info.sample_rate < PROMPT_SR:
        raise HTTPException(
            status_code=400,
            detail=f"prompt 音频采样率 {wav_info.sample_rate} 低于最低要求 {PROMPT_SR}",
        )


def _resolve_seed(seed: int) -> int:
    actual = seed if seed != 0 else random.randint(1, 100000000)
    set_all_random_seed(actual)
    return actual


def _tts_response(audio_np: np.ndarray):
    """将合成音频封装为 StreamingResponse"""
    if audio_np.size == 0:
        raise HTTPException(status_code=500, detail="音频生成失败，结果为空")
    wav_buffer = _numpy_to_wav_bytes(audio_np, cosyvoice.sample_rate)
    return StreamingResponse(
        wav_buffer,
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=tts_output.wav"},
    )


def _vc_with_cached_spk(source_wav_path: str, spk_id: str, speed: float):
    """使用已注册音色进行语音转换"""
    cached = cosyvoice.frontend.spk2info[spk_id]
    source_speech_token, source_speech_token_len = cosyvoice.frontend._extract_speech_token(source_wav_path)
    model_input = {
        'source_speech_token': source_speech_token,
        'source_speech_token_len': source_speech_token_len,
        'flow_prompt_speech_token': cached['flow_prompt_speech_token'],
        'flow_prompt_speech_token_len': cached['flow_prompt_speech_token_len'],
        'prompt_speech_feat': cached['prompt_speech_feat'],
        'prompt_speech_feat_len': cached['prompt_speech_feat_len'],
        'flow_embedding': cached['flow_embedding'],
    }
    for model_output in cosyvoice.model.tts(**model_input, stream=False, speed=speed):
        yield model_output


# ===================== 查询接口 =====================

@app.get("/api/speakers", summary="获取所有可用音色（预训练 + 已克隆）")
def get_speakers():
    return JSONResponse(content={
        "sft_speakers": sft_spk,
        "cloned_speakers": sorted(cloned_spks),
    })


# ===================== 音色克隆接口 =====================

@app.post("/api/clone", summary="注册克隆音色并持久化为 pt 文件")
def clone_voice(
    spk_id: str = Form(..., description="克隆音色名称（唯一标识）"),
    prompt_wav: UploadFile = File(..., description="参考音频（采样率 ≥ 16kHz，时长 ≤ 30s）"),
    prompt_text: Optional[str] = Form(default=None,
                                      description="参考音频对应的文本（提供则走 zero-shot 模式，不提供则走跨语种模式）"),
):
    """提取参考音频的音色特征并持久化到 spk2info.pt，后续 TTS/VC 接口可直接使用 spk_id。"""
    if spk_id in sft_spk:
        raise HTTPException(status_code=400, detail=f"spk_id '{spk_id}' 与预训练音色冲突")

    _,prompt_wav_path = _save_upload_wav(prompt_wav)
    _check_prompt_wav(prompt_wav_path)

    if prompt_text:
        cosyvoice.add_zero_shot_spk(prompt_text, prompt_wav_path, spk_id)
    else:
        # 跨语种模式：prompt_text 留空，但仍然需要前端提取流程
        model_input = cosyvoice.frontend.frontend_zero_shot('', '', prompt_wav_path, cosyvoice.sample_rate, '')
        del model_input['text']
        del model_input['text_len']
        cosyvoice.frontend.spk2info[spk_id] = model_input

    cosyvoice.save_spkinfo()
    cloned_spks.add(spk_id)
    _save_cloned_spks()
    logging.info(f"Cloned voice '{spk_id}' registered and saved to spk2info.pt")
    return JSONResponse(content={"status": "ok", "spk_id": spk_id})


@app.get("/api/clone", summary="列出已克隆音色")
def list_cloned():
    return JSONResponse(content={"cloned_speakers": sorted(cloned_spks)})


@app.delete("/api/clone/{spk_id}", summary="删除克隆音色")
def delete_clone(spk_id: str):
    if spk_id not in cloned_spks:
        raise HTTPException(status_code=404, detail=f"克隆音色 '{spk_id}' 不存在")
    del cosyvoice.frontend.spk2info[spk_id]
    cloned_spks.discard(spk_id)
    cosyvoice.save_spkinfo()
    _save_cloned_spks()
    logging.info(f"Cloned voice '{spk_id}' removed from spk2info.pt")
    return JSONResponse(content={"status": "ok", "spk_id": spk_id})


# ===================== TTS 接口（四种模式各自独立） =====================

@app.post("/api/tts/sft", summary="预训练音色合成")
def tts_sft(
    tts_text: str = Form(..., description="需要合成的文本"),
    spk_id: str = Form(..., description="预训练音色名称"),
    speed: float = Form(default=1.0, ge=0.5, le=2.0, description="语速 0.5~2.0"),
    seed: int = Form(default=0, description="随机种子，0 表示随机"),
):
    if spk_id not in sft_spk:
        raise HTTPException(status_code=400, detail=f"预训练音色 '{spk_id}' 不存在")
    _resolve_seed(seed)
    logging.info(f"sft TTS: spk={spk_id}, text={tts_text[:50]}...")
    audio_np = _collect_full_audio(
        cosyvoice.inference_sft(tts_text, spk_id, stream=False, speed=speed)
    )
    return _tts_response(audio_np)


@app.post("/api/tts/zero_shot", summary="3s 极速复刻合成")
def tts_zero_shot(
    tts_text: str = Form(..., description="需要合成的文本"),
    prompt_wav: UploadFile = File(..., description="参考音频（未注册音色时使用）"),
    prompt_text: str = Form(..., description="参考音频对应的文本（未注册音色时使用）"),
    speed: float = Form(default=1.0, ge=0.5, le=2.0, description="语速 0.5~2.0"),
    seed: int = Form(default=0, description="随机种子，0 表示随机"),
):
    _resolve_seed(seed)

    # 保存上传的音频文件
    wav_md5,prompt_wav_path = _save_upload_wav(prompt_wav)
    _check_prompt_wav(prompt_wav_path)

    # 构建缓存键：(wav_md5, prompt_text)
    cache_key = (wav_md5, prompt_text)

    # 检查是否有缓存的克隆ID
    cached_spk_id = clone_cache.get(cache_key)

    if cached_spk_id and cached_spk_id in cosyvoice.frontend.spk2info:
        # 使用缓存的克隆ID
        logging.info(f"zero_shot TTS (adhoc with cache): using cached spk_id={cached_spk_id}, text={tts_text[:50]}...")
        audio_np = _collect_full_audio(
            cosyvoice.inference_zero_shot(tts_text, '', '', cached_spk_id, stream=False, speed=speed)
        )
    else:
        # 没有缓存，进行克隆
        # 生成唯一的 spk_id
        new_spk_id = f"auto_clone_{uuid.uuid4().hex[:8]}"

        # 执行克隆
        logging.info(f"zero_shot TTS (adhoc cloning): creating new spk_id={new_spk_id}, text={tts_text[:50]}...")
        cosyvoice.add_zero_shot_spk(prompt_text, prompt_wav_path, new_spk_id)
        cosyvoice.save_spkinfo()
        cloned_spks.add(new_spk_id)
        _save_cloned_spks()

        # 缓存克隆ID
        clone_cache[cache_key] = new_spk_id

        # 使用新克隆的音色进行合成
        audio_np = _collect_full_audio(
            cosyvoice.inference_zero_shot(tts_text, '', '', new_spk_id, stream=False, speed=speed)
        )
    
    return _tts_response(audio_np)


@app.post("/api/tts/cross_lingual", summary="跨语种复刻合成")
def tts_cross_lingual(
    tts_text: str = Form(..., description="需要合成的文本"),
    spk_id: Optional[str] = Form(default=None, description="已克隆音色名称（二选一：spk_id 或 prompt_wav）"),
    prompt_wav: Optional[UploadFile] = File(default=None, description="参考音频（未注册音色时使用）"),
    speed: float = Form(default=1.0, ge=0.5, le=2.0, description="语速 0.5~2.0"),
    seed: int = Form(default=0, description="随机种子，0 表示随机"),
):
    use_cached = spk_id is not None and spk_id != ''
    use_adhoc = prompt_wav is not None and prompt_wav.filename

    if use_cached and use_adhoc:
        raise HTTPException(status_code=400, detail="spk_id 和 prompt_wav 不能同时提供，请二选一")
    if not use_cached and not use_adhoc:
        raise HTTPException(status_code=400, detail="请提供 spk_id（已克隆音色）或 prompt_wav（即时复刻）")

    _resolve_seed(seed)

    if use_cached:
        if spk_id not in cosyvoice.frontend.spk2info:
            raise HTTPException(status_code=400, detail=f"音色 '{spk_id}' 未注册，请先调用 /api/clone")
        logging.info(f"cross_lingual TTS (cached): spk={spk_id}, text={tts_text[:50]}...")
        audio_np = _collect_full_audio(
            cosyvoice.inference_cross_lingual(tts_text, '', spk_id, stream=False, speed=speed)
        )
    else:
        _,prompt_wav_path = _save_upload_wav(prompt_wav)
        _check_prompt_wav(prompt_wav_path)
        logging.info(f"cross_lingual TTS (adhoc): text={tts_text[:50]}...")
        audio_np = _collect_full_audio(
            cosyvoice.inference_cross_lingual(tts_text, prompt_wav_path, '', stream=False, speed=speed)
        )
    return _tts_response(audio_np)


@app.post("/api/tts/instruct", summary="自然语言控制合成")
def tts_instruct(
    tts_text: str = Form(..., description="需要合成的文本"),
    spk_id: str = Form(..., description="预训练音色名称"),
    instruct_text: str = Form(..., description="自然语言控制指令"),
    speed: float = Form(default=1.0, ge=0.5, le=2.0, description="语速 0.5~2.0"),
    seed: int = Form(default=0, description="随机种子，0 表示随机"),
):
    if spk_id not in sft_spk:
        raise HTTPException(status_code=400, detail=f"预训练音色 '{spk_id}' 不存在")
    _resolve_seed(seed)
    logging.info(f"instruct TTS: spk={spk_id}, instruct={instruct_text[:50]}, text={tts_text[:50]}...")
    audio_np = _collect_full_audio(
        cosyvoice.inference_instruct(tts_text, spk_id, instruct_text, stream=False, speed=speed)
    )
    return _tts_response(audio_np)


# ===================== 语音转换接口 =====================

@app.post("/api/vc", summary="语音转换（将源音频转换为目标音色）")
def voice_conversion(
    source_wav: UploadFile = File(..., description="待转换的源音频"),
    spk_id: Optional[str] = Form(default=None, description="已克隆音色名称（二选一：spk_id 或 prompt_wav）"),
    prompt_wav: Optional[UploadFile] = File(default=None, description="参考音频（未注册音色时使用）"),
    speed: float = Form(default=1.0, ge=0.5, le=2.0, description="语速 0.5~2.0"),
    seed: int = Form(default=0, description="随机种子，0 表示随机"),
):
    use_cached = spk_id is not None and spk_id != ''
    use_adhoc = prompt_wav is not None and prompt_wav.filename

    if use_cached and use_adhoc:
        raise HTTPException(status_code=400, detail="spk_id 和 prompt_wav 不能同时提供，请二选一")
    if not use_cached and not use_adhoc:
        raise HTTPException(status_code=400, detail="请提供 spk_id（已克隆音色）或 prompt_wav（即时转换）")

    _resolve_seed(seed)

    source_wav_path = _save_ls_wav(source_wav)
    try:
        if use_cached:
            if spk_id not in cosyvoice.frontend.spk2info:
                raise HTTPException(status_code=400, detail=f"音色 '{spk_id}' 未注册，请先调用 /api/clone")
            logging.info(f"VC (cached): spk={spk_id}")
            audio_np = _collect_full_audio(
                _vc_with_cached_spk(source_wav_path, spk_id, speed)
            )
        else:
            _,prompt_wav_path = _save_upload_wav(prompt_wav)
            _check_prompt_wav(prompt_wav_path)
            logging.info(f"VC (adhoc)")
            audio_np = _collect_full_audio(
                cosyvoice.inference_vc(source_wav_path, prompt_wav_path, stream=False, speed=speed)
            )
        return _tts_response(audio_np)
    finally:
        os.remove(source_wav_path)

# ===================== 启动入口 =====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CosyVoice FastAPI TTS Service")
    parser.add_argument("--port", type=int, default=50000, help="服务监听端口")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务监听地址")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="pretrained_models/CosyVoice2-0.5B",
        help="模型本地路径或 ModelScope repo id",
    )
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)