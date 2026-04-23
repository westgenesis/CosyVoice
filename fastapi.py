import os
import sys
import argparse
import io
import tempfile
import random
from typing import Optional, List
from enum import Enum

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


# ===================== 枚举 & 常量 =====================

class InferenceMode(str, Enum):
    """推理模式"""
    PRETRAINED    = "预训练音色"
    ZERO_SHOT     = "3s极速复刻"
    CROSS_LINGUAL = "跨语种复刻"
    INSTRUCT      = "自然语言控制"


PROMPT_SR = 16000


# ===================== 全局模型 =====================

cosyvoice: Optional[AutoModel] = None
sft_spk: List[str] = []


# ===================== FastAPI 实例 =====================

app = FastAPI(
    title="CosyVoice TTS Service",
    description="基于 CosyVoice 的语音合成 RESTful API，一次性返回完整音频",
    version="1.0.0",
)


# ===================== 启动时加载模型 =====================

@app.on_event("startup")
async def load_model():
    """应用启动时加载模型"""
    global cosyvoice, sft_spk
    model_dir = os.environ.get("COSYVOICE_MODEL_DIR", args.model_dir)
    cosyvoice = AutoModel(model_dir=model_dir)
    sft_spk = cosyvoice.list_available_spks()
    if len(sft_spk) == 0:
        sft_spk = [""]
    logging.info(f"Model loaded from {model_dir}, available speakers: {sft_spk}")


# ===================== 工具函数 =====================

def _save_upload_wav(upload_file: UploadFile) -> str:
    """将上传的音频文件保存到临时路径并返回路径"""
    suffix = os.path.splitext(upload_file.filename or "wav")[1] or ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(upload_file.file.read())
    tmp.close()
    return tmp.name


def _collect_full_audio(generator):
    """
    从 CosyVoice 的生成器中收集所有音频片段并拼接，
    确保一次性返回完整音频而不是分段流式返回。
    """
    chunks = []
    for item in generator:
        chunks.append(item["tts_speech"].numpy().flatten())
    if not chunks:
        return np.array([])
    return np.concatenate(chunks)


def _numpy_to_wav_bytes(audio_np: np.ndarray, sample_rate: int) -> io.BytesIO:
    """将 numpy 数组转为 WAV 格式的 BytesIO 对象"""
    tensor = torch.from_numpy(audio_np).unsqueeze(0)  # (1, T)
    buffer = io.BytesIO()
    torchaudio.save(buffer, tensor, sample_rate, format="wav")
    buffer.seek(0)
    return buffer


# ===================== API 接口 =====================

@app.get("/api/speakers", summary="获取可用预训练音色列表")
async def get_speakers():
    """返回当前模型支持的所有预训练音色"""
    return JSONResponse(content={"speakers": sft_spk})


@app.post("/api/tts", summary="语音合成（一次性返回完整音频）")
async def text_to_speech(
    tts_text: str = Form(..., description="需要合成的文本"),
    mode: InferenceMode = Form(..., description="推理模式: 预训练音色 / 3s极速复刻 / 跨语种复刻 / 自然语言控制"),
    sft_speaker: Optional[str] = Form(default="", description="预训练音色名称（预训练音色 / 自然语言控制 模式使用）"),
    prompt_text: Optional[str] = Form(default="", description="prompt文本（3s极速复刻 模式使用，需与prompt音频内容一致）"),
    prompt_wav: Optional[UploadFile] = File(default=None, description="prompt音频文件（3s极速复刻 / 跨语种复刻 模式使用，采样率≥16kHz，不超过30s）"),
    instruct_text: Optional[str] = Form(default="", description="instruct文本（自然语言控制 模式使用）"),
    seed: int = Form(default=0, description="随机推理种子，传入 0 表示随机生成"),
    speed: float = Form(default=1.0, ge=0.5, le=2.0, description="语速调节，范围 0.5~2.0"),
):
    """
    根据指定的推理模式合成语音。

    所有音频片段生成完毕后拼接为一个完整的 WAV 文件一次性返回。
    """

    # ---------- 保存上传的 prompt 音频 ----------
    prompt_wav_path: Optional[str] = None
    if prompt_wav is not None and prompt_wav.filename:
        prompt_wav_path = _save_upload_wav(prompt_wav)

    # ---------- 模式参数校验 ----------

    # 自然语言控制模式
    if mode == InferenceMode.INSTRUCT:
        if not instruct_text:
            raise HTTPException(status_code=400, detail="自然语言控制模式需要提供 instruct_text")
        if prompt_wav_path or prompt_text:
            logging.info("自然语言控制模式: prompt音频/prompt文本将被忽略")

    # 跨语种复刻模式
    if mode == InferenceMode.CROSS_LINGUAL:
        if instruct_text:
            logging.info("跨语种复刻模式: instruct文本将被忽略")
        if prompt_wav_path is None:
            raise HTTPException(status_code=400, detail="跨语种复刻模式需要提供 prompt_wav 音频文件")

    # 3s极速复刻 / 跨语种复刻 共用的校验
    if mode in (InferenceMode.ZERO_SHOT, InferenceMode.CROSS_LINGUAL):
        if prompt_wav_path is None:
            raise HTTPException(status_code=400, detail="当前模式需要提供 prompt_wav 音频文件")
        wav_info = torchaudio.info(prompt_wav_path)
        if wav_info.sample_rate < PROMPT_SR:
            raise HTTPException(
                status_code=400,
                detail=f"prompt音频采样率 {wav_info.sample_rate} 低于最低要求 {PROMPT_SR}",
            )

    # 预训练音色模式
    if mode == InferenceMode.PRETRAINED:
        if instruct_text or prompt_wav_path or prompt_text:
            logging.info("预训练音色模式: prompt文本/prompt音频/instruct文本将被忽略")
        if not sft_speaker:
            raise HTTPException(status_code=400, detail="预训练音色模式需要提供 sft_speaker")

    # 3s极速复刻模式
    if mode == InferenceMode.ZERO_SHOT:
        if not prompt_text:
            raise HTTPException(status_code=400, detail="3s极速复刻模式需要提供 prompt_text")
        if instruct_text:
            logging.info("3s极速复刻模式: instruct文本将被忽略")

    # ---------- 随机种子 ----------
    actual_seed = seed if seed != 0 else random.randint(1, 100000000)
    set_all_random_seed(actual_seed)

    # ---------- 推理（统一 stream=False，收集所有片段后一次性返回） ----------
    audio_np = np.array([])
    try:
        if mode == InferenceMode.PRETRAINED:
            logging.info("get sft inference request")
            audio_np = _collect_full_audio(
                cosyvoice.inference_sft(tts_text, sft_speaker, stream=False, speed=speed)
            )
        elif mode == InferenceMode.ZERO_SHOT:
            logging.info("get zero_shot inference request")
            audio_np = _collect_full_audio(
                cosyvoice.inference_zero_shot(
                    tts_text, prompt_text, prompt_wav_path, stream=False, speed=speed
                )
            )
        elif mode == InferenceMode.CROSS_LINGUAL:
            logging.info("get cross_lingual inference request")
            audio_np = _collect_full_audio(
                cosyvoice.inference_cross_lingual(
                    tts_text, prompt_wav_path, stream=False, speed=speed
                )
            )
        elif mode == InferenceMode.INSTRUCT:
            logging.info("get instruct inference request")
            audio_np = _collect_full_audio(
                cosyvoice.inference_instruct(
                    tts_text, sft_speaker, instruct_text, stream=False, speed=speed
                )
            )
    finally:
        # 无论成功失败都清理临时文件
        if prompt_wav_path is not None:
            try:
                os.unlink(prompt_wav_path)
            except OSError:
                pass

    # ---------- 空音频检查 ----------
    if audio_np.size == 0:
        raise HTTPException(status_code=500, detail="音频生成失败，结果为空")

    # ---------- 返回完整 WAV ----------
    wav_buffer = _numpy_to_wav_bytes(audio_np, cosyvoice.sample_rate)
    return StreamingResponse(
        wav_buffer,
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=tts_output.wav"},
    )


# ===================== 启动入口 =====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CosyVoice FastAPI TTS Service")
    parser.add_argument("--port", type=int, default=8000, help="服务监听端口")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务监听地址")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="pretrained_models/CosyVoice2-0.5B",
        help="模型本地路径或 ModelScope repo id",
    )
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)
