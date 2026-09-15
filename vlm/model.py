"""Qwen2.5-VL 모델 로딩과 스트리밍 생성."""

from __future__ import annotations

import os
from collections.abc import Iterator
from threading import Thread

import transformers
from packaging.version import Version
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, TextIteratorStreamer

from .prompts import TRANSCRIBE_PROMPT
from .runtime import DEVICE, DTYPE, GPU

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-VL-3B-Instruct")

DEFAULT_MAX_NEW_TOKENS = 2048
MAX_NEW_TOKENS_LIMIT = 4096
REPETITION_PENALTY = 1.05
GPU_DURATION_SEC = 180

# 생성 스레드가 죽어 토큰이 끊기면 소비자가 영원히 대기하게 되므로 상한을 둔다.
# CPU/MPS는 첫 토큰까지 수 분이 걸릴 수 있어 제한하지 않는다(None = 무한 대기).
STREAM_TIMEOUT_SEC = 120 if DEVICE == "cuda" else None

# transformers 4.56에서 torch_dtype -> dtype으로 이름이 바뀌었고 둘 다 **kwargs로
# 조용히 삼켜지기 때문에, try/except가 아니라 버전으로 골라야 한다.
_DTYPE_KWARG = "dtype" if Version(transformers.__version__) >= Version("4.56.0") else "torch_dtype"


def _load() -> tuple[Qwen2_5_VLForConditionalGeneration, AutoProcessor]:
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        **{_DTYPE_KWARG: DTYPE},
    )
    model = model.to(DEVICE).eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    return model, processor


model, processor = _load()


def _build_inputs(image: Image.Image, prompt: str):
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": prompt}],
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return processor(text=[text], images=[image], return_tensors="pt").to(model.device)


@GPU(duration=GPU_DURATION_SEC)
def stream_transcribe(
    image: Image.Image,
    prompt: str = TRANSCRIBE_PROMPT,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> Iterator[str]:
    """생성된 텍스트 조각(delta)을 순서대로 내보낸다. 누적은 호출한 쪽에서 한다."""
    inputs = _build_inputs(image, prompt)
    streamer = TextIteratorStreamer(
        processor.tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
        timeout=STREAM_TIMEOUT_SEC,
    )

    failures: list[BaseException] = []

    def _generate() -> None:
        try:
            model.generate(
                **inputs,
                streamer=streamer,
                max_new_tokens=min(int(max_new_tokens), MAX_NEW_TOKENS_LIMIT),
                do_sample=False,
                repetition_penalty=REPETITION_PENALTY,
            )
        except BaseException as exc:  # 스트림이 끝난 뒤 호출자에게 다시 던진다
            failures.append(exc)
            streamer.end()

    worker = Thread(target=_generate, daemon=True)
    worker.start()

    for delta in streamer:
        yield delta

    worker.join(timeout=10)
    if failures:
        raise failures[0]


def transcribe(
    image: Image.Image,
    prompt: str = TRANSCRIBE_PROMPT,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> str:
    return "".join(stream_transcribe(image, prompt, max_new_tokens)).strip()
