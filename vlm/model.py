"""Qwen2.5-VL 모델 로딩, 이미지 종류 판별, 스트리밍 생성."""

from __future__ import annotations

import os
from collections.abc import Iterator
from threading import Thread

import transformers
from packaging.version import Version
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, TextIteratorStreamer

from .image import DEFAULT_MAX_PIXELS, DOCUMENT_MIN_PIXELS, fit_pixels
from .prompts import CLASSIFY_PROMPT, PROMPTS
from .runtime import DEVICE, DTYPE, GPU

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-VL-3B-Instruct")

DEFAULT_MAX_NEW_TOKENS = 2048
MAX_NEW_TOKENS_LIMIT = 4096
REPETITION_PENALTY = 1.05
GPU_DURATION_SEC = 180

# 분류는 한 단어만 받으면 되고 저해상도로 충분하다. 본 생성과 같은 GPU 세션
# 안에서 돌기 때문에 ZeroGPU 할당은 여전히 한 번이다.
CLASSIFY_MAX_TOKENS = 8
CLASSIFY_MAX_PIXELS = 400_000

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


def classify(image: Image.Image) -> str:
    """이미지가 'document'인지 'photo'인지 한 단어로 판별한다.

    모델이 스스로 문서/사진을 분기하게 두면 문서를 그림 하나로 취급해 한 줄만
    뱉는다. 종류를 먼저 확정해서 프롬프트로 단정해 주기 위한 단계다.
    """
    inputs = _build_inputs(fit_pixels(image, CLASSIFY_MAX_PIXELS), CLASSIFY_PROMPT)
    generated = model.generate(**inputs, max_new_tokens=CLASSIFY_MAX_TOKENS, do_sample=False)
    answer = processor.batch_decode(
        [generated[0][inputs.input_ids.shape[1] :]], skip_special_tokens=True
    )[0]
    # 애매하면 문서로 본다. 이 도구의 입력은 대부분 문서다.
    return "photo" if "PHOTO" in answer.upper() else "document"


@GPU(duration=GPU_DURATION_SEC)
def stream_describe(
    image: Image.Image,
    mode: str | None = None,
    prompt: str = "",
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    max_pixels: int = DEFAULT_MAX_PIXELS,
) -> Iterator[tuple[str, str]]:
    """(모드, 새로 생성된 텍스트 조각)을 순서대로 내보낸다. 누적은 호출한 쪽에서 한다.

    mode가 None이면 분류해서 정한다. prompt를 주면 분류도 기본 프롬프트도 쓰지 않는다.
    """
    if prompt.strip():
        mode = mode or "custom"
    else:
        mode = mode or classify(image)
        prompt = PROMPTS[mode]

    # 작은 스캔본은 확대해야 잔글씨가 산다. 사진은 확대해도 얻을 게 없다.
    min_pixels = DOCUMENT_MIN_PIXELS if mode != "photo" else 0
    inputs = _build_inputs(fit_pixels(image, max_pixels, min_pixels), prompt)

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
        yield mode, delta

    worker.join(timeout=10)
    if failures:
        raise failures[0]


def describe(
    image: Image.Image,
    mode: str | None = None,
    prompt: str = "",
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    max_pixels: int = DEFAULT_MAX_PIXELS,
) -> tuple[str, str]:
    """스트림을 끝까지 받아 (모드, 전체 텍스트)로 돌려준다."""
    chunks: list[str] = []
    resolved = mode or "document"
    for resolved, delta in stream_describe(image, mode, prompt, max_new_tokens, max_pixels):
        chunks.append(delta)
    return resolved, "".join(chunks).strip()
