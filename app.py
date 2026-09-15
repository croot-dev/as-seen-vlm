"""Hugging Face Space 엔트리포인트: 문서는 전사하고, 사진은 서술한다."""

from __future__ import annotations

from collections.abc import Iterator

import gradio as gr
from PIL import Image

from vlm.image import DEFAULT_MAX_PIXELS, MAX_MAX_PIXELS, MIN_MAX_PIXELS
from vlm.model import DEFAULT_MAX_NEW_TOKENS, MAX_NEW_TOKENS_LIMIT, MODEL_ID, stream_describe
from vlm.prompts import PROMPTS
from vlm.runtime import DEVICE

PIXELS_STEP = 100_000
TOKENS_STEP = 128

MODE_CHOICES = [("자동 판별", "auto"), ("문서 (전사)", "document"), ("사진 (서술)", "photo")]
MODE_LABELS = {
    "document": "문서 — 글자를 그대로 전사",
    "photo": "사진 — 보이는 것을 서술",
    "custom": "직접 입력한 프롬프트",
}
SPLIT_CHOICES = [
    ("자동 — 넓은 문서만 세로 3분할", "auto"),
    ("항상 세로 3분할", "columns"),
    ("분할 안 함", "none"),
]

CSS = """
.wrap-narrow { max-width: 1180px; margin: 0 auto; }
#result textarea {
  font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, monospace;
  font-size: 14px;
  line-height: 1.7;
}
#prompt textarea { font-size: 13px; line-height: 1.6; }
"""


def describe(
    image: Image.Image | None,
    mode: str,
    prompt: str,
    max_new_tokens: float,
    max_pixels: float,
    split: str,
) -> Iterator[tuple[str, str]]:
    if image is None:
        raise gr.Error("이미지를 먼저 올려 주세요.")

    chunks: list[str] = []
    for resolved, delta in stream_describe(
        image,
        None if mode == "auto" else mode,
        prompt or "",
        int(max_new_tokens),
        int(max_pixels),
        split,
    ):
        chunks.append(delta)
        yield MODE_LABELS.get(resolved, resolved), "".join(chunks)


def fill_prompt(mode: str) -> str:
    """모드를 고르면 그 모드의 기본 프롬프트를 채워 보여준다. 자동이면 비운다."""
    return PROMPTS.get(mode, "")


with gr.Blocks(title="이미지 전사 · Qwen2.5-VL") as demo:
    with gr.Column(elem_classes="wrap-narrow"):
        gr.Markdown(
            "# 이미지 전사 (Qwen2.5-VL 3B)\n"
            "문서는 글자를 요약 없이 원문 그대로 옮겨 적고, 사진은 보이는 것을 상세히 서술합니다. "
            "어떤 언어로 쓰여 있든 번역하지 않습니다."
        )

        with gr.Row():
            with gr.Column(scale=4):
                image_input = gr.Image(
                    label="이미지",
                    type="pil",
                    sources=["upload", "clipboard"],
                    height=400,
                )
                mode_input = gr.Radio(
                    choices=MODE_CHOICES,
                    value="auto",
                    label="처리 방식",
                    info="자동 판별은 모델에게 문서인지 사진인지 먼저 물어봅니다.",
                )
                with gr.Row():
                    run_button = gr.Button("실행", variant="primary", scale=3)
                    stop_button = gr.Button("중지", variant="stop", scale=1)

                with gr.Accordion("고급 설정", open=False):
                    split_input = gr.Radio(
                        choices=SPLIT_CHOICES,
                        value="auto",
                        label="문서 분할",
                        info="넓은 대시보드·다열 UI는 영역별로 전사한 뒤 중복 줄을 제거해 합칩니다.",
                    )
                    prompt_input = gr.Textbox(
                        label="프롬프트",
                        value="",
                        placeholder=(
                            "비워두면 처리 방식에 맞는 기본 프롬프트를 씁니다. "
                            "위에서 문서나 사진을 고르면 그 프롬프트가 여기 채워집니다."
                        ),
                        lines=10,
                        elem_id="prompt",
                    )
                    max_tokens_input = gr.Slider(
                        label="최대 생성 토큰",
                        minimum=256,
                        maximum=MAX_NEW_TOKENS_LIMIT,
                        step=TOKENS_STEP,
                        value=DEFAULT_MAX_NEW_TOKENS,
                    )
                    max_pixels_input = gr.Slider(
                        label="입력 해상도 상한 (픽셀)",
                        info="작은 문서는 잔글씨를 살리려고 자동으로 확대합니다. 이 값은 상한입니다.",
                        minimum=MIN_MAX_PIXELS,
                        maximum=MAX_MAX_PIXELS,
                        step=PIXELS_STEP,
                        value=DEFAULT_MAX_PIXELS,
                    )

            with gr.Column(scale=5):
                mode_output = gr.Textbox(label="처리 방식 판별 결과", interactive=False)
                output = gr.Textbox(
                    label="결과",
                    lines=23,
                    max_lines=23,
                    buttons=["copy"],
                    interactive=False,
                    elem_id="result",
                )

        gr.Markdown(f"모델: `{MODEL_ID}` · 디바이스: `{DEVICE}`")

    mode_input.change(fn=fill_prompt, inputs=mode_input, outputs=prompt_input)
    run_event = run_button.click(
        fn=describe,
        inputs=[
            image_input,
            mode_input,
            prompt_input,
            max_tokens_input,
            max_pixels_input,
            split_input,
        ],
        outputs=[mode_output, output],
        api_name="describe",
    )
    stop_button.click(fn=None, inputs=None, outputs=None, cancels=[run_event])

demo.queue(max_size=16)

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), css=CSS)
