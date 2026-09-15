"""Hugging Face Space 엔트리포인트: 이미지 → 텍스트 전사 + 시각 요소 서술."""

from __future__ import annotations

from collections.abc import Iterator

import gradio as gr
from PIL import Image

from vlm.image import DEFAULT_MAX_PIXELS, MAX_MAX_PIXELS, MIN_MAX_PIXELS, cap_pixels
from vlm.model import DEFAULT_MAX_NEW_TOKENS, MAX_NEW_TOKENS_LIMIT, MODEL_ID, stream_transcribe
from vlm.prompts import TRANSCRIBE_PROMPT
from vlm.runtime import DEVICE

PIXELS_STEP = 100_000
TOKENS_STEP = 128

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
    prompt: str,
    max_new_tokens: float,
    max_pixels: float,
) -> Iterator[str]:
    if image is None:
        raise gr.Error("이미지를 먼저 올려 주세요.")

    prompt = (prompt or "").strip() or TRANSCRIBE_PROMPT
    prepared = cap_pixels(image, int(max_pixels))

    chunks: list[str] = []
    for delta in stream_transcribe(prepared, prompt, int(max_new_tokens)):
        chunks.append(delta)
        yield "".join(chunks)


# Gradio 6부터 theme/css는 Blocks가 아니라 launch()에서 받는다.
with gr.Blocks(title="이미지 전사 · Qwen2.5-VL") as demo:
    with gr.Column(elem_classes="wrap-narrow"):
        gr.Markdown(
            "# 이미지 전사 (Qwen2.5-VL 3B)\n"
            "이미지 속 텍스트는 원문 그대로 옮기고, 텍스트가 아닌 시각 요소는 "
            "`[이미지: 상세 설명]`으로 대체합니다. 사진 한 장뿐이면 보이는 그대로 상세히 서술합니다."
        )

        with gr.Row():
            with gr.Column(scale=4):
                image_input = gr.Image(
                    label="이미지",
                    type="pil",
                    sources=["upload", "clipboard"],
                    height=420,
                )
                with gr.Row():
                    run_button = gr.Button("전사하기", variant="primary", scale=3)
                    stop_button = gr.Button("중지", variant="stop", scale=1)

                with gr.Accordion("고급 설정", open=False):
                    prompt_input = gr.Textbox(
                        label="프롬프트",
                        value=TRANSCRIBE_PROMPT,
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
                        info="높일수록 잔글씨 인식에 유리하지만 느려집니다.",
                        minimum=MIN_MAX_PIXELS,
                        maximum=MAX_MAX_PIXELS,
                        step=PIXELS_STEP,
                        value=DEFAULT_MAX_PIXELS,
                    )

            with gr.Column(scale=5):
                output = gr.Textbox(
                    label="결과",
                    lines=26,
                    max_lines=26,
                    buttons=["copy"],
                    interactive=False,
                    elem_id="result",
                )

        gr.Markdown(f"모델: `{MODEL_ID}` · 디바이스: `{DEVICE}`")

    run_event = run_button.click(
        fn=describe,
        inputs=[image_input, prompt_input, max_tokens_input, max_pixels_input],
        outputs=output,
        api_name="describe",
    )
    stop_button.click(fn=None, inputs=None, outputs=None, cancels=[run_event])

demo.queue(max_size=16)

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), css=CSS)
