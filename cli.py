"""로컬에서 이미지 한 장을 처리하는 CLI. Space를 띄우지 않고 프롬프트를 확인할 때 쓴다.

사용법:
    python cli.py ./sample.png
    python cli.py ./scan.jpg --mode document
    python cli.py https://example.com/page.jpg --max-new-tokens 3072
    python cli.py ./sample.png --prompt-file ./my_prompt.txt --out result.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vlm.image import DEFAULT_MAX_PIXELS, load_image
from vlm.model import DEFAULT_MAX_NEW_TOKENS, stream_describe

MODES = ("auto", "document", "photo")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen2.5-VL로 이미지를 전사하거나 서술한다.")
    parser.add_argument("source", help="이미지 파일 경로 또는 http(s) URL")
    parser.add_argument("--mode", choices=MODES, default="auto", help="기본값은 자동 판별")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS)
    parser.add_argument(
        "--split",
        choices=("auto", "columns", "none"),
        default="auto",
        help="문서 열 분할 방식 (기본값: 넓은 문서만 자동 분할)",
    )
    parser.add_argument("--prompt-file", type=Path, default=None, help="프롬프트를 담은 텍스트 파일")
    parser.add_argument("--out", type=Path, default=None, help="결과를 저장할 파일 경로")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    try:
        image = load_image(args.source)
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"이미지 로드 실패: {exc}", file=sys.stderr)
        return 1

    prompt = ""
    if args.prompt_file is not None:
        try:
            prompt = args.prompt_file.read_text(encoding="utf-8").strip()
        except OSError as exc:
            print(f"프롬프트 파일 읽기 실패: {exc}", file=sys.stderr)
            return 1

    chunks: list[str] = []
    announced = False
    for mode, delta in stream_describe(
        image,
        None if args.mode == "auto" else args.mode,
        prompt,
        args.max_new_tokens,
        args.max_pixels,
        args.split,
    ):
        if not announced:
            print(f"[모드: {mode}]", file=sys.stderr, flush=True)
            announced = True
        chunks.append(delta)
        print(delta, end="", flush=True)
    print()

    if args.out is not None:
        args.out.write_text("".join(chunks).strip() + "\n", encoding="utf-8")
        print(f"저장: {args.out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
