"""이미지 로딩과 전처리. 원본을 변경하지 않고 항상 새 이미지를 반환한다."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

from PIL import Image, ImageOps

# 비주얼 토큰은 대략 pixels / (28 * 28) 이므로 1.6MP ≈ 2,000 토큰이다.
# 문서 OCR 정확도와 ZeroGPU 실행 시간 사이의 기본 절충값.
DEFAULT_MAX_PIXELS = 1_600_000
MIN_MAX_PIXELS = 200_000
MAX_MAX_PIXELS = 4_000_000

# ViT 패치 크기는 고정이라, 작은 스캔본을 그대로 넣으면 본문 글자가 패치 몇 개에
# 걸쳐 뭉개진다. 정보가 늘지는 않지만 글자당 패치 수가 늘어 잔글씨 인식이 나아지므로
# 작은 문서는 이 하한까지 확대한다. 원본이 지나치게 작으면 확대해도 소용없으니 배율을 제한한다.
DOCUMENT_MIN_PIXELS = 1_200_000
MAX_UPSCALE = 2.0

_ALLOWED_SCHEMES = ("http", "https")
_DOWNLOAD_TIMEOUT_SEC = 20


def load_image(source: str) -> Image.Image:
    """URL 또는 로컬 경로에서 이미지를 읽는다."""
    scheme = urlparse(source).scheme.lower()
    if scheme in _ALLOWED_SCHEMES:
        with urlopen(source, timeout=_DOWNLOAD_TIMEOUT_SEC) as response:  # noqa: S310 - scheme 검증 완료
            data = response.read()
        return _normalize(Image.open(BytesIO(data)))

    if scheme and scheme != "file":
        raise ValueError(f"지원하지 않는 경로 형식입니다: {source}")

    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")
    return _normalize(Image.open(path))


def fit_pixels(
    image: Image.Image,
    max_pixels: int = DEFAULT_MAX_PIXELS,
    min_pixels: int = 0,
) -> Image.Image:
    """총 픽셀 수를 [min_pixels, max_pixels] 안으로 맞춘 새 이미지를 반환한다.

    상한을 넘으면 줄이고, 하한에 못 미치면 MAX_UPSCALE 배까지 키운다.
    """
    max_pixels = max(MIN_MAX_PIXELS, min(int(max_pixels), MAX_MAX_PIXELS))
    min_pixels = min(int(min_pixels), max_pixels)
    width, height = image.size
    pixels = width * height

    if pixels > max_pixels:
        # 반올림하면 상한을 근소하게 넘길 수 있어 양쪽 다 내림한다.
        scale = (max_pixels / pixels) ** 0.5
        target = (max(1, int(width * scale)), max(1, int(height * scale)))
        return image.resize(target, Image.LANCZOS)

    if pixels < min_pixels:
        scale = min(MAX_UPSCALE, (min_pixels / pixels) ** 0.5)
        target = (max(1, int(width * scale)), max(1, int(height * scale)))
        return image.resize(target, Image.LANCZOS)

    return image


def _normalize(image: Image.Image) -> Image.Image:
    """EXIF 회전을 적용하고 RGB로 변환한다. 휴대폰으로 찍은 문서가 눕지 않도록."""
    return ImageOps.exif_transpose(image).convert("RGB")
