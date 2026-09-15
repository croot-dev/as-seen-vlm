---
title: Qwen2.5 VL 3B
emoji: 🏢
colorFrom: indigo
colorTo: indigo
sdk: gradio
sdk_version: 6.27.0
python_version: '3.12'
app_file: app.py
pinned: false
short_description: 이미지 속 텍스트 전사 + 시각 요소 상세 서술
---

# 이미지 전사 (Qwen2.5-VL 3B)

이미지를 넣으면 종류를 먼저 판별한 뒤 처리한다.

- **문서**: 제목·본문·글머리 기호·표·상자 안 글자까지 요약 없이 원문 그대로 옮겨 적는다. 표는 `|` 로 칸을 구분한다.
- **사진**: 색·형태·개수·배치·배경까지 보이는 것을 상세히 서술한다.
- 어느 쪽이든 **원문 언어를 유지하고 번역하지 않으며**, 추측·해석·감정 묘사를 넣지 않는다.

처리 방식은 UI에서 자동 판별 / 문서 / 사진 중에 고를 수 있고, 문서나 사진을 고르면 해당 프롬프트가 입력창에 채워져 그대로 고쳐 쓸 수 있다. 프롬프트 전문은 [vlm/prompts.py](vlm/prompts.py)에 있다.

## 구조

| 파일 | 역할 |
|------|------|
| [app.py](app.py) | Gradio UI, Space 엔트리포인트 |
| [cli.py](cli.py) | 로컬에서 이미지 한 장 처리 |
| [vlm/prompts.py](vlm/prompts.py) | 분류·문서·사진 프롬프트와 그렇게 쓴 이유 |
| [vlm/model.py](vlm/model.py) | 모델 로딩, 종류 판별, 스트리밍 생성 |
| [vlm/image.py](vlm/image.py) | 이미지 로딩, EXIF 회전 보정, 해상도 상·하한 |
| [vlm/runtime.py](vlm/runtime.py) | 디바이스/dtype 판별, ZeroGPU 데코레이터 |

## 동작 방식

프롬프트와 해상도 기본값은 추측이 아니라 **실제로 돌려보고 고른 것이다.** 같은 문서로 조합을 바꿔가며 측정했고, 원문 문자열 35개가 출력에 그대로 들어갔는지로 채점했다.

| 조합 | 결과 |
|------|------|
| 통합 프롬프트(예시 문장 포함) / 원본 해상도 | 65자 — 프롬프트 안 예시를 그대로 복사 |
| 통합 프롬프트(예시 문장 포함) / 2.6배 확대 | 65자 — 위와 바이트 단위로 동일 |
| 문서 전용(예시 없음) / 원본 478K px | 32/35 — 각주급 잔글씨를 놓침 |
| 문서 전용(예시 없음) / 1.2M px로 확대 | **35/35** |
| 문서 전용(예시 없음) / 2배 확대 | 32/35 — 과하게 키우면 되레 나빠짐 |

구축 과정 전체(실패한 배포 두 번, 빗나간 추측, 측정 로그, 포기한 기능)는 [REPORT.md](REPORT.md)에 정리해 두었다.

여기서 나온 규칙:

- **프롬프트에 예시 문장을 넣지 않는다.** 3B는 지시문 안의 구체적인 문장을 출력 틀로 삼아 그대로 복사한다. `'[이미지: 상세 설명]'` 같은 자리표시자도, 내용을 채운 예시도 똑같이 복사됐다.
- **문서/사진을 먼저 분류한다.** 모델이 스스로 분기하게 두면 문서 페이지를 그림 한 장으로 취급해 한 줄만 뱉는다. 종류를 확정해서 "이건 문서다"라고 단정해 주면 전사가 동작한다. 분류는 저해상도로 8토큰만 생성하고, 본 생성과 같은 GPU 세션 안에서 돌기 때문에 ZeroGPU 할당은 한 번이다.
- **작은 문서는 확대하고, 사진은 확대하지 않는다.** ViT 패치 크기가 고정이라 작은 스캔본은 잔글씨가 뭉개진다. 1.2MP까지(최대 2배) 키우면 각주까지 살아난다. 다만 더 키운다고 계속 좋아지지는 않는다.
- 출력 언어를 고정하지 않는다. `그대로 옮겨 적어라`와 충돌해서 비한국어 문서를 번역해 버린다.
- 생성은 greedy(`do_sample=False`) + `repetition_penalty=1.05`. 전사는 창의성이 필요 없고 같은 줄 반복만 막으면 된다.
- 결과는 토큰 단위로 스트리밍된다.

## 로컬 실행

```bash
pip install -r requirements.txt

python app.py                      # Gradio UI (http://127.0.0.1:7860)
python cli.py ./sample.png         # CLI 한 장 (자동 판별)
python cli.py ./scan.jpg --mode document
python cli.py ./sample.png --out result.txt
python cli.py https://example.com/page.jpg --max-new-tokens 3072
```

CUDA가 없으면 Apple Silicon(MPS) → CPU 순으로 떨어진다. CPU는 동작은 하지만 매우 느리다.

## API로 호출하기

Space가 뜨면 Gradio가 `/describe` 엔드포인트를 자동으로 노출한다. 별도 FastAPI 서버는 필요 없다.

```python
from gradio_client import Client, handle_file

client = Client("<user>/<space>")
result = client.predict(
    image=handle_file("sample.png"),
    mode="auto",        # "auto" / "document" / "photo"
    prompt="",          # 비워두면 모드에 맞는 기본 프롬프트
    max_new_tokens=2048,
    max_pixels=1600000,
    api_name="/describe",
)
print(result)  # (판별된 모드, 결과 텍스트)
```

## 설정

| 환경 변수 | 기본값 | 설명 |
|-----------|--------|------|
| `MODEL_ID` | `Qwen/Qwen2.5-VL-3B-Instruct` | 다른 Qwen2.5-VL 체크포인트로 교체 |

하드웨어는 ZeroGPU 또는 GPU를 권장한다. ZeroGPU에서는 GPU 점유 시간이 함수당 180초로 잡혀 있으며([vlm/model.py](vlm/model.py)의 `GPU_DURATION_SEC`), 긴 문서에서 잘리면 이 값을 늘린다.

## 빌드 메모

`transformers` 5.x에서 `AutoProcessor`는 이미지만 쓰는 경우에도 비디오 서브프로세서를 함께 로드하고, 그 클래스가 **torchvision을 요구한다.** torchvision이 빠지면 기동 시 `Qwen2VLVideoProcessor requires the Torchvision library` ImportError로 Space가 죽는다. [requirements.txt](requirements.txt)에 torch와 torchvision을 함께(버전 고정 없이) 두는 이유다.
