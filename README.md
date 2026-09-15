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

이미지를 넣으면 다음 규칙대로 결과를 내놓는다.

1. 읽을 수 있는 문장·단락은 요약 없이 원문 그대로 (오탈자·줄바꿈 유지)
2. 사진·삽화·도표·그래프는 `[이미지: 상세 설명]` 형식으로 대체
3. 텍스트 없이 사진 한 장뿐이면 대체 형식 없이 보이는 그대로 상세 서술
4. 추측·해석·감정 묘사 없이 관찰 가능한 사실만
5. 이미지에 나온 순서(위→아래, 왼쪽→오른쪽) 유지

프롬프트 전문은 [vlm/prompts.py](vlm/prompts.py)에 있고, UI의 **고급 설정**에서 직접 수정해 실행할 수 있다.

## 구조

| 파일 | 역할 |
|------|------|
| [app.py](app.py) | Gradio UI, Space 엔트리포인트 |
| [cli.py](cli.py) | 로컬에서 이미지 한 장 전사 |
| [vlm/prompts.py](vlm/prompts.py) | 시스템 프롬프트 + 전사 프롬프트 |
| [vlm/model.py](vlm/model.py) | 모델 로딩, 스트리밍 생성 |
| [vlm/image.py](vlm/image.py) | 이미지 로딩, EXIF 회전 보정, 해상도 상한 |
| [vlm/runtime.py](vlm/runtime.py) | 디바이스/dtype 판별, ZeroGPU 데코레이터 |

## 동작 방식

- 문서/사진을 미리 분류하는 사전 호출 없이 **프롬프트 한 번**으로 처리한다. 규칙 3이 사진 전용 경우를 이미 포함하기 때문에, 분류용 호출을 없애면 GPU 시간이 절반 가까이 줄어든다.
- 입력 이미지는 기본 1.6MP로 축소해서 넣는다. Qwen2.5-VL의 비주얼 토큰은 대략 `픽셀 / 784`개라, 원본 그대로 넣으면 1만 토큰을 넘겨 ZeroGPU 실행 시간을 다 써버린다. 잔글씨가 뭉개지면 슬라이더로 올리면 된다.
- 생성은 greedy(`do_sample=False`) + `repetition_penalty=1.05`. 전사는 창의성이 필요 없고 같은 줄 반복만 막으면 된다.
- 결과는 토큰 단위로 스트리밍된다.

## 로컬 실행

```bash
pip install -r requirements.txt

python app.py                      # Gradio UI (http://127.0.0.1:7860)
python cli.py ./sample.png         # CLI 한 장
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
    prompt="",          # 비워두면 기본 프롬프트
    max_new_tokens=2048,
    max_pixels=1600000,
    api_name="/describe",
)
print(result)
```

## 설정

| 환경 변수 | 기본값 | 설명 |
|-----------|--------|------|
| `MODEL_ID` | `Qwen/Qwen2.5-VL-3B-Instruct` | 다른 Qwen2.5-VL 체크포인트로 교체 |

하드웨어는 ZeroGPU 또는 GPU를 권장한다. ZeroGPU에서는 GPU 점유 시간이 함수당 180초로 잡혀 있으며([vlm/model.py](vlm/model.py)의 `GPU_DURATION_SEC`), 긴 문서에서 잘리면 이 값을 늘린다.
