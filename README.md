# 🎨 Stable Diffusion 통합 API 서버 v2.0

RTX 4050 GPU를 활용한 Stable Diffusion 통합 API 서버입니다.
FastAPI, 자동 스케줄링, SQLite 데이터베이스 연동, CORS 지원을 포함합니다.

## ✨ v2.0 주요 업데이트

### 🔒 보안 강화
- **CORS 지원**: 웹 클라이언트에서 API 접근 가능
- **입력값 검증 강화**: Pydantic 필드 검증기로 모든 입력값 엄격 검사
- **환경변수 설정**: 민감한 설정을 환경변수로 관리

### ⚡ 성능 최적화
- **모델 컴포넌트 공유**: Text2Img/Img2Img 파이프라인이 동일 모델 컴포넌트 공유 (메모리 50% 절약)
- **Attention Slicing**: GPU 메모리 효율성 향상
- **자동 GPU 메모리 정리**: 생성 완료 후 자동 캐시 클리어
- **Inference Mode**: 추론 최적화 모드 적용

### 🎯 새로운 기능
- **8가지 스타일 프리셋**: realistic, anime, oil_painting, watercolor, cyberpunk, fantasy, minimalist, vintage
- **이미지 다운로드 API**: `/download/{image_id}`
- **이미지 삭제 API**: `/history/{image_id}` (DELETE)
- **페이지네이션**: 히스토리 조회 시 offset/limit 지원
- **Seed 반환**: 재현성을 위한 실제 사용된 seed 반환
- **상세 통계**: 스타일별 통계, GPU 정보 포함

### 📋 코드 품질
- **Modern FastAPI Lifespan**: deprecated startup/shutdown 이벤트 대체
- **로깅 시스템**: 파일 + 콘솔 동시 로깅
- **타입 힌트 강화**: 모든 함수와 모델에 명시적 타입

## 🚀 빠른 시작

### 필수 요구사항

- **GPU**: NVIDIA GPU (CUDA 지원)
- **Python**: 3.10.11
- **CUDA**: 12.1 이상 권장
- **VRAM**: 최소 4GB (RTX 4050 이상)

### 설치

1. **저장소 클론**
```bash
git clone https://github.com/sechan9999/img2img.git
cd img2img
```

2. **가상환경 생성 및 활성화**
```bash
# Windows
python -m venv stable-diffusion-env
.\stable-diffusion-env\Scripts\activate

# Linux/Mac
python -m venv stable-diffusion-env
source stable-diffusion-env/bin/activate
```

3. **의존성 설치**
```bash
pip install -r requirements.txt
```

### 환경변수 설정 (선택사항)

```bash
# .env 파일 또는 환경변수로 설정
SD_MODEL_ID=runwayml/stable-diffusion-v1-5  # 사용할 모델
SD_HOST=0.0.0.0                               # 호스트 주소
SD_PORT=8000                                  # 포트 번호
SD_DB_PATH=sqlite:///sd_database.db          # 데이터베이스 경로
SD_BATCH_HOUR=2                               # 배치 실행 시간 (24시간)
SD_MAX_STEPS=100                              # 최대 inference steps
SD_MAX_DIMENSION=1024                         # 최대 이미지 크기
SD_CORS_ORIGINS=*                             # CORS 허용 도메인 (쉼표로 구분)
```

### 실행

```bash
python integrated_sd_server.py
```

서버가 시작되면 다음 주소로 접속:
- **API 문서 (Swagger)**: http://localhost:8000/docs
- **API 문서 (ReDoc)**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## 📚 API 사용 예제

### Text-to-Image 생성 (스타일 적용)

```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a beautiful sunset over mountains",
    "style": "oil_painting",
    "steps": 50,
    "guidance_scale": 7.5
  }'
```

### PowerShell 예제

```powershell
$body = @{
    prompt = "cyberpunk city at night"
    style = "cyberpunk"
    steps = 50
    guidance_scale = 7.5
    seed = 42
} | ConvertTo-Json

$response = Invoke-RestMethod -Uri "http://localhost:8000/generate" -Method POST -Body $body -ContentType "application/json"

# 이미지 저장
[System.IO.File]::WriteAllBytes("output.png", [System.Convert]::FromBase64String($response.image_base64))
```

### Image-to-Image 변환

```bash
curl -X POST "http://localhost:8000/img2img" \
  -F "file=@input.png" \
  -F "prompt=transform to watercolor painting" \
  -F "style=watercolor" \
  -F "strength=0.75"
```

### 이미지 다운로드

```bash
curl -O "http://localhost:8000/download/1"
```

### 이미지 삭제 (파일 포함)

```bash
curl -X DELETE "http://localhost:8000/history/1?delete_file=true"
```

## 📊 API 엔드포인트

| 엔드포인트 | 메서드 | 설명 |
|-----------|--------|------|
| `/` | GET | API 정보 |
| `/health` | GET | 서버 상태 확인 (GPU 정보, 업타임) |
| `/generate` | POST | Text-to-Image 생성 |
| `/img2img` | POST | Image-to-Image 변환 |
| `/styles` | GET | 사용 가능한 스타일 목록 |
| `/history` | GET | 생성 히스토리 조회 (페이지네이션) |
| `/stats` | GET | 통계 정보 |
| `/download/{id}` | GET | 이미지 다운로드 |
| `/history/{id}` | DELETE | 이미지 삭제 |

## 🎨 스타일 프리셋

| 스타일 | 설명 |
|--------|------|
| `realistic` | 사실적인 사진 스타일 |
| `anime` | 애니메이션/스튜디오 지브리 스타일 |
| `oil_painting` | 클래식 유화 스타일 |
| `watercolor` | 부드러운 수채화 스타일 |
| `cyberpunk` | 네온 사이버펑크 스타일 |
| `fantasy` | 에픽 판타지 아트 스타일 |
| `minimalist` | 미니멀리스트 디자인 (v2.0 신규) |
| `vintage` | 빈티지/레트로 스타일 (v2.0 신규) |

## ⚙️ 설정

### 스케줄 변경

환경변수로 배치 실행 시간을 변경할 수 있습니다:

```bash
# 오후 6시로 변경
SD_BATCH_HOUR=18 python integrated_sd_server.py
```

또는 코드에서 직접 수정:

```python
# integrated_sd_server.py
BATCH_HOUR = int(os.getenv("SD_BATCH_HOUR", "18"))
```

## 🗂️ 프로젝트 구조

```
img2img/
├── integrated_sd_server.py    # 메인 서버 파일 (v2.0)
├── requirements.txt            # Python 의존성
├── README.md                   # 프로젝트 문서
├── .gitignore                  # Git 제외 파일
├── sd_server.log              # 서버 로그 (자동 생성)
├── generated_images/           # API 생성 이미지 (git 제외)
├── batch_generated/            # 배치 생성 이미지 (git 제외)
└── sd_database.db              # SQLite DB (git 제외)
```

## 🔒 보안 주의사항

- **API 키**: 환경변수로 관리하세요
- **CORS**: 프로덕션 환경에서는 특정 도메인만 허용
  ```bash
  SD_CORS_ORIGINS=https://example.com,https://app.example.com
  ```
- **외부 노출**: 프로덕션 환경에서는 인증 미들웨어 추가 권장
- **데이터베이스**: 민감 정보 저장 시 암호화 권장

## 🛠️ 개발 환경

- **OS**: Windows 11
- **GPU**: NVIDIA GeForce RTX 4050 Laptop GPU
- **Python**: 3.10.11
- **CUDA**: 12.1
- **Framework**: FastAPI 0.128+, Diffusers 0.36+, PyTorch 2.5+

## 📝 변경 로그

### v2.0.0 (2026-02-08)
- CORS 미들웨어 추가
- 환경변수 설정 지원
- 8가지 스타일 프리셋 (minimalist, vintage 추가)
- 이미지 다운로드/삭제 API 추가
- 입력값 검증 강화 (Pydantic field validators)
- 모델 컴포넌트 공유로 메모리 최적화
- 로깅 시스템 추가 (파일 + 콘솔)
- Modern FastAPI Lifespan 패턴 적용
- Seed 반환으로 재현성 보장
- 페이지네이션 지원

### v1.0.0 (2026-02-06)
- 초기 릴리즈
- Text-to-Image, Image-to-Image API
- 6가지 스타일 프리셋
- SQLite 데이터베이스 연동
- APScheduler 배치 생성

## 📝 라이선스

MIT License

## 👤 작성자

**sechan9999**
- GitHub: [@sechan9999](https://github.com/sechan9999)

## 🙏 감사의 말

- [Stable Diffusion](https://github.com/Stability-AI/stablediffusion) - AI 이미지 생성 모델
- [Hugging Face Diffusers](https://github.com/huggingface/diffusers) - 확산 모델 라이브러리
- [FastAPI](https://fastapi.tiangolo.com/) - 현대적인 웹 프레임워크
