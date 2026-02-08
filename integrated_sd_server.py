"""
통합 Stable Diffusion 서버 v2.0
- FastAPI REST API
- CORS 지원
- 자동 배치 생성 스케줄링
- SQLite 데이터베이스 연동
- 메모리 최적화
- 로깅 시스템
- 환경변수 설정 지원
"""

import os
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from io import BytesIO
from typing import Optional, List
from pathlib import Path

import base64
import torch
import uvicorn
from PIL import Image

from fastapi import FastAPI, HTTPException, UploadFile, File, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator

# Diffusers
from diffusers import (
    DiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
)

# Scheduling
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

# Database
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, func
from sqlalchemy.orm import declarative_base, sessionmaker, Session

# ============================================================================
# 환경 설정
# ============================================================================

# 환경 변수 (기본값 포함)
MODEL_ID = os.getenv("SD_MODEL_ID", "runwayml/stable-diffusion-v1-5")
HOST = os.getenv("SD_HOST", "0.0.0.0")
PORT = int(os.getenv("SD_PORT", "8000"))
DB_PATH = os.getenv("SD_DB_PATH", "sqlite:///sd_database.db")
BATCH_HOUR = int(os.getenv("SD_BATCH_HOUR", "2"))  # 배치 실행 시간 (24시간 형식)
MAX_STEPS = int(os.getenv("SD_MAX_STEPS", "100"))  # 최대 inference steps
MAX_DIMENSION = int(os.getenv("SD_MAX_DIMENSION", "1024"))  # 최대 이미지 크기
CORS_ORIGINS = os.getenv("SD_CORS_ORIGINS", "*").split(",")

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sd_server.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 출력 디렉토리
OUTPUT_DIR = Path("generated_images")
BATCH_DIR = Path("batch_generated")
OUTPUT_DIR.mkdir(exist_ok=True)
BATCH_DIR.mkdir(exist_ok=True)

# ============================================================================
# 데이터베이스 설정
# ============================================================================

Base = declarative_base()


class GeneratedImage(Base):
    """생성된 이미지 메타데이터 테이블"""
    __tablename__ = 'generated_images'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    prompt = Column(Text, nullable=False)
    negative_prompt = Column(Text, default="")
    steps = Column(Integer, default=50)
    guidance_scale = Column(Float, default=7.5)
    width = Column(Integer, default=512)
    height = Column(Integer, default=512)
    seed = Column(Integer, nullable=True)
    style = Column(String(50), nullable=True)  # 새로 추가: 스타일 이름
    model_type = Column(String(50))  # text2img, img2img, batch
    filename = Column(String(255))
    generation_time = Column(Float)  # 생성 소요 시간 (초)
    created_at = Column(DateTime, default=datetime.now)
    batch_id = Column(String(50), nullable=True)  # 배치 생성 시 그룹 ID


# SQLite 데이터베이스 생성
engine = create_engine(DB_PATH, echo=False, pool_pre_ping=True)
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def migrate_database():
    """기존 데이터베이스에 누락된 컬럼 추가 (마이그레이션)"""
    from sqlalchemy import inspect, text
    
    inspector = inspect(engine)
    
    if 'generated_images' in inspector.get_table_names():
        columns = [col['name'] for col in inspector.get_columns('generated_images')]
        
        with engine.connect() as conn:
            # style 컬럼이 없으면 추가
            if 'style' not in columns:
                try:
                    conn.execute(text("ALTER TABLE generated_images ADD COLUMN style VARCHAR(50)"))
                    conn.commit()
                    logger.info("✅ 마이그레이션: 'style' 컬럼 추가 완료")
                except Exception as e:
                    logger.warning(f"마이그레이션 스킵 (이미 존재할 수 있음): {e}")


# 마이그레이션 실행
migrate_database()


def get_db():
    """데이터베이스 세션 의존성"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ============================================================================
# 스타일 프리셋 (상수로 분리)
# ============================================================================

STYLE_PRESETS = {
    "realistic": {
        "name": "realistic",
        "description": "사실적인 사진 스타일",
        "prompt_suffix": "photorealistic, highly detailed, 4k, professional photography, sharp focus",
        "negative_prompt": "cartoon, anime, painting, drawing, blurry, low quality"
    },
    "anime": {
        "name": "anime",
        "description": "애니메이션 스타일",
        "prompt_suffix": "anime style, studio ghibli, vibrant colors, detailed, cel shading",
        "negative_prompt": "photorealistic, 3d render, low quality"
    },
    "oil_painting": {
        "name": "oil_painting",
        "description": "유화 스타일",
        "prompt_suffix": "oil painting, classical art style, artistic, detailed brushwork, masterpiece",
        "negative_prompt": "photo, digital art, low quality"
    },
    "watercolor": {
        "name": "watercolor",
        "description": "수채화 스타일",
        "prompt_suffix": "watercolor painting, soft colors, artistic, flowing, delicate",
        "negative_prompt": "photo, digital art, harsh colors, low quality"
    },
    "cyberpunk": {
        "name": "cyberpunk",
        "description": "사이버펑크 스타일",
        "prompt_suffix": "cyberpunk style, neon lights, futuristic, high tech, dystopian, rain",
        "negative_prompt": "natural, pastoral, low quality"
    },
    "fantasy": {
        "name": "fantasy",
        "description": "판타지 스타일",
        "prompt_suffix": "fantasy art, magical, epic, detailed, concept art, ethereal lighting",
        "negative_prompt": "modern, realistic, photo, low quality"
    },
    "minimalist": {
        "name": "minimalist",
        "description": "미니멀리스트 스타일",
        "prompt_suffix": "minimalist design, clean lines, simple composition, modern art",
        "negative_prompt": "cluttered, busy, detailed, complex, low quality"
    },
    "vintage": {
        "name": "vintage",
        "description": "빈티지/레트로 스타일",
        "prompt_suffix": "vintage style, retro aesthetic, film grain, warm tones, nostalgic",
        "negative_prompt": "modern, digital, clean, low quality"
    }
}

# ============================================================================
# Pydantic 모델 (API 요청/응답) - 검증 강화
# ============================================================================


class Text2ImgRequest(BaseModel):
    """Text-to-Image 요청 모델"""
    prompt: str = Field(..., min_length=1, max_length=2000, description="이미지 생성 프롬프트")
    negative_prompt: Optional[str] = Field("", max_length=2000, description="네거티브 프롬프트")
    style: Optional[str] = Field(None, description="스타일 프리셋 이름")
    steps: int = Field(50, ge=1, le=MAX_STEPS, description=f"Inference steps (1-{MAX_STEPS})")
    guidance_scale: float = Field(7.5, ge=1.0, le=20.0, description="Guidance scale (1.0-20.0)")
    width: int = Field(512, ge=256, le=MAX_DIMENSION, description=f"이미지 너비 (256-{MAX_DIMENSION})")
    height: int = Field(512, ge=256, le=MAX_DIMENSION, description=f"이미지 높이 (256-{MAX_DIMENSION})")
    seed: Optional[int] = Field(None, ge=0, description="랜덤 시드 (재현성)")
    
    @field_validator('width', 'height')
    @classmethod
    def validate_dimensions(cls, v):
        """이미지 크기는 8의 배수여야 함"""
        if v % 8 != 0:
            raise ValueError(f"이미지 크기는 8의 배수여야 합니다 (입력값: {v})")
        return v
    
    @field_validator('style')
    @classmethod
    def validate_style(cls, v):
        """스타일 이름 검증 (빈 문자열은 None으로 처리)"""
        if v is None or v == "" or v.strip() == "":
            return None
        if v not in STYLE_PRESETS:
            raise ValueError(f"유효하지 않은 스타일입니다. 사용 가능: {list(STYLE_PRESETS.keys())}")
        return v


class Img2ImgRequest(BaseModel):
    """Image-to-Image 요청 모델"""
    prompt: str = Field(..., min_length=1, max_length=2000, description="변환 프롬프트")
    negative_prompt: Optional[str] = Field("", max_length=2000, description="네거티브 프롬프트")
    style: Optional[str] = Field(None, description="스타일 프리셋 이름")
    strength: float = Field(0.75, ge=0.1, le=1.0, description="변환 강도 (0.1-1.0)")
    steps: int = Field(50, ge=1, le=MAX_STEPS, description=f"Inference steps (1-{MAX_STEPS})")
    guidance_scale: float = Field(7.5, ge=1.0, le=20.0, description="Guidance scale")
    seed: Optional[int] = Field(None, ge=0, description="랜덤 시드")
    
    @field_validator('style')
    @classmethod
    def validate_style(cls, v):
        """스타일 이름 검증 (빈 문자열은 None으로 처리)"""
        if v is None or v == "" or v.strip() == "":
            return None
        if v not in STYLE_PRESETS:
            raise ValueError(f"유효하지 않은 스타일입니다. 사용 가능: {list(STYLE_PRESETS.keys())}")
        return v


class ImageResponse(BaseModel):
    """이미지 생성 응답 모델"""
    id: int
    image_base64: str
    prompt: str
    style: Optional[str] = None
    filename: str
    generation_time: float
    seed: Optional[int] = None


class StyleInfo(BaseModel):
    """스타일 정보 모델"""
    name: str
    description: str
    prompt_suffix: str
    negative_prompt: str


class ImageHistory(BaseModel):
    """이미지 히스토리 모델"""
    id: int
    prompt: str
    style: Optional[str] = None
    steps: int
    guidance_scale: float
    model_type: str
    filename: str
    generation_time: float
    created_at: datetime
    
    class Config:
        from_attributes = True


class StatsResponse(BaseModel):
    """통계 응답 모델"""
    total_images: int
    by_type: dict
    by_style: dict
    avg_generation_time_sec: float
    last_24h_count: int
    gpu_info: dict


class HealthResponse(BaseModel):
    """헬스체크 응답 모델"""
    status: str
    version: str
    gpu_available: bool
    gpu_name: Optional[str]
    gpu_memory_gb: Optional[float]
    total_images_generated: int
    uptime_seconds: float


# ============================================================================
# 글로벌 변수
# ============================================================================

pipe_text2img = None
pipe_img2img = None
scheduler = None
server_start_time = None


# ============================================================================
# 헬퍼 함수
# ============================================================================

def save_to_database(
    db: Session,
    prompt: str,
    params: dict,
    filename: str,
    generation_time: float,
    model_type: str = "text2img",
    batch_id: str = None,
    style: str = None
) -> int:
    """데이터베이스에 이미지 정보 저장"""
    try:
        db_image = GeneratedImage(
            prompt=prompt,
            negative_prompt=params.get("negative_prompt", ""),
            steps=params.get("steps", 50),
            guidance_scale=params.get("guidance_scale", 7.5),
            width=params.get("width", 512),
            height=params.get("height", 512),
            seed=params.get("seed"),
            style=style,
            model_type=model_type,
            filename=filename,
            generation_time=generation_time,
            batch_id=batch_id
        )
        db.add(db_image)
        db.commit()
        db.refresh(db_image)
        logger.info(f"이미지 저장 완료: ID={db_image.id}, 파일={filename}")
        return db_image.id
    except Exception as e:
        db.rollback()
        logger.error(f"데이터베이스 저장 실패: {e}")
        raise


def image_to_base64(image: Image.Image) -> str:
    """PIL Image를 Base64 문자열로 변환"""
    buffered = BytesIO()
    image.save(buffered, format="PNG", optimize=True)
    return base64.b64encode(buffered.getvalue()).decode()


def apply_style(prompt: str, negative_prompt: str, style_name: Optional[str]) -> tuple:
    """스타일 프리셋을 프롬프트에 적용"""
    if style_name and style_name in STYLE_PRESETS:
        style = STYLE_PRESETS[style_name]
        enhanced_prompt = f"{prompt}, {style['prompt_suffix']}"
        enhanced_negative = f"{negative_prompt}, {style['negative_prompt']}" if negative_prompt else style['negative_prompt']
        return enhanced_prompt, enhanced_negative
    return prompt, negative_prompt


def clear_gpu_memory():
    """GPU 메모리 정리"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ============================================================================
# Lifespan 관리 (startup/shutdown 대체)
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 수명 주기 관리"""
    global pipe_text2img, pipe_img2img, scheduler, server_start_time
    
    # Startup
    logger.info("=" * 60)
    logger.info("🚀 통합 Stable Diffusion 서버 v2.0 시작")
    logger.info("=" * 60)
    
    server_start_time = datetime.now()
    
    logger.info("📦 모델 로딩 중...")
    
    try:
        # Text-to-Image 파이프라인 (메모리 최적화)
        pipe_text2img = DiffusionPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            safety_checker=None,  # 메모리 절약
            requires_safety_checker=False
        ).to("cuda")
        
        # 메모리 효율적인 attention 활성화
        if hasattr(pipe_text2img, 'enable_attention_slicing'):
            pipe_text2img.enable_attention_slicing()
        
        # Image-to-Image 파이프라인 (컴포넌트 공유로 메모리 절약)
        pipe_img2img = StableDiffusionImg2ImgPipeline(
            vae=pipe_text2img.vae,
            text_encoder=pipe_text2img.text_encoder,
            tokenizer=pipe_text2img.tokenizer,
            unet=pipe_text2img.unet,
            scheduler=pipe_text2img.scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False
        ).to("cuda")
        
        logger.info("✅ 모델 로딩 완료!")
        
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"🎮 GPU: {torch.cuda.get_device_name(0)} ({gpu_mem:.1f}GB)")
    
    except Exception as e:
        logger.error(f"❌ 모델 로딩 실패: {e}")
        raise
    
    # 스케줄러 설정
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        scheduled_batch_generation,
        trigger=CronTrigger(hour=BATCH_HOUR, minute=0),
        id='daily_batch_generation',
        name='매일 자동 배치 이미지 생성',
        misfire_grace_time=3600  # 1시간 내 실행 허용
    )
    scheduler.start()
    logger.info(f"⏰ 스케줄러 시작됨: 매일 {BATCH_HOUR}시 자동 배치 생성")
    
    logger.info("=" * 60)
    logger.info("✅ 서버 준비 완료!")
    logger.info(f"📍 API 문서: http://localhost:{PORT}/docs")
    logger.info(f"📍 Health Check: http://localhost:{PORT}/health")
    logger.info("=" * 60)
    
    yield  # 서버 실행
    
    # Shutdown
    logger.info("⏹️ 서버 종료 중...")
    if scheduler:
        scheduler.shutdown(wait=False)
    clear_gpu_memory()
    logger.info("✅ 서버 종료 완료")


# ============================================================================
# FastAPI 앱 초기화
# ============================================================================

app = FastAPI(
    title="통합 Stable Diffusion API",
    description="""
## 🎨 Stable Diffusion 이미지 생성 API

RTX 4050 GPU를 활용한 고품질 이미지 생성 서버입니다.

### 주요 기능
- **Text-to-Image**: 텍스트 프롬프트로 이미지 생성
- **Image-to-Image**: 기존 이미지를 다른 스타일로 변환
- **8가지 스타일 프리셋**: realistic, anime, oil_painting, watercolor, cyberpunk, fantasy, minimalist, vintage
- **히스토리 조회**: 생성된 이미지 기록 확인
- **통계 대시보드**: 생성 통계 및 성능 지표
    """,
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS 미들웨어 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# API 엔드포인트
# ============================================================================

@app.get("/", include_in_schema=False)
def root_redirect():
    """루트 경로: 웹 UI로 리다이렉트"""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/ui")


@app.get("/ui", include_in_schema=False)
def serve_ui():
    """웹 UI 제공"""
    from fastapi.responses import HTMLResponse
    ui_path = Path(__file__).parent / "ui.html"
    if ui_path.exists():
        return HTMLResponse(content=ui_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>UI 파일을 찾을 수 없습니다</h1>", status_code=404)


@app.get("/api", tags=["Info"])
def api_info():
    """API 정보 및 엔드포인트 목록"""
    return {
        "message": "통합 Stable Diffusion API 서버",
        "version": "2.0.0",
        "ui": "/ui",
        "documentation": "/docs",
        "endpoints": {
            "ui": "/ui (웹 인터페이스)",
            "docs": "/docs",
            "redoc": "/redoc",
            "health": "/health",
            "generate": "/generate (POST)",
            "img2img": "/img2img (POST)",
            "styles": "/styles",
            "history": "/history",
            "stats": "/stats",
            "download": "/download/{image_id}"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
def health_check(db: Session = Depends(get_db)):
    """서버 상태 확인"""
    try:
        total_images = db.query(GeneratedImage).count()
    except Exception as e:
        logger.warning(f"DB 쿼리 실패: {e}")
        total_images = 0
    
    gpu_name = None
    gpu_memory = None
    gpu_available = False
    
    try:
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
    except Exception as e:
        logger.warning(f"GPU 정보 조회 실패: {e}")
    
    try:
        uptime = (datetime.now() - server_start_time).total_seconds() if server_start_time else 0
    except Exception:
        uptime = 0
    
    return HealthResponse(
        status="healthy",
        version="2.0.0",
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        gpu_memory_gb=gpu_memory,
        total_images_generated=total_images,
        uptime_seconds=round(uptime, 2)
    )


@app.post("/generate", response_model=ImageResponse, tags=["Generation"])
def generate_text2img(request: Text2ImgRequest, db: Session = Depends(get_db)):
    """Text-to-Image 생성"""
    start_time = datetime.now()
    
    try:
        # 스타일 적용
        enhanced_prompt, enhanced_negative = apply_style(
            request.prompt, 
            request.negative_prompt or "", 
            request.style
        )
        
        # Generator 설정 (seed)
        generator = None
        actual_seed = request.seed
        if actual_seed is None:
            actual_seed = torch.randint(0, 2**32 - 1, (1,)).item()
        generator = torch.Generator("cuda").manual_seed(actual_seed)
        
        # 이미지 생성
        with torch.inference_mode():
            image = pipe_text2img(
                prompt=enhanced_prompt,
                negative_prompt=enhanced_negative,
                num_inference_steps=request.steps,
                guidance_scale=request.guidance_scale,
                width=request.width,
                height=request.height,
                generator=generator
            ).images[0]
        
        # 생성 시간 계산
        generation_time = (datetime.now() - start_time).total_seconds()
        
        # 파일 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = str(OUTPUT_DIR / f"text2img_{timestamp}.png")
        image.save(filename, optimize=True)
        
        # DB 저장
        params = request.model_dump()
        params["seed"] = actual_seed
        db_id = save_to_database(
            db, request.prompt, params, filename, 
            generation_time, "text2img", style=request.style
        )
        
        # Base64 인코딩
        img_base64 = image_to_base64(image)
        
        # GPU 메모리 정리
        clear_gpu_memory()
        
        logger.info(f"Text2Img 생성 완료: {generation_time:.2f}초, seed={actual_seed}")
        
        return ImageResponse(
            id=db_id,
            image_base64=img_base64,
            prompt=request.prompt,
            style=request.style,
            filename=filename,
            generation_time=round(generation_time, 2),
            seed=actual_seed
        )
    
    except Exception as e:
        logger.error(f"Text2Img 생성 실패: {e}")
        clear_gpu_memory()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/img2img", response_model=ImageResponse, tags=["Generation"])
async def generate_img2img(
    file: UploadFile = File(..., description="입력 이미지 파일"),
    prompt: str = Query("beautiful artwork", description="변환 프롬프트"),
    negative_prompt: str = Query("", description="네거티브 프롬프트"),
    style: Optional[str] = Query(None, description="스타일 프리셋"),
    strength: float = Query(0.75, ge=0.1, le=1.0, description="변환 강도"),
    steps: int = Query(50, ge=1, le=MAX_STEPS, description="Inference steps"),
    guidance_scale: float = Query(7.5, ge=1.0, le=20.0, description="Guidance scale"),
    seed: Optional[int] = Query(None, ge=0, description="랜덤 시드"),
    db: Session = Depends(get_db)
):
    """Image-to-Image 변환"""
    start_time = datetime.now()
    
    # 스타일 검증 (빈 문자열은 None으로 처리)
    if style is not None and style.strip() == "":
        style = None
    if style is not None and style not in STYLE_PRESETS:
        raise HTTPException(
            status_code=400, 
            detail=f"유효하지 않은 스타일입니다. 사용 가능: {list(STYLE_PRESETS.keys())}"
        )
    
    # 파일 타입 검증
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다.")
    
    try:
        # 업로드된 이미지 읽기
        contents = await file.read()
        init_image = Image.open(BytesIO(contents)).convert("RGB")
        init_image = init_image.resize((512, 512), Image.Resampling.LANCZOS)
        
        # 스타일 적용
        enhanced_prompt, enhanced_negative = apply_style(prompt, negative_prompt, style)
        
        # Generator 설정
        generator = None
        actual_seed = seed
        if actual_seed is None:
            actual_seed = torch.randint(0, 2**32 - 1, (1,)).item()
        generator = torch.Generator("cuda").manual_seed(actual_seed)
        
        # 이미지 생성
        with torch.inference_mode():
            result_image = pipe_img2img(
                prompt=enhanced_prompt,
                negative_prompt=enhanced_negative,
                image=init_image,
                strength=strength,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator
            ).images[0]
        
        # 생성 시간 계산
        generation_time = (datetime.now() - start_time).total_seconds()
        
        # 파일 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = str(OUTPUT_DIR / f"img2img_{timestamp}.png")
        result_image.save(filename, optimize=True)
        
        # DB 저장
        params = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "strength": strength,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "seed": actual_seed
        }
        db_id = save_to_database(
            db, prompt, params, filename, 
            generation_time, "img2img", style=style
        )
        
        # Base64 인코딩
        img_base64 = image_to_base64(result_image)
        
        # GPU 메모리 정리
        clear_gpu_memory()
        
        logger.info(f"Img2Img 생성 완료: {generation_time:.2f}초, seed={actual_seed}")
        
        return ImageResponse(
            id=db_id,
            image_base64=img_base64,
            prompt=prompt,
            style=style,
            filename=filename,
            generation_time=round(generation_time, 2),
            seed=actual_seed
        )
    
    except Exception as e:
        logger.error(f"Img2Img 생성 실패: {e}")
        clear_gpu_memory()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/styles", response_model=List[StyleInfo], tags=["Info"])
def get_styles():
    """사용 가능한 스타일 목록"""
    return [StyleInfo(**style) for style in STYLE_PRESETS.values()]


@app.get("/history", response_model=List[ImageHistory], tags=["History"])
def get_history(
    limit: int = Query(20, ge=1, le=100, description="조회 개수"),
    offset: int = Query(0, ge=0, description="시작 위치"),
    model_type: Optional[str] = Query(None, description="모델 타입 필터"),
    style: Optional[str] = Query(None, description="스타일 필터"),
    db: Session = Depends(get_db)
):
    """생성 히스토리 조회 (페이지네이션 지원)"""
    query = db.query(GeneratedImage)
    
    if model_type:
        query = query.filter(GeneratedImage.model_type == model_type)
    if style:
        query = query.filter(GeneratedImage.style == style)
    
    images = query.order_by(GeneratedImage.created_at.desc()).offset(offset).limit(limit).all()
    
    return [ImageHistory.model_validate(img) for img in images]


@app.get("/stats", response_model=StatsResponse, tags=["Info"])
def get_statistics(db: Session = Depends(get_db)):
    """통계 정보"""
    total_images = db.query(GeneratedImage).count()
    
    # 모델 타입별 통계
    type_stats = {}
    for model_type in ["text2img", "img2img", "batch"]:
        count = db.query(GeneratedImage).filter(
            GeneratedImage.model_type == model_type
        ).count()
        type_stats[model_type] = count
    
    # 스타일별 통계
    style_stats = {}
    for style_name in STYLE_PRESETS.keys():
        count = db.query(GeneratedImage).filter(
            GeneratedImage.style == style_name
        ).count()
        if count > 0:
            style_stats[style_name] = count
    
    # null 스타일 (스타일 미사용)
    no_style_count = db.query(GeneratedImage).filter(
        GeneratedImage.style.is_(None)
    ).count()
    if no_style_count > 0:
        style_stats["none"] = no_style_count
    
    # 평균 생성 시간
    avg_time = db.query(func.avg(GeneratedImage.generation_time)).scalar()
    
    # 최근 24시간 생성 수
    yesterday = datetime.now() - timedelta(days=1)
    recent_count = db.query(GeneratedImage).filter(
        GeneratedImage.created_at >= yesterday
    ).count()
    
    # GPU 정보
    gpu_info = {
        "available": torch.cuda.is_available(),
        "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "memory_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2) if torch.cuda.is_available() else None
    }
    
    return StatsResponse(
        total_images=total_images,
        by_type=type_stats,
        by_style=style_stats,
        avg_generation_time_sec=round(avg_time, 2) if avg_time else 0,
        last_24h_count=recent_count,
        gpu_info=gpu_info
    )


@app.get("/download/{image_id}", tags=["History"])
def download_image(image_id: int, db: Session = Depends(get_db)):
    """이미지 다운로드"""
    image = db.query(GeneratedImage).filter(GeneratedImage.id == image_id).first()
    
    if not image:
        raise HTTPException(status_code=404, detail="이미지를 찾을 수 없습니다.")
    
    if not Path(image.filename).exists():
        raise HTTPException(status_code=404, detail="이미지 파일이 존재하지 않습니다.")
    
    return FileResponse(
        image.filename,
        media_type="image/png",
        filename=f"sd_image_{image_id}.png"
    )


@app.delete("/history/{image_id}", tags=["History"])
def delete_image(image_id: int, delete_file: bool = Query(False, description="파일도 삭제"), db: Session = Depends(get_db)):
    """이미지 기록 삭제"""
    image = db.query(GeneratedImage).filter(GeneratedImage.id == image_id).first()
    
    if not image:
        raise HTTPException(status_code=404, detail="이미지를 찾을 수 없습니다.")
    
    filename = image.filename
    
    db.delete(image)
    db.commit()
    
    if delete_file and Path(filename).exists():
        try:
            Path(filename).unlink()
            logger.info(f"파일 삭제: {filename}")
        except Exception as e:
            logger.warning(f"파일 삭제 실패: {e}")
    
    return {"message": f"이미지 {image_id} 삭제 완료", "file_deleted": delete_file}


# ============================================================================
# 배치 생성 함수
# ============================================================================

def scheduled_batch_generation():
    """스케줄된 배치 이미지 생성"""
    logger.info("=" * 60)
    logger.info(f"🤖 자동 배치 생성 시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
    
    batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 배치 프롬프트 (다양한 스타일 포함)
    prompts = [
        {"prompt": "serene mountain landscape at sunrise", "style": "realistic"},
        {"prompt": "cyberpunk city street at night", "style": "cyberpunk"},
        {"prompt": "cute cat playing with yarn", "style": "anime"},
        {"prompt": "fantasy castle on floating island", "style": "fantasy"},
        {"prompt": "abstract colorful geometric art", "style": "minimalist"},
    ]
    
    db = SessionLocal()
    
    try:
        for idx, item in enumerate(prompts, 1):
            try:
                prompt = item["prompt"]
                style = item.get("style")
                
                logger.info(f"[{idx}/{len(prompts)}] 생성 중: {prompt[:50]}... (스타일: {style})")
                
                start_time = datetime.now()
                
                # 스타일 적용
                enhanced_prompt, enhanced_negative = apply_style(prompt, "", style)
                
                # 이미지 생성
                with torch.inference_mode():
                    image = pipe_text2img(
                        prompt=enhanced_prompt,
                        negative_prompt=enhanced_negative,
                        num_inference_steps=30,
                        guidance_scale=7.5
                    ).images[0]
                
                generation_time = (datetime.now() - start_time).total_seconds()
                
                # 저장
                filename = str(BATCH_DIR / f"batch_{batch_id}_{idx:02d}.png")
                image.save(filename, optimize=True)
                
                # DB 저장
                params = {"steps": 30, "guidance_scale": 7.5, "width": 512, "height": 512}
                save_to_database(db, prompt, params, filename, generation_time, "batch", batch_id, style)
                
                logger.info(f"  ✅ 완료 ({generation_time:.2f}초) - {filename}")
                
                # 메모리 정리
                clear_gpu_memory()
            
            except Exception as e:
                logger.error(f"  ❌ 에러: {e}")
    
    finally:
        db.close()
    
    logger.info(f"✅ 배치 생성 완료! Batch ID: {batch_id}")


# ============================================================================
# 서버 시작
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "integrated_sd_server:app",
        host=HOST,
        port=PORT,
        log_level="info",
        reload=False  # 프로덕션에서는 False
    )
