"""
통합 Stable Diffusion 서버
- FastAPI REST API
- 자동 배치 생성 스케줄링
- SQLite 데이터베이스 연동
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
import uvicorn
import base64
from io import BytesIO
from PIL import Image
import torch
import os

# Diffusers
from diffusers import (
    DiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline
)

# Scheduling
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

# Database
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import json

# ============================================================================
# 데이터베이스 설정
# ============================================================================

Base = declarative_base()

class GeneratedImage(Base):
    """생성된 이미지 메타데이터 테이블"""
    __tablename__ = 'generated_images'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    prompt = Column(Text, nullable=False)
    negative_prompt = Column(Text)
    steps = Column(Integer)
    guidance_scale = Column(Float)
    width = Column(Integer)
    height = Column(Integer)
    seed = Column(Integer)
    model_type = Column(String(50))  # text2img, img2img, inpaint
    filename = Column(String(255))
    generation_time = Column(Float)  # 생성 소요 시간 (초)
    created_at = Column(DateTime, default=datetime.now)
    batch_id = Column(String(50))  # 배치 생성 시 그룹 ID

# SQLite 데이터베이스 생성
engine = create_engine('sqlite:///sd_database.db', echo=False)
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)

# ============================================================================
# FastAPI 앱 초기화
# ============================================================================

app = FastAPI(
    title="통합 Stable Diffusion API",
    description="이미지 생성, 스케줄링, 데이터베이스 연동",
    version="1.0.0"
)

# ============================================================================
# 모델 로딩
# ============================================================================

print("=" * 60)
print("🚀 통합 Stable Diffusion 서버 시작")
print("=" * 60)

# 출력 디렉토리 생성
os.makedirs("generated_images", exist_ok=True)
os.makedirs("batch_generated", exist_ok=True)

print("\n📦 모델 로딩 중...")

# Text-to-Image 파이프라인
pipe_text2img = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

# Image-to-Image 파이프라인
pipe_img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

print("✅ 모델 로딩 완료!")

# ============================================================================
# Pydantic 모델 (API 요청/응답)
# ============================================================================

class Text2ImgRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = ""
    steps: int = 50
    guidance_scale: float = 7.5
    width: int = 512
    height: int = 512
    seed: Optional[int] = None

class Img2ImgRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = ""
    strength: float = 0.75
    steps: int = 50
    guidance_scale: float = 7.5
    seed: Optional[int] = None

class ImageResponse(BaseModel):
    id: int
    image_base64: str
    prompt: str
    filename: str
    generation_time: float

class StyleInfo(BaseModel):
    name: str
    description: str
    prompt_suffix: str

class ImageHistory(BaseModel):
    id: int
    prompt: str
    steps: int
    guidance_scale: float
    model_type: str
    filename: str
    generation_time: float
    created_at: datetime

# ============================================================================
# 헬퍼 함수
# ============================================================================

def save_to_database(prompt: str, params: dict, filename: str, 
                     generation_time: float, model_type: str = "text2img",
                     batch_id: str = None) -> int:
    """데이터베이스에 이미지 정보 저장"""
    db = SessionLocal()
    try:
        db_image = GeneratedImage(
            prompt=prompt,
            negative_prompt=params.get("negative_prompt", ""),
            steps=params.get("steps", 50),
            guidance_scale=params.get("guidance_scale", 7.5),
            width=params.get("width", 512),
            height=params.get("height", 512),
            seed=params.get("seed"),
            model_type=model_type,
            filename=filename,
            generation_time=generation_time,
            batch_id=batch_id
        )
        db.add(db_image)
        db.commit()
        db.refresh(db_image)
        return db_image.id
    finally:
        db.close()

def image_to_base64(image: Image.Image) -> str:
    """PIL Image를 Base64 문자열로 변환"""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# ============================================================================
# API 엔드포인트
# ============================================================================

@app.get("/")
def read_root():
    return {
        "message": "통합 Stable Diffusion API 서버",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "generate": "/generate",
            "img2img": "/img2img",
            "styles": "/styles",
            "history": "/history",
            "stats": "/stats"
        }
    }

@app.get("/health")
def health_check():
    """서버 상태 확인"""
    db = SessionLocal()
    try:
        total_images = db.query(GeneratedImage).count()
    finally:
        db.close()
    
    return {
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "total_images_generated": total_images
    }

@app.post("/generate", response_model=ImageResponse)
def generate_text2img(request: Text2ImgRequest):
    """Text-to-Image 생성"""
    start_time = datetime.now()
    
    try:
        # Generator 설정 (seed)
        generator = None
        if request.seed:
            generator = torch.Generator("cuda").manual_seed(request.seed)
        
        # 이미지 생성
        image = pipe_text2img(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            num_inference_steps=request.steps,
            guidance_scale=request.guidance_scale,
            width=request.width,
            height=request.height,
            generator=generator
        ).images[0]
        
        # 생성 시간 계산
        generation_time = (datetime.now() - start_time).total_seconds()
        
        # 파일 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"generated_images/text2img_{timestamp}.png"
        image.save(filename)
        
        # DB 저장
        params = request.dict()
        db_id = save_to_database(request.prompt, params, filename, generation_time, "text2img")
        
        # Base64 인코딩
        img_base64 = image_to_base64(image)
        
        return ImageResponse(
            id=db_id,
            image_base64=img_base64,
            prompt=request.prompt,
            filename=filename,
            generation_time=generation_time
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/img2img")
async def generate_img2img(
    file: UploadFile = File(...),
    prompt: str = "beautiful artwork",
    strength: float = 0.75,
    steps: int = 50,
    guidance_scale: float = 7.5
):
    """Image-to-Image 변환"""
    start_time = datetime.now()
    
    try:
        # 업로드된 이미지 읽기
        contents = await file.read()
        init_image = Image.open(BytesIO(contents)).convert("RGB")
        init_image = init_image.resize((512, 512))
        
        # 이미지 생성
        result_image = pipe_img2img(
            prompt=prompt,
            image=init_image,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=guidance_scale
        ).images[0]
        
        # 생성 시간 계산
        generation_time = (datetime.now() - start_time).total_seconds()
        
        # 파일 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"generated_images/img2img_{timestamp}.png"
        result_image.save(filename)
        
        # DB 저장
        params = {
            "prompt": prompt,
            "strength": strength,
            "steps": steps,
            "guidance_scale": guidance_scale
        }
        db_id = save_to_database(prompt, params, filename, generation_time, "img2img")
        
        # Base64 인코딩
        img_base64 = image_to_base64(result_image)
        
        return {
            "id": db_id,
            "image_base64": img_base64,
            "prompt": prompt,
            "filename": filename,
            "generation_time": generation_time
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/styles", response_model=List[StyleInfo])
def get_styles():
    """사용 가능한 스타일 목록"""
    styles = [
        StyleInfo(
            name="realistic",
            description="사실적인 사진 스타일",
            prompt_suffix="photorealistic, highly detailed, 4k, professional photography"
        ),
        StyleInfo(
            name="anime",
            description="애니메이션 스타일",
            prompt_suffix="anime style, studio ghibli, vibrant colors, detailed"
        ),
        StyleInfo(
            name="oil_painting",
            description="유화 스타일",
            prompt_suffix="oil painting, classical art style, artistic, detailed brushwork"
        ),
        StyleInfo(
            name="watercolor",
            description="수채화 스타일",
            prompt_suffix="watercolor painting, soft colors, artistic, flowing"
        ),
        StyleInfo(
            name="cyberpunk",
            description="사이버펑크 스타일",
            prompt_suffix="cyberpunk style, neon lights, futuristic, high tech"
        ),
        StyleInfo(
            name="fantasy",
            description="판타지 스타일",
            prompt_suffix="fantasy art, magical, epic, detailed, concept art"
        )
    ]
    return styles

@app.get("/history", response_model=List[ImageHistory])
def get_history(limit: int = 20, model_type: Optional[str] = None):
    """생성 히스토리 조회"""
    db = SessionLocal()
    try:
        query = db.query(GeneratedImage)
        
        if model_type:
            query = query.filter(GeneratedImage.model_type == model_type)
        
        images = query.order_by(GeneratedImage.created_at.desc()).limit(limit).all()
        
        return [
            ImageHistory(
                id=img.id,
                prompt=img.prompt,
                steps=img.steps,
                guidance_scale=img.guidance_scale,
                model_type=img.model_type,
                filename=img.filename,
                generation_time=img.generation_time,
                created_at=img.created_at
            )
            for img in images
        ]
    finally:
        db.close()

@app.get("/stats")
def get_statistics():
    """통계 정보"""
    db = SessionLocal()
    try:
        total_images = db.query(GeneratedImage).count()
        
        # 모델 타입별 통계
        text2img_count = db.query(GeneratedImage).filter(
            GeneratedImage.model_type == "text2img"
        ).count()
        img2img_count = db.query(GeneratedImage).filter(
            GeneratedImage.model_type == "img2img"
        ).count()
        batch_count = db.query(GeneratedImage).filter(
            GeneratedImage.model_type == "batch"
        ).count()
        
        # 평균 생성 시간
        from sqlalchemy import func
        avg_time = db.query(func.avg(GeneratedImage.generation_time)).scalar()
        
        # 최근 24시간 생성 수
        from datetime import timedelta
        yesterday = datetime.now() - timedelta(days=1)
        recent_count = db.query(GeneratedImage).filter(
            GeneratedImage.created_at >= yesterday
        ).count()
        
        return {
            "total_images": total_images,
            "by_type": {
                "text2img": text2img_count,
                "img2img": img2img_count,
                "batch": batch_count
            },
            "avg_generation_time_sec": round(avg_time, 2) if avg_time else 0,
            "last_24h_count": recent_count
        }
    finally:
        db.close()

# ============================================================================
# 배치 생성 함수
# ============================================================================

def scheduled_batch_generation():
    """스케줄된 배치 이미지 생성"""
    print(f"\n{'='*60}")
    print(f"🤖 자동 배치 생성 시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
    
    batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 배치 프롬프트
    prompts = [
        "serene mountain landscape at sunrise",
        "cyberpunk city street at night",
        "cute cat playing with yarn",
        "fantasy castle on floating island",
        "abstract colorful geometric art"
    ]
    
    for idx, prompt in enumerate(prompts, 1):
        try:
            print(f"[{idx}/{len(prompts)}] 생성 중: {prompt[:50]}...")
            
            start_time = datetime.now()
            
            # 이미지 생성
            image = pipe_text2img(
                prompt=prompt,
                num_inference_steps=30,
                guidance_scale=7.5
            ).images[0]
            
            generation_time = (datetime.now() - start_time).total_seconds()
            
            # 저장
            filename = f"batch_generated/batch_{batch_id}_{idx:02d}.png"
            image.save(filename)
            
            # DB 저장
            params = {"steps": 30, "guidance_scale": 7.5, "width": 512, "height": 512}
            save_to_database(prompt, params, filename, generation_time, "batch", batch_id)
            
            print(f"  ✅ 완료 ({generation_time:.2f}초) - {filename}")
        
        except Exception as e:
            print(f"  ❌ 에러: {e}")
    
    print(f"\n✅ 배치 생성 완료! Batch ID: {batch_id}\n")

# ============================================================================
# 스케줄러 설정
# ============================================================================

scheduler = BackgroundScheduler()

# 매일 오전 2시에 배치 생성 실행
scheduler.add_job(
    scheduled_batch_generation,
    trigger=CronTrigger(hour=2, minute=0),
    id='daily_batch_generation',
    name='매일 자동 배치 이미지 생성'
)

# 테스트용: 서버 시작 5분 후 실행 (주석 처리 가능)
# from datetime import timedelta
# scheduler.add_job(
#     scheduled_batch_generation,
#     'date',
#     run_date=datetime.now() + timedelta(minutes=5),
#     id='test_batch_generation'
# )

scheduler.start()
print("\n⏰ 스케줄러 시작됨: 매일 오전 2시 자동 배치 생성")

# ============================================================================
# 서버 시작
# ============================================================================

@app.on_event("startup")
def startup_event():
    print("\n" + "="*60)
    print("✅ 서버 준비 완료!")
    print("="*60)
    print(f"📍 API 문서: http://localhost:8000/docs")
    print(f"📍 Health Check: http://localhost:8000/health")
    print(f"📍 데이터베이스: sd_database.db")
    print("="*60 + "\n")

@app.on_event("shutdown")
def shutdown_event():
    scheduler.shutdown()
    print("\n⏹️ 서버 종료됨")

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )