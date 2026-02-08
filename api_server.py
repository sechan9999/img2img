# api_server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from diffusers import DiffusionPipeline
import torch
import base64
from io import BytesIO
import uvicorn

# FastAPI 앱
app = FastAPI(title="Stable Diffusion API")

# 파이프라인 글로벌 로드
print("📦 모델 로딩 중...")
pipe = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")
print("✅ API 서버 준비 완료!")

# 요청 모델
class ImageRequest(BaseModel):
    prompt: str
    steps: int = 50
    guidance_scale: float = 7.5
    width: int = 512
    height: int = 512

# 응답 모델
class ImageResponse(BaseModel):
    image_base64: str
    prompt: str
    steps: int

@app.get("/")
def read_root():
    return {"message": "Stable Diffusion API is running!"}

@app.post("/generate", response_model=ImageResponse)
def generate_image(request: ImageRequest):
    try:
        # 이미지 생성
        image = pipe(
            request.prompt,
            num_inference_steps=request.steps,
            guidance_scale=request.guidance_scale,
            height=request.height,
            width=request.width
        ).images[0]
        
        # Base64 인코딩
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        return ImageResponse(
            image_base64=img_base64,
            prompt=request.prompt,
            steps=request.steps
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy", "gpu_available": torch.cuda.is_available()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)