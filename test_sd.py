from diffusers import DiffusionPipeline
import torch

# GPU 사용 설정
pipe = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16  # GPU 최적화 (CPU는 float32)
).to("cuda")  # GPU로 이동!

# 이미지 생성
prompt = "a beautiful sunset over mountains, highly detailed, 4k"
image = pipe(prompt, num_inference_steps=50).images[0]

# 이미지 저장
image.save("output_gpu.png")
print("✅ GPU로 이미지 생성 완료! output_gpu.png 파일을 확인하세요.")