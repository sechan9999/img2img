# img2img_test.py
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import torch
import requests
from io import BytesIO

# 파이프라인 로드
print("📦 모델 로딩 중...")
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")
print("✅ 모델 로딩 완료!")

# 테스트 이미지 다운로드 (또는 본인 이미지 사용)
print("🖼️ 샘플 이미지 다운로드 중...")
url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
response = requests.get(url)
init_image = Image.open(BytesIO(response.content)).convert("RGB")
init_image = init_image.resize((512, 512))
init_image.save("input_original.png")
print("✅ 원본 이미지 저장: input_original.png")

# 다양한 스타일로 변환
styles = [
    {
        "prompt": "beautiful oil painting of mountains at sunset, impressionist style",
        "strength": 0.75,
        "filename": "output_oil_painting.png"
    },
    {
        "prompt": "photorealistic mountain landscape, professional photography, golden hour",
        "strength": 0.6,
        "filename": "output_photorealistic.png"
    },
    {
        "prompt": "watercolor painting of mountains, soft colors, artistic",
        "strength": 0.8,
        "filename": "output_watercolor.png"
    },
    {
        "prompt": "anime style mountain landscape, vibrant colors, studio ghibli",
        "strength": 0.85,
        "filename": "output_anime.png"
    }
]

print("\n🎨 스타일 변환 시작...")
for i, style in enumerate(styles, 1):
    print(f"\n[{i}/{len(styles)}] {style['prompt'][:50]}...")
    
    image = pipe(
        prompt=style['prompt'],
        image=init_image,
        strength=style['strength'],  # 0.0=원본 유지, 1.0=완전 새로 생성
        guidance_scale=7.5,
        num_inference_steps=50
    ).images[0]
    
    image.save(style['filename'])
    print(f"✅ 저장됨: {style['filename']}")

print("\n🎉 모든 변환 완료! 파일들을 확인하세요.")