# img2img_app.py
import gradio as gr
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import torch

# 파이프라인 로드
print("📦 Img2Img 파이프라인 로딩 중...")
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")
print("✅ 로딩 완료!")

def transform_image(input_image, prompt, strength, steps, guidance_scale):
    """이미지 변환 함수"""
    if input_image is None:
        return None
    
    # PIL Image로 변환 및 리사이즈
    init_image = Image.fromarray(input_image).convert("RGB")
    init_image = init_image.resize((512, 512))
    
    # 이미지 생성
    result = pipe(
        prompt=prompt,
        image=init_image,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=steps
    ).images[0]
    
    return result

# Gradio 인터페이스
demo = gr.Interface(
    fn=transform_image,
    inputs=[
        gr.Image(label="원본 이미지 업로드"),
        gr.Textbox(
            label="변환 스타일 (프롬프트)", 
            placeholder="예: oil painting style, vibrant colors",
            value="beautiful oil painting, artistic"
        ),
        gr.Slider(0.3, 1.0, value=0.75, step=0.05, label="Strength (변환 강도: 높을수록 원본과 달라짐)"),
        gr.Slider(20, 100, value=50, step=10, label="Steps"),
        gr.Slider(1, 20, value=7.5, step=0.5, label="Guidance Scale")
    ],
    outputs=gr.Image(label="변환된 이미지"),
    title="🎨 이미지-to-이미지 AI 변환기",
    description="이미지를 업로드하고 원하는 스타일을 입력하세요. RTX 4050으로 빠르게 변환됩니다!",
    examples=[
        [None, "oil painting style, impressionist", 0.75, 50, 7.5],
        [None, "anime style, studio ghibli", 0.85, 50, 7.5],
        [None, "photorealistic, professional photography", 0.6, 50, 7.5],
        [None, "watercolor painting, soft colors", 0.8, 50, 7.5]
    ]
)

demo.launch(share=True)