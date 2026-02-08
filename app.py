import gradio as gr
from diffusers import DiffusionPipeline
import torch

# 파이프라인 로드 (한 번만)
print("모델 로딩 중...")
pipe = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")
print("✅ 모델 로딩 완료!")

def generate_image(prompt, steps, guidance_scale):
    """이미지 생성 함수"""
    image = pipe(
        prompt, 
        num_inference_steps=steps,
        guidance_scale=guidance_scale
    ).images[0]
    return image

# Gradio 인터페이스
demo = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Textbox(label="프롬프트", placeholder="원하는 이미지를 영어로 설명하세요..."),
        gr.Slider(20, 100, value=50, step=10, label="Steps (품질)"),
        gr.Slider(1, 20, value=7.5, step=0.5, label="Guidance Scale (프롬프트 충실도)")
    ],
    outputs=gr.Image(label="생성된 이미지"),
    title="🎨 Stable Diffusion 이미지 생성기",
    description="RTX 4050으로 구동되는 AI 이미지 생성기"
)

demo.launch(share=True)  # share=True로 외부 접속 가능한 링크 생성