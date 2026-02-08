import gradio as gr
import requests
import base64
from io import BytesIO
from PIL import Image

# API 서버 주소
API_URL = "http://localhost:8000"

# 스타일 목록 (서버와 동기화)
STYLES = [
    "lcm", "realistic", "anime", "oil_painting", "watercolor", 
    "cyberpunk", "fantasy", "minimalist", "vintage", ""
]

def check_server():
    """서버 상태 확인"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        if response.status_code == 200:
            data = response.json()
            gpu = data.get("gpu_name", "Unknown")
            return f"✅ 서버 연결됨 (GPU: {gpu})"
    except:
        return "❌ 서버 연결 실패 (python integrated_sd_server.py 실행 필요)"
    return "❌ 서버 응답 오류"

def generate_image(prompt, negative_prompt, style, steps, guidance_scale, width, height, seed):
    """Text-to-Image 생성 요청"""
    try:
        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "style": style if style else None,
            "steps": int(steps),
            "guidance_scale": float(guidance_scale),
            "width": int(width),
            "height": int(height),
            "seed": int(seed) if seed != -1 else None
        }
        
        response = requests.post(f"{API_URL}/generate", json=payload, timeout=120)
        
        if response.status_code == 200:
            data = response.json()
            image_data = base64.b64decode(data["image_base64"])
            image = Image.open(BytesIO(image_data))
            info = f"생성 시간: {data['generation_time']}초\nSeed: {data['seed']}"
            return image, info
        else:
            return None, f"에러: {response.text}"
    except Exception as e:
        return None, f"요청 실패: {str(e)}"

def generate_img2img(image, prompt, negative_prompt, style, strength, steps, seed):
    """Image-to-Image 변환 요청"""
    if image is None:
        return None, "이미지를 업로드해주세요."
        
    try:
        # 이미지를 바이트로 변환
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        buffered.seek(0)
        
        files = {"file": ("image.png", buffered, "image/png")}
        params = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "style": style if style else None,
            "strength": float(strength),
            "steps": int(steps),
            "guidance_scale": 7.5,
            "seed": int(seed) if seed != -1 else None
        }
        
        response = requests.post(f"{API_URL}/img2img", files=files, params=params, timeout=120)
        
        if response.status_code == 200:
            data = response.json()
            image_data = base64.b64decode(data["image_base64"])
            result_image = Image.open(BytesIO(image_data))
            info = f"변환 시간: {data['generation_time']}초\nSeed: {data['seed']}"
            return result_image, info
        else:
            return None, f"에러: {response.text}"
    except Exception as e:
        return None, f"요청 실패: {str(e)}"

# Gradio UI 구성
with gr.Blocks(title="Stable Diffusion V2", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎨 Stable Diffusion V2.0 (with LCM)")
    
    server_status = gr.Textbox(value=check_server, label="서버 상태", interactive=False)
    
    with gr.Tabs():
        # Text-to-Image 탭
        with gr.TabItem("✨ 텍스트 → 이미지"):
            with gr.Row():
                with gr.Column(scale=1):
                    t2i_prompt = gr.Textbox(label="프롬프트 (영어)", placeholder="e.g. a futuristic city with flying cars", lines=3)
                    t2i_negative = gr.Textbox(label="네거티브 프롬프트", placeholder="low quality, blurry", value="low quality, blurry")
                    
                    with gr.Row():
                        t2i_style = gr.Dropdown(choices=STYLES, value="lcm", label="스타일 (lcm = 고속 모드)")
                        t2i_steps = gr.Slider(minimum=4, maximum=100, value=20, step=1, label="Steps (lcm은 4-8 권장)")
                    
                    with gr.Accordion("고급 설정", open=False):
                        t2i_guidance = gr.Slider(minimum=1.0, maximum=20.0, value=7.5, step=0.5, label="Guidance Scale")
                        t2i_width = gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="가로 크기")
                        t2i_height = gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="세로 크기")
                        t2i_seed = gr.Number(value=-1, label="Seed (-1 = Random)", precision=0)
                    
                    t2i_btn = gr.Button("🚀 이미지 생성", variant="primary")
                
                with gr.Column(scale=1):
                    t2i_output = gr.Image(label="생성 결과", type="pil")
                    t2i_info = gr.Textbox(label="생성 정보")
            
            t2i_btn.click(
                fn=generate_image,
                inputs=[t2i_prompt, t2i_negative, t2i_style, t2i_steps, t2i_guidance, t2i_width, t2i_height, t2i_seed],
                outputs=[t2i_output, t2i_info]
            )

        # Img2Img 탭
        with gr.TabItem("🔄 이미지 → 이미지"):
            with gr.Row():
                with gr.Column(scale=1):
                    i2i_input = gr.Image(label="원본 이미지", type="pil")
                    i2i_prompt = gr.Textbox(label="변환 프롬프트", placeholder="e.g. oil painting style")
                    i2i_negative = gr.Textbox(label="네거티브 프롬프트", value="low quality, blurry")
                    i2i_style = gr.Dropdown(choices=STYLES, value="oil_painting", label="스타일")
                    
                    i2i_strength = gr.Slider(minimum=0.1, maximum=1.0, value=0.6, step=0.05, label="변환 강도 (높을수록 원본 파괴)")
                    i2i_steps = gr.Slider(minimum=4, maximum=100, value=30, step=1, label="Steps")
                    i2i_seed = gr.Number(value=-1, label="Seed (-1 = Random)", precision=0)
                    
                    i2i_btn = gr.Button("🔄 변환하기", variant="primary")
                
                with gr.Column(scale=1):
                    i2i_output = gr.Image(label="변환 결과", type="pil")
                    i2i_info = gr.Textbox(label="변환 정보")
            
            i2i_btn.click(
                fn=generate_img2img,
                inputs=[i2i_input, i2i_prompt, i2i_negative, i2i_style, i2i_strength, i2i_steps, i2i_seed],
                outputs=[i2i_output, i2i_info]
            )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
