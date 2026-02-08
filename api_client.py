# api_client.py
import requests
import base64
from PIL import Image
from io import BytesIO

# API 엔드포인트
API_URL = "http://localhost:8000/generate"

# 요청 데이터
data = {
    "prompt": "cyberpunk city at night, neon lights, futuristic",
    "steps": 40,
    "guidance_scale": 7.5
}

print("📤 API 요청 전송 중...")
response = requests.post(API_URL, json=data)

if response.status_code == 200:
    result = response.json()
    
    # Base64 디코드
    image_data = base64.b64decode(result["image_base64"])
    image = Image.open(BytesIO(image_data))
    
    # 저장
    filename = "api_generated.png"
    image.save(filename)
    
    print(f"✅ 성공! {filename} 저장됨")
    print(f"프롬프트: {result['prompt']}")
    print(f"Steps: {result['steps']}")
else:
    print(f"❌ 에러: {response.status_code}")