# batch_generation.py
from diffusers import DiffusionPipeline
import torch
import pandas as pd
from datetime import datetime
import os

# 결과 저장 폴더 생성
os.makedirs("batch_results", exist_ok=True)

# 파이프라인 로드
print("📦 파이프라인 로딩 중...")
pipe = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")
print("✅ 로딩 완료!")

# 테스트할 프롬프트 데이터
test_data = [
    # 풍경
    {"category": "landscape", "prompt": "serene mountain lake at sunrise, misty, peaceful", "steps": 30},
    {"category": "landscape", "prompt": "tropical beach paradise, crystal clear water, palm trees", "steps": 30},
    {"category": "landscape", "prompt": "autumn forest path, golden leaves, soft sunlight", "steps": 30},
    
    # 동물
    {"category": "animal", "prompt": "majestic lion portrait, professional wildlife photography", "steps": 40},
    {"category": "animal", "prompt": "cute red panda in bamboo forest, detailed fur", "steps": 40},
    {"category": "animal", "prompt": "hummingbird feeding on flower, macro photography", "steps": 40},
    
    # 도시/건축
    {"category": "urban", "prompt": "futuristic cyberpunk city at night, neon lights", "steps": 50},
    {"category": "urban", "prompt": "traditional japanese temple in cherry blossom season", "steps": 50},
    {"category": "urban", "prompt": "modern minimalist architecture, concrete and glass", "steps": 50},
    
    # 추상/예술
    {"category": "abstract", "prompt": "colorful abstract fluid art, vibrant swirls", "steps": 30},
    {"category": "abstract", "prompt": "geometric patterns, minimalist design, pastel colors", "steps": 30},
    {"category": "abstract", "prompt": "cosmic nebula, space art, deep purple and blue", "steps": 30}
]

results = []

print(f"\n🎨 배치 생성 시작: 총 {len(test_data)}개 이미지\n")

for idx, data in enumerate(test_data, 1):
    print(f"[{idx}/{len(test_data)}] 생성 중: {data['category']} - {data['prompt'][:40]}...")
    
    start_time = datetime.now()
    
    # 이미지 생성
    image = pipe(
        data['prompt'],
        num_inference_steps=data['steps'],
        guidance_scale=7.5
    ).images[0]
    
    generation_time = (datetime.now() - start_time).total_seconds()
    
    # 파일 저장
    filename = f"batch_results/{idx:02d}_{data['category']}.png"
    image.save(filename)
    
    # 결과 기록
    results.append({
        'index': idx,
        'category': data['category'],
        'prompt': data['prompt'],
        'steps': data['steps'],
        'filename': filename,
        'generation_time_sec': generation_time,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })
    
    print(f"  ✅ 완료 ({generation_time:.2f}초) - {filename}")

# 결과를 DataFrame으로 저장
results_df = pd.DataFrame(results)

# 통계 출력
print("\n" + "="*60)
print("📊 생성 완료 통계")
print("="*60)
print(f"총 생성 이미지 수: {len(results_df)}")
print(f"평균 생성 시간: {results_df['generation_time_sec'].mean():.2f}초")
print(f"최단 생성 시간: {results_df['generation_time_sec'].min():.2f}초")
print(f"최장 생성 시간: {results_df['generation_time_sec'].max():.2f}초")
print(f"총 소요 시간: {results_df['generation_time_sec'].sum():.2f}초 ({results_df['generation_time_sec'].sum()/60:.1f}분)")

# 카테고리별 통계
print("\n📈 카테고리별 평균 생성 시간:")
category_stats = results_df.groupby('category')['generation_time_sec'].agg(['mean', 'count'])
print(category_stats)

# CSV 저장
csv_filename = f"batch_results/results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
results_df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
print(f"\n💾 결과 저장됨: {csv_filename}")

print("\n🎉 배치 생성 완료! batch_results 폴더를 확인하세요.")