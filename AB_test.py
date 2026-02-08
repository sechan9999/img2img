# prompt_testing.py
from diffusers import DiffusionPipeline
import torch
import pandas as pd
from datetime import datetime

pipe = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

# A/B 테스트할 프롬프트 변형
base_concept = "mountain landscape"

prompt_variations = [
    f"{base_concept}",
    f"{base_concept}, photorealistic, 4k",
    f"{base_concept}, oil painting style",
    f"{base_concept}, dramatic lighting, cinematic",
    f"{base_concept}, highly detailed, professional photography",
    f"beautiful {base_concept} at sunset",
    f"majestic {base_concept}, epic view",
]

results = []

for idx, prompt in enumerate(prompt_variations, 1):
    print(f"[{idx}/{len(prompt_variations)}] 테스트 중: {prompt}")
    
    # 동일 시드로 3번 생성
    for seed in [42, 123, 999]:
        generator = torch.Generator("cuda").manual_seed(seed)
        
        start = datetime.now()
        image = pipe(
            prompt,
            num_inference_steps=50,
            generator=generator
        ).images[0]
        time_taken = (datetime.now() - start).total_seconds()
        
        filename = f"ab_test_{idx}_{seed}.png"
        image.save(filename)
        
        results.append({
            'prompt': prompt,
            'seed': seed,
            'filename': filename,
            'time_sec': time_taken
        })

df = pd.DataFrame(results)
df.to_csv('prompt_ab_test_results.csv', index=False)
print("\n✅ A/B 테스트 완료! 결과를 비교해보세요.")