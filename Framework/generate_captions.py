from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import os
import csv

# 모델 로딩
device = "cuda" 
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

image_dir = "unsplash_images"
output_file = "output_explain.csv"
batch_size = 128

# 이미지 목록
image_files = [
    f for f in os.listdir(image_dir)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

results = []

# 🔁 배치 단위로 처리
for i in range(0, len(image_files), batch_size):
    batch_files = image_files[i:i + batch_size]
    
    images = []
    for fname in batch_files:
        image_path = os.path.join(image_dir, fname)
        
        try:
            # 이미지 열기
            with Image.open(image_path) as img:
                # 이미지 크기 체크 (너무 크면 건너뜀)
                if img.size[0] * img.size[1] > 178956970:  # 예시: 크기가 너무 크면 건너뜀
                    print(f"Skipping {fname} due to large size.")
                    continue  # 너무 큰 이미지는 건너뛰기

                # 정상적인 이미지는 RGB로 변환하여 리스트에 추가
                images.append(img.convert("RGB"))
        except Exception as e:
            print(f"Error opening {fname}: {e}")
            continue  # 오류가 발생한 이미지는 건너뛰기

    if images:  # 이미지가 있는 경우에만 처리
        # 모델 입력 준비 및 처리
        inputs = processor(images, return_tensors="pt", padding=True).to(device)
        outputs = model.generate(**inputs)

        # 캡션 디코딩
        captions = processor.batch_decode(outputs, skip_special_tokens=True)

        # 파일 이름과 캡션을 결과 리스트에 추가
        for fname, caption in zip(batch_files, captions):
            results.append([fname, caption])

    # 진행 상황 출력
    progress = min(i + batch_size, len(image_files)) / len(image_files) * 100
    print(f"Progress: {progress:.2f}% ({min(i + batch_size, len(image_files))}/{len(image_files)})")

# 결과를 CSV 파일로 저장
with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Image", "Caption"])
    writer.writerows(results)

print(f"\n✅ 저장 완료: {output_file}")
