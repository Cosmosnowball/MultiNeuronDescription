import os
import csv
from PIL import Image
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms

# 1. ResNet-18 모델 불러오기 (ImageNet 사전학습 가중치 사용) 및 평가 모드로 설정
model = models.resnet18(pretrained=True)
model.eval()

# 중간 레이어 출력을 저장할 딕셔너리 생성
activations = {}

# hook 함수 정의: forward 단계에서 호출되어 출력값을 사본(detach)으로 저장
def save_activation(name):
    def hook(module, input, output):
        activations[name] = output.detach()
    return hook

# 지정된 레이어에 forward hook 등록
model.conv1.register_forward_hook(save_activation('conv1'))
model.layer1.register_forward_hook(save_activation('layer1'))
model.layer2.register_forward_hook(save_activation('layer2'))
model.layer3.register_forward_hook(save_activation('layer3'))
model.layer4.register_forward_hook(save_activation('layer4'))

# 이미지 전처리 파이프라인 정의 (Resize -> CenterCrop -> ToTensor -> Normalize)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 메인 루프: 현재 디렉토리 내의 모든 폴더 순회
base_folder = '.'  # 현재 작업 디렉토리 (또는 절대경로 지정 가능)
for folder_name in os.listdir(base_folder):
    folder_path = os.path.join(base_folder, folder_name)
    if not os.path.isdir(folder_path):
        continue  # 폴더만 대상으로 함

    # 이미지 파일 목록 읽기
    image_paths = [os.path.join(folder_path, fname)
                   for fname in os.listdir(folder_path)
                   if fname.endswith('.jpg')]

    if not image_paths:
        #print(f"{folder_name}: 이미지 없음, 건너뜀.")
        image_paths = [os.path.join(folder_path, fname)
                   for fname in os.listdir(folder_path)
                   if fname.endswith('.png')]
        continue

    print(f"{folder_name}: {len(image_paths)}개의 이미지를 처리 중...")

    # 레이어별 필터 카운트 초기화
    filter_counts = {
        'conv1': None,
        'layer1': None,
        'layer2': None,
        'layer3': None,
        'layer4': None
    }

    with torch.no_grad():
        for idx, img_path in enumerate(image_paths):
            try:
                image = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"{img_path} 불러오기 실패: {e}")
                continue

            input_tensor = transform(image).unsqueeze(0)
            _ = model(input_tensor)

            # 첫 이미지에서만 배열 초기화
            if idx == 0:
                for layer_name, activation in activations.items():
                    num_filters = activation.shape[1]
                    filter_counts[layer_name] = np.zeros(num_filters, dtype=int)

            # 상위 10% 필터 찾기
            for layer_name, activation in activations.items():
                # activation shape: (1, 필터수, H, W)
                mean_per_filter = activation.mean(dim=(0, 2, 3)).cpu().numpy()  # (필터수,)
                threshold = np.percentile(mean_per_filter, 90)
                selected_filters = np.where(mean_per_filter > threshold)[0]

                # 카운트 업데이트
                for f_idx in selected_filters:
                    filter_counts[layer_name][f_idx] += 1

    # 전체 이미지 개수의 절반 이상 활성화된 필터 인덱스 추출
    half_count = len(image_paths) / 2
    results = []
    for layer_name, counts in filter_counts.items():
        selected_indices = np.where(counts >= half_count)[0].tolist()
        print(f"{folder_name} - {layer_name}: {selected_indices}")
        results.append({
            'folder': folder_name,
            'layer': layer_name,
            'selected_filters': selected_indices
        })

    # 결과를 CSV로 저장
    csv_filename = f"{folder_name}_selected_filters.csv"
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Layer', 'Selected Filter Indices'])
        for r in results:
            writer.writerow([r['layer'], ', '.join(map(str, r['selected_filters']))])

    print(f"{folder_name} 결과가 {csv_filename}로 저장되었습니다.")