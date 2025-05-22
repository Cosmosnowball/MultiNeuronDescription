import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import numpy as np
from sklearn.metrics import roc_auc_score

# 1. 뉴런 인덱스 정의
# 아래 뉴런 그룹은 레벨1 클러스터0의 뉴런 그룹. 대표 캡션은 A waterfall in the forest
layer_neuron_indices = {
    'conv1': [3, 19, 22, 45, 54, 58, 62],
    'layer1': [8, 11, 46, 47, 50, 51, 63],
    'layer2': [36, 41, 43, 58, 66, 67, 84, 96, 97, 99, 106, 119],
    'layer3': [8, 22, 28, 33, 52, 68, 71, 89, 95, 98, 109, 112, 136, 152, 158, 169, 192, 197, 208, 212, 239],
    'layer4': [21, 39, 65, 88, 101, 120, 149, 163, 168, 181, 182, 193, 213, 226, 250, 256, 274, 317, 366, 370, 400, 420, 424, 428, 435, 438, 454, 480, 481]
}

# 2. 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 3. 모델 준비
model = models.resnet18(pretrained=True)
model.eval()

# 4. 중간 활성화 저장용 딕셔너리 및 hook 등록
activation = {}

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

model.conv1.register_forward_hook(get_activation('conv1'))
model.layer1.register_forward_hook(get_activation('layer1'))
model.layer2.register_forward_hook(get_activation('layer2'))
model.layer3.register_forward_hook(get_activation('layer3'))
model.layer4.register_forward_hook(get_activation('layer4'))

# 5. 폴더별 활성화 평균 계산 함수
def compute_average_activations(folder_path):
    results = []
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        image = Image.open(img_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            _ = model(input_tensor)

        all_activations = []
        for layer_name, neuron_indices in layer_neuron_indices.items():
            layer_output = activation[layer_name].squeeze()  # (C, H, W)
            if layer_output.ndim == 3:
                pooled = layer_output.mean(dim=(1, 2))  # (C,)
            else:
                pooled = layer_output  # (C,)

            selected_neurons = pooled[neuron_indices]
            all_activations.append(selected_neurons.mean().item())

        avg_activation = np.mean(all_activations)
        results.append(avg_activation)

    return results



# COSY 논문 기반 AUC 계산
def calculate_auc(A0, A1):
    labels = [0] * len(A0) + [1] * len(A1)
    scores = A0 + A1
    return roc_auc_score(labels, scores)

# COSY 논문 기반 MAD 계산
def calculate_mad(A0, A1):
    A0 = np.array(A0)
    A1 = np.array(A1)
    mean_A0 = A0.mean()
    std_A0 = A0.std(ddof=1)  # sample std dev
    mean_A1 = A1.mean()
    if std_A0 == 0:  # edge case 방지
        return 0
    return (mean_A1 - mean_A0) / std_A0




# 두 개의 폴더에 대해 계산
# 비교할 두 이미지셋 폴더 (각 폴더 안에는 이미지들이 있어야 함) 
A1 = compute_average_activations('level1_cluster_0')
A0 = compute_average_activations('random') # A0는 비교군으로, Random 이미지가 고정적으로 들어감.

# 결과 출력 
# 각 이미지들에 대한 평균 활성화 값 A0, A1
print("A1 :", A1)
print("A0 :", A0)

auc_score = calculate_auc(A0, A1)
mad_score = calculate_mad(A0, A1)

# 평균 활성화 값으로 계산한 AUC, MAD 값
print(f"AUC: {auc_score:.4f}")
print(f"MAD: {mad_score:.4f}")