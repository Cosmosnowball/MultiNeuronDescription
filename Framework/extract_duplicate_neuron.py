import os
import pandas as pd
from functools import reduce

# 현재 디렉토리의 모든 CSV 파일 찾기 (output_explain 제외)
csv_files = [f for f in os.listdir('.') if f.endswith('.csv') and 'output_explain' not in f]

# 레이어별로 뉴런 인덱스 세트를 모으기 위한 딕셔너리 초기화
layer_neuron_sets = {}

for file in csv_files:
    df = pd.read_csv(file)
    print(f"{file} 처리 중...")

    for _, row in df.iterrows():
        layer = row['Layer']
        filters_str = row['Selected Filter Indices']
        
        # 문자열을 숫자 리스트로 변환
        neuron_indices = set(map(int, filters_str.split(',')))
        
        if layer not in layer_neuron_sets:
            layer_neuron_sets[layer] = []
        
        layer_neuron_sets[layer].append(neuron_indices)

# 레이어별로 공통된 뉴런 인덱스 계산
common_neurons_per_layer = {}

for layer, list_of_sets in layer_neuron_sets.items():
    # 교집합 계산
    common_neurons = set.intersection(*list_of_sets)
    common_neurons_per_layer[layer] = sorted(list(common_neurons))  # 정렬해서 보기 좋게

# 결과를 DataFrame으로 변환 후 저장
result_df = pd.DataFrame([
    {'Layer': layer, 'Common Filter Indices': ', '.join(map(str, indices))}
    for layer, indices in common_neurons_per_layer.items()
])

result_df.to_csv('common_neuron_groups_per_layer.csv', index=False, encoding='utf-8')
print("공통 뉴런 그룹을 common_neuron_groups_per_layer.csv 파일로 저장했습니다.")