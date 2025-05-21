import os
import pandas as pd

# CSV 파일 로드 (이미지 이름과 Caption 포함)
caption_df = pd.read_csv('output_explain.csv')

# 이미지 이름과 Caption을 딕셔너리로 변환
caption_dict = dict(zip(caption_df['Image'], caption_df['Caption']))

# 현재 디렉토리 내의 모든 폴더 탐색
for folder_name in os.listdir('.'):
    if os.path.isdir(folder_name):
        captions = []
        # 폴더 안의 모든 파일 확인
        for img_file in os.listdir(folder_name):
            if img_file in caption_dict:
                caption = caption_dict[img_file]
                captions.append(f"{caption}")
            else:
                captions.append(f"{img_file}: (Caption 없음)")

        # Caption을 현재 디렉토리에 txt 파일로 저장 (폴더명 + _Caption.txt)
        output_txt_name = f"{folder_name}_Caption.txt"
        with open(output_txt_name, 'w', encoding='utf-8') as f:
            for line in captions:
                f.write(line + '\n')

        print(f"{output_txt_name} 작성 완료!")