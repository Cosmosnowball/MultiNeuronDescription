01 generate_captions.py 
모든 이미지에 대한 캡션을 생성. csv 파일 생성.

02 FINCH_clustering.py
생성된 캡션을 기반으로 이미지를 클러스터링. 계층적으로 클러스터링된 폴더 구조 생성.

03 find_caption.py
폴더구조상 level 2 디렉토리에서 속한 level 1 폴더들을 대상으로, 이미지들의 캡션을 찾아오는 파일. 캡션 파일 생성. -> 이후 대표 캡션을 chat gpt로 생성하는 과정은 현재 수작업임.

04 extract_neuron.py 
level 2 디렉토리에서 각 level 1 폴더의 이미지들이 활성화 시키는 뉴런 그룹을 추출함. selected_filter.csv 파일을 level1 폴더의 개수만큼 생성.

05 extract_duplicate_neuron.py
level 2 디렉토리에서 생성된 selected_filter.csv 파일을 읽어 중복된 뉴런들을 추출함. 