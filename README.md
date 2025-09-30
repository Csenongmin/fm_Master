# ⚽ Football Tracking data를 이용하여 Events data 예측 및 분석
2025 전기 졸업과제 fm 마스터

###  프로젝트 개요
#### 1.1. 국내외 시장 현황 및 문제점
> 현대 스포츠에서는 **트래킹 데이터**(선수 위치 및 움직임 정보)가 경기 전략 수립과 해석의 핵심 요소로 떠오르고 있습니다. 특히 축구에서는 22명의 선수와 공이 끊임없이 상호작용하기 때문에, 단순 이벤트 기록으로는 한계가 있으며 **시공간 데이터를 기반으로 한 정밀한 분석**이 필요합니다. 이에 따른 많은 연구가 진행되어야 스포츠 분야가 더 발전할 수 있다고 생각합니다.
> 하지만, 축구 경기에 관련한 데이터는 구단 내 private한 데이터로 취급하여 **연구에 사용될 public한 데이터를 수집하는 것**에 많은 어려움이 있습니다.

#### 1.2. 필요성과 기대효과
> 본 프로젝트는 **트래킹 데이터를 통해 축구 경기 중 발생하는 이벤트(패스, 슈팅, 태클 등)를 예측하고 분석하는 시스템**을 연구함으로써, 기존에 축구 데이터를 일일이 수집하던 것을 넘어 생성형 AI 등으로 완전한 축구 경기를 직접 만들어낼 수 있는 시스템에 기여하고자 합니다.


### 2. 개발 목표
#### 2.1. 목표 및 세부 내용
> 트래킹 데이터를 기반으로 **패스, 슈팅, 드리블**과 같이 공의 주요 이벤트를 예측하는 모델을 설계 및 튜닝합니다.
> **Football Manager 2024** 시뮬레이터를 활용하여 경기 트래킹 데이터를 수집합니다.
> 추출된 데이터를 정제·보정하여 학습 가능한 형태로 가공합니다.

#### 2.2. 기존 서비스 대비 차별성 
> 저희가 연구하는 모델은 트래킹 데이터만으로 공의 상태(state)를 예측함으로써 이벤트를 수집하는 것이 목표입니다. 기존의 기록가가 일일이 이벤트를 기록하는 것과 차별성이 존재합니다.

#### 2.3. 사회적 가치 도입 계획 
> 저희가 연구한 모델은 아직 미비하지만, 더 많은 연구가 이뤄진다면 더 쉽게 축구 데이터를 수집함으로써 더 나아가 데이터를 생성하는데에 기여할 수 있을 것이라고 생각합니다.



### 3. 시스템 설계
#### 3.1. 시스템 구성도
<img src="https://github.com/user-attachments/assets/de4a8e9a-f096-4264-a9b7-a1e5031f8558" width="600px" title="Title" alt="Alt text"></img>

#### 3.2. 사용 기술
 - 프로그래밍 언어: Python 3.10.18
 - 개발 도구: Jupyter Notebook, VS code
 - Fm data 추출: Yolov8
 - Event 예측 모델: Transformer, CNN, TCN
 - 가상환경: conda
 - PyTorch(nn/optim), scikit-learn(StandardScaler, GroupKFold, metrics), Metrica sports, Floodlight.io.dfl

### 4. 개발 결과
#### 4.1. 전체 시스템 흐름도

1. Fm in-game data extracting<br>
<img src="https://github.com/user-attachments/assets/631ea624-c7e6-4e22-b49f-201934e32cbc" width="600px" title="Title" alt="Alt text"></img>
</br>
2. Model architecture <br>
<img src="https://github.com/user-attachments/assets/fc68b4eb-1802-4be1-bd1b-f4bdf08cef05" width="600px" title="Title" alt="Alt text"></img>
</br>

#### 4.2. 기능 설명 및 주요 기능 명세서

|요구사항|기능|상세 설명|
|------|---|---|
|FM 인게임 이미지 처리|image_processing/cropping |OpenCV기반 25fps 이미지 캡쳐 및 cropping.|
|FM 인게임 데이터 추출|movement2cords, interpolating, filter, ball2cord|선수: OpenCV기반 객체 탐지 및 interpolating, 공: Yolov8 탐지|
|모델학습 데이터셋|data_processing, visualization|Floodlight.io.dfl 기반 xml data 처리 프로세스|
|모델 설계 및 시각화|ball_dat, ball_state_viz, model/train_val|Transformer 기반 모델 설계 및 시각화 plot|

#### 4.3. 디렉토리 구조
```
📦fmMaster
 ┣ 📂data
 ┣ 📂fmdata_extracting
 ┃ ┣ 📜filter.ipynb
 ┃ ┣ 📜find_image_cords.py
 ┃ ┣ 📜image_processing.py
 ┃ ┣ 📜interpolating.py
 ┃ ┣ 📜Metrica_IO.py
 ┃ ┣ 📜Metrica_Viz.py
 ┃ ┣ 📜movement2cords.ipynb
 ┃ ┗ 📜visualization.ipynb
 ┣ 📂model
 ┃ ┗ 📜model.py
 ┣ 📂plots
 ┣ 📂utils
 ┃ ┣ 📜data_processing.py
 ┃ ┗ 📜visualization.py
 ┣ 📜ball_data.py
 ┣ 📜ball_state_viz.py
 ┣ 📜README.md
 ┗ 📜train_val.py
```
 
#### 4.4. 산업체 멘토링 의견 및 반영 사항
1. FM2024 공과 선수의 싱크 문제: 칼만 필터 및 사비츠키-골레이 필터와 같은 보간법으로 어느정도 해결.
2. 데이터 증강 세미 자동 라벨링 도구: Labeling 범위를 공의 event에 집중하는 것으로 바꿔서 pass와 dribbling, shooting 그 외로 줄여 자동 라벨링 수행.
3. 모델 Baseline 추가: Conv1D, TCN, Transformer 모델로 수행.

### 5. 설치 및 실행 방법

#### Installation
FM in-game data 추출은 해당 게임이 있어야하고 선수와 공의 인게임 촬영 방식도 달라 제외한다.

Use conda to create a virtual environment and pip to install the requirements.
```
conda create --name fmMaster python==3.10.18
conda activate fmMaster
pip install -r requirements.txt
```
#### Usage
1. Download the raw data [here](https://springernature.figshare.com/articles/dataset/An_integrated_dataset_of_spatiotemporal_and_event_data_in_elite_soccer/28196177)
2. Remove invalid dataset start with ID DFL-MAT-J03WN1.
3. Run `train_valid.py`
```
cd your/path/to/fm_Master
```
#### 5.2. 오류 발생 시 해결 방법
floodlight.io 라이브러리를 사용할 때 파이썬 버전 충돌이 일어날 수 있는데, 그럴 때에는 파이썬을 지우고 버전을 3.9.xx로 낮춰서 진행한다.

### 6. 소개 자료 및 시연 영상
#### 6.1. 프로젝트 소개 자료
> [PPT]()

#### 6.2 프로젝트 홍보 영상
[![2025 전기 졸업과제 15 fm마스터](http://img.youtube.com/vi/DlT91KUyxKE/3.jpg)](https://www.youtube.com/watch?v=DlT91KUyxKE)    

### 7. 팀 구성
#### 7.1. 팀원별 소개 및 역할 분담
|학번|성명|역할|
|---|---|---|
|202055614|최성민|FM2024 인게임 Ball data 추출, 이벤트 예측 모델 설계 및 학습, Feature Engineering 및 labeling, github page 관리|
|202055502|강동권|FM2024 인게임 촬영 및 캡쳐링, 데이터셋 preprocessing. 모델 survey|

#### 7.2. 팀원 별 참여 후기
- 최성민: 스포츠 데이터를 딥러닝 모델 학습을 위해서 가공하는 것이 쉬운 일이 아니었다. 또한, 영상 캡처를 통해서 데이터셋을 만드는 과정이 생각한 것보다 쉽지 않았다. 산업체 멘토링을 통해서 조언받은대로 베이스라인 모델을 만들어서 같이 비교군을 만들어 연구를 하는 것이 도움이 되었다.
- 강동권: 평범한 게임이나 스포츠를 통해서 이러한 학문적인 가치를 찾아 연구를 하게 된 것이 뜻 깊었고, 앞으로 석사 연구에 큰 도움과 경험이 될 것 같다.
### 8. 참고 문헌 및 출처

**데이터 출처**: 본 프로젝트는 다음 논문에서 제공하는 공개 축구 트래킹 데이터를 활용합니다.  
> Bassek, M., Rein, R., Weber, H., & Memmert, D. (2025). *An integrated dataset of spatiotemporal and event data in elite soccer*. *Scientific Data*, 12(1), 195.
> [Metrica sports IO code](https://github.com/metrica-sports/sample-data)

---

## 🚧 진행 현황
- [x] 축구 트래킹 데이터 수집
- [x] 학습을 위해 기본 전처리 및 가공
- [x] 딥러닝 모델 설계 및 학습
- [x] 이벤트 예측 결과 분석
- [x] FM 인게임 데이터 수집 및 가공
- [x] Yolo 모델 학습

---

## 문의
- 작성자: fm마스터 - 팀장 202055614 최성민
- 이메일: csm010311@pusan.ac.kr
