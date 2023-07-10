# Data-Centric AI
> Boostcamp AI Tech 5기 Level 2 죠죠의 기묘한 모험

<br>

## 1. 개요

- Data-Centric의 취지에 맞게, **베이스라인 모델의 수정 없이 오로지 데이터의 수정**만으로 모델이 문장의 주제를 정확히 판단할 수 있도록 성능을 높이는 것이 이번 프로젝트의 목표.
- 자연어 독해 및 분석 과정을 거쳐 주어진 태스크를 수행하기 위해서는 자연어의 주제에 대한 이해가 필수적. Topic Classification 태스크는 모델이 자연어를 잘 이해하고 있는지 평가할 수 있는 가장 간단한 태스크.
- 그중에서도 KLUE-Topic Classification Benchmark는 뉴스의 헤드라인을 통해 그 뉴스가 어떤 Topic을 갖는지 분류해 내는 태스크. 각 자연어 데이터에는 생활문화(Society), 스포츠(Sports), 세계(World), 정치(Politics), 경제(Economy), IT 과학(IT/Science), 사회(Society) 등 다양한 주제 중 하나가 라벨링 되어 있음.
- 노이즈 데이터의 경우 전체 학습 데이터의 15%(총 6,852개). Noise Data 중 80%(5,481개)는 G2P를 이용한 Text Perturbation으로 생성되었으며, Prescriptive Pronunciation과 Descriptive Pronunciation을 1:1 비율로 사용하여 Noise를 생성. 나머지 20%(1,371개)는 Labeling 오류로 만들어진 데이터.

### A. 평가 기준

- KLUE-TC의 공식 리더보드와 동일한 평가 방법 적용
- F1 Score
    - Private F1: KLUE-TC dev 데이터에서 무작위로 50% 선정
    - Public F1: KLUE-TC dev 데이터에서 무작위로 50% 선정

### B. 멤버 및 역할

#### 1) 멤버
|전민수|조민우|조재관|진정민|홍지호|
|:-:|:-:|:-:|:-:|:-:|
|<img src='https://github.com/boostcampaitech5/level2_klue-nlp-11/assets/102800474/e1fd55d4-617a-436e-9ab0-e18eaeda685c' height=125 width=125></img>|<img src='https://github.com/boostcampaitech5/level2_klue-nlp-11/assets/102800474/1060e554-e822-4bac-9d7e-ddafdbf7d9c1' height=125 width=125></img>|<img src='https://github.com/boostcampaitech5/level2_klue-nlp-11/assets/102800474/5038030e-b30c-43e1-a930-3c63a1332843' height=125 width=125></img>|<img src='https://github.com/boostcampaitech5/level2_klue-nlp-11/assets/102800474/f871e7ea-7b41-494d-a858-2e6b2df815b9' height=125 width=125></img>|<img src='https://github.com/boostcampaitech5/level2_klue-nlp-11/assets/102800474/f5914167-bf44-40b6-8c78-964a8fb90b10' height=125 width=125></img>|
|[<img src='https://img.shields.io/badge/GitHub-181717?style&logo=github&logoColor=white' ></img>](https://github.com/line1029)|[<img src='https://img.shields.io/badge/GitHub-181717?style&logo=github&logoColor=white' ></img>](https://github.com/Minwoo0206)|[<img src='https://img.shields.io/badge/GitHub-181717?style&logo=github&logoColor=white' ></img>](https://github.com/jaekwanyda)|[<img src='https://img.shields.io/badge/GitHub-181717?style&logo=github&logoColor=white' ></img>](https://github.com/wjdals3406)|[<img src='https://img.shields.io/badge/GitHub-181717?style&logo=github&logoColor=white' ></img>](https://github.com/jiho-hong)|

#### 2) 역할
|이름|역할|
| --- | --- |
| 전민수 | Cleanlab, T5 모델 코드 리팩토링 |
| 조민우 | G2P Filtering, Cleanlab 적용 |
| 조재관 | LLM 활용한 합성데이터 제작 |
| 진정민 | T5 모델 Fine-tuning 및 Augmentation 진행, Cleanlab 적용 |
| 홍지호 | Data Augmentation, Data Cleaning |

### C. Skill

PyTorch, Pandas

<br>

## 2. Data Centric Methods

### A. Baseline Score
- `monologg/kobert` 모델 사용

|Accuracy|F1|
|-|-|
|0.8720|0.8698|

### B. Delete Noise

#### 1) Label Error Detection - Cleanlab

- Noise Data의 20%(1,371개)는 Labeling 오류가 포함되어 있기 때문에 Label Issue를 확인하는 `cleanlab`을 사용함
- 3-fold로 학습한 예측 결과(Probability)를 바탕으로 `cleanlab`을 적용
- Label Error를 모두 제거한 버전과 Label Error의 값을 Probability가 가장 높은 값으로 바꾼 두 가지 버전으로 실험을 진행함

    | |Accuracy|F1|
    |-|-|-|
    |Baseline|0.8720|0.8698|
    |Label Error를 모두 제거한 버전|0.8750|0.8753|
    |Label Error의 값을 Probability가 가장 높은 값으로 바꾼 버전|0.8748|0.8753|



#### 2) Data Cleaning

- 사용 배경
    - 학습 데이터의 문장에서 맞춤법 교정을 통해 데이터 정제하여 모델의 성능을 높이고자 함.
- 활용 방법
    1. `py-hanspell`
        
        네이버 맞춤법 검사기를 이용한 한글 맞춤법 검사 라이브러리.
        
        기존 데이터로 학습한 모델에 비해 성능 향상
        
        | |Accuracy|F1|
        |-|-|-|
        |Baseline|0.8720|0.8698|
        |py-hanspell|0.8814|0.8821|
        
    2. `symspell-ko`
        
        `symspellpy`을 한국어 특성에 맞게 음소 분해를 이용한 교정 라이브러리.
        
        문장에 포함된 노이즈를 교정해 주지 못해 데이터 생성 중단
        
        ```
        교정 전 문장: 어버이날 막따가 흐려저…남부지방 여튼 황사
        교정 후 문장: 어서 이빨 마 가가 그 로저 나무 지방 여튼 형사
        ```
        

#### 3) Data Filtering - Grapheme to Phoneme(G2P) Filtering

- 부산대학교에서 개발한 [한국어 맞춤법/문법 검사기](http://speller.cs.pusan.ac.kr/) 사용
- 띄어쓰기 교정이나 종결어미 교정을 제외한 의미 있는 교정 횟수를 계산
- 교정 횟수가 3회 이상인 경우 Filtering
- 초반 100개 데이터 기준 6개의 G2P Noise 데이터 중 5개를 Filtering

    | |Accuracy|F1|
    |-|-|-|
    |Baseline|0.8720|0.8698|
    |G2P|0.8752|0.8764|

### C. Data Augmentation

#### 1) Back Translation (T5)

- 사용 배경
    - 인코더-디코더 모델이기 때문에 입력 형태도 자연어로 받을 수 있고, 출력 또한 자연어로 생성하기 때문에 다양한 NLP 태스크에 적용 가능
    - 특히 서로 다른 두 언어에 대해 같은 의미를 가지는 텍스트를 각각 입력, 타깃으로 학습시켜 기계번역 모델에 사용이 가능
- 활용 방법
    - 한국어 → 영어 → 한국어로 Back Translation을 진행하기 위해 T5 모델을 활용
    - 한국어에 적합한 `KETI-AIR/ke-t5-small` 모델을 사용하여 Fine-tuning
    - 한국어 → 영어 / 영어 → 한국어 모델 두 개를 활용해서 번역을 진행, 데이터 총 100% 증강
    - Back Translation 데이터와 기존 데이터의 비율을 1:1로 두어 학습을 진행

#### 2) KoAlpaca

- 사용 배경
    - LLM을 활용해 합성 데이터를 생성하고자 함
    - 그중 한국어로 학습됐고 용량도 작은 `KoAlpaca`를 선택
- 활용 방법
    - 모델은 `beomi/KoAlpaca` 을 불러옴
    - 여러 가지 방법의 Prompt를 이용해서 데이터를 새롭게 생성
    - 하지만 비일관적이고 질적으로 좋지 않은 데이터가 대부분임

#### 3) 한국어 말뭉치 사용

- 사용 배경
    - 기존 데이터 세트에는 노이즈 데이터가 포함되어 있어 학습 시 모델 성능을 낮춤
- 활용 방법
    - `한국어 신문 말뭉치 2022`의 8,734,559개의 문장과 주제 데이터 추출
    - 기존 데이터 세트와 말뭉치 주제 데이터의 분류가 서로 다름
        
        ```
        기존 데이터 세트의 분류: 생활문화, 스포츠, 세계, 정치, 경제, IT 과학, 사회
        한국어 신문 말뭉치의 분류: 경제, 무역|경제, 지역, 대전|사회, IT_과학, 스포츠>농구_배구, 경제,자동차|경제 등
        ```
        
    - 말뭉치 주제 분류를 기존 데이터 세트에 맞게 재분류
        - `IT_과학,과학|사회` 와 같이 분류가 중복되는 데이터는 삭제
    - 데이터 분포를 고려하여 분류별 10,000개(총 10,000*7개), 40,000개(총 280,000*7개)의 데이터로 구성
    - 실험 결과

        | |Accuracy|F1|
        |-|-|-|
        |Baseline|0.8720|0.8698|
        |한국어 말뭉치(10,000개)|0.7105|0.7320|
        |한국어 말뭉치(40,000개)|0.7382|0.7476|
