# 1. 음악 장르 분류 - autogloun 기법

[https://dacon.io/competitions/official/236056/overview/description](https://dacon.io/competitions/official/236056/overview/description)

**주제/목표**

음악 장르를 분류하는 AI 알고리즘 개발

**데이터 : 음악 샘플의 특징 정보**

- **train (25383개)**
    - 25383개의 데이터
    - ID : 음악 샘플 고유 ID
    - `feature` : 음악 샘플의 특징을 담은 변수들
        - danceability, energy, key, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration (음악 재생 시간, 초)
    - `label` : 음악 장르 (총 15개)
        - Underground Rap, Dark Trap, trance, Hiphop, trap, techhouse, techno, psytrance, hardstyle, dnb, RnB, Trap Metal, Rap, Emo, Pop
- **test (16922개)**

### **코드 흐름**

**(1) autogluon 라이브러리 설치**

`!pip install autogluon`

**(2) TabularPredictor, Tabular Dataset import**

`from autogluon.tabular import TabularDataset, TabularPredictor`

TabularDataset : 데이터 로드에 사용

TabularPredictor : 예측 모델

**(3) 데이터 전처리**

drop ID

*ID drop 한 것 외에 별다른 전처리를 안 했음에도 불구, 좋은 점수가 나옴. 본인도 놀랐다고.. 

**(4) 모델 생성 및 학습**

label : genre, eval_metric : f1_macro, time_limit : 3600 * 3 (time limit을 늘리면 더 좋은 결과)

`predictor = TabularPredictor(label=label, eval_metric=eval_metric).fit(train_data, presets='best_quality', time_limit=time_limit)`

(presets : 성능 위주/적당한 성능과 추론 시간/빠른 추론 시간을 선택할 수 있음, ‘best_quality’ : 제일 좋은 지표가 나옴)

**(5) best model 확인**

leaderboard에서 확인 가능 (내림차순)

`predictor.leaderboard(silent = True)`

![image.png](image.png)

**(6) best model을 사용하여 예측**

get_model_best() : 제일 좋은 모델 사용

predict() : 예측

`model_to_use = predictor.get_model_best()
model_pred = predictor.predict(test_data, model=model_to_use)`

**(7) feature importance 확인**

`predictor.feature_importance(test_data)`

**사용한 모델 : autogluon**

<autogluon 장점 >

> robust
> 

> classifier, regression 모두 훈련이 가능
> 

> 시간 설정 가능
> 

> ensemble, stacking을 알아서 해줌
> 

<기존 방식>

- 데이터 가공 > 알고리즘 선택 > 하이퍼 파라미터 튜닝 (결과가 안 좋으면 다시 알고리즘 선택 단계로 돌아가서 새롭게 테스트, 무한 반복)

< automl 패키지 : ’autogluon’, ‘mljar-supervised’ >

- model.fit() 으로 가공부터 학습까지 자동적으로 처리해줌

< Autogluon의 학습 방식 >

- 데이터 전처리 > 학습 및 튜닝 > 모델 평가

**사용한 평가 지표 : f1_macro**

f1 score의 한 종류 (f1 score : macro average, weighted average, micro average 크게 3종류가 있음)

 

1. macro average: averaging the unweighted mean per label (라벨 별 f1-score 산술 평균)

2. weighted average: averaging the support-weighted mean per label

3. micro average: averaging the total true positives, false negatives and false positives

**차별점, 배울점**

최신 트렌드의 라이브러리 파악이 중요함

좋은 모델로 더 정확한 예측을 하려면 n시간 단위의 학습을 감내해야 한다는 것을 알게 되었다..
