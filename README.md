# SKN15-2nd-1Team


# 1. 팀 소개
<div align="center">
<img width="256" height="256" alt="Image" src="https://github.com/user-attachments/assets/fed6f75a-fed1-4ba4-b815-351edf2ed29a" />



  
| 기현택     | 김주형     | 이소정     | 임가은     | 최서린     |
| ---------- | ---------- | ---------- | ---------- | ---------- |
|[@mathplanet](https://github.com/mathplanet)|[@wngud09](https://github.com/wngud09)|[@leesojunghub](https://github.com/leesojunghub)|[@mars7421](https://github.com/mars7421)|[@seorinchoi](https://github.com/seorinchoi)|

</div>

# 2. 프로젝트 기간
2025년 7월 10일 ~ 2025년 7월 11일 (2일)

# 3. 프로젝트 개요

## 📕 프로젝트명
통신사 이탈 고객 방지 프로젝트

## ✅ 프로젝트 배경 및 목적
https://www.kaggle.com/datasets/jpacse/datasets-for-churn-telecom - Teradata center for customer relationship management at Duke University.
Cell2Cell 통신사의 고객 이탈 관련 데이터로 듀크 대학에서 수집한 데이터를 캐글을 통해 얻게 되었습니다. Cell2Cell 통신사는 가상의 회사로 실제 기업의 데이터를 익명화를 통해 공개되는 경우가 많습니다.

## 🖐️ 프로젝트 소개
이 데이터에는 50여개가 넘는 컬럼이 존재하며, 이를 통해 데이터 간의 상관관계를 분석하고 이상치, 결측치를 비롯한 EDA를 통해 고객 이탈에 어느 연관이 있는지 머신러닝 모델을 통해 알아보고자 합니다.

## ❤️ 기대효과
약 5만 건의 데이터를 통한 머신러닝 모델 훈련과 이를 바탕으로 2만건의 테스트를 통해 모델의 성능을 평가합니다. 이를 통해, 통신사 고객의 이탈 이유를 정밀하게 분석하고 이를 통해 기업은 자사 고객의 유출을 막기 위한 전략을 세울 수 있을 것입니다.


## 👤 대상 사용자
통신사의 CEO와 임원진은 데이터의 분석을 통해, 장단기적으로 다양한 전략을 준비하여 고객 이탈을 최소화하는 방법을 찾을 수 있게 될 것입니다.


# 4. 기술 스택
### --Environment--
<img src="https://img.shields.io/badge/github-181717?style=for-the-badge&logo=github&logoColor=white">
<img src="https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white">
<img src="https://img.shields.io/badge/Visual Studio Code-61DAFB?style=for-the-badge&logo=VisualStudioCode&logoColor=white">
<img src="https://img.shields.io/badge/figma-FFCA28?style=for-the-badge&logo=figma&logoColor=white">
<img src="https://img.shields.io/badge/streamlit-7952B3?style=for-the-badge&logo=streamlit&logoColor=white">
<img src="https://img.shields.io/badge/kaggle-003545?style=for-the-badge&logo=kaggle&logoColor=white">
<img src="https://img.shields.io/badge/Google Colab-F9AB00?style=for-the-badge&logo=GoogleColab&logoColor=white">


### --Communication--
<img src="https://img.shields.io/badge/Discord-02569B?style=for-the-badge&logo=Discord&logoColor=white">
<img src="https://img.shields.io/badge/Notion-F7DF1E?style=for-the-badge&logo=notion&logoColor=black">



# 5. EDA 및 데이터 전처리

|컬럼 별 원형비교|컬럼 별 상대도수|컬럼 별 비율비교|
| ---------- | ---------- | ---------- |
|<img width="300" height="300" alt="Image" src="https://github.com/user-attachments/assets/3df9ec57-a54d-48b6-b203-ebc81a54673d" />|<img width="300" height="300" alt="Image" src="https://github.com/user-attachments/assets/32d6bafa-346a-4316-9b5e-b1d73bb6e712" />|<img width="300" height="300" alt="Image" src="https://github.com/user-attachments/assets/d740e681-d84d-4134-ad7b-5553d03b000b" />|



|히트맵|이상치 박스플롯|
| ---------- | ---------- |
|<img width="450" height="450" alt="Image" src="https://github.com/user-attachments/assets/19e6eb0b-6e0c-4cc3-9b5c-38201a57ecfc" />|<img width="450" height="450" alt="Image" src="https://github.com/user-attachments/assets/456537bc-0b9c-44a0-adf5-97689b736894" />|

### 컬럼 내 전처리 방법
|0,1 이진 분류|라벨|기타|
| ---------- | ---------- | ---------- |
|MadeCallToRetentionTeam, Churn, ChildrenInHH, Homeownership, OptOutMailings, HandsetWebCapable, HandsetRefurbished, OwnsComputer, BuysViaMailOrder, HandsetPrice|ServiceAre, PrizmCode, Occupation|CreditRating-문자 제거|

### 모델 별 성능 작업
|결측치|이상치|인코딩|파라미터 튜닝|
| ---------- | ---------- | ---------- | ---------- |
|결측치의 갯수가 많지 않아 일괄적으로 처음에는 평균값으로 바꾸어 계산 -> 성능에 큰 영향이 없어 Dropna로 일괄 처리|이상치는 초반에는 Minmax랑 Standard 스케일러를 사용했고, 이후 Robust를 사용|초반에는 Ordinal 사용 -> 성능 문제로 label과 Frequency 사용|중반 이후, Grid-Search 튜닝 도입|



# 6. 사용모델
|모델| LightGBM| XGBoost |CatBoost|RandomForest|
| ---------- | ---------- | ---------- | ---------- | ---------- |
|특징|Gradient Boosting 기반의 트리 모델로 매우 빠르고 메모리 효율이 좋음|트리 기반의 부스팅 알고리즘, overfitting을 방지하는 정규화 포함|범주형 데이터 처리에 특화된 Gradient Boosting 모델|여러 결정트리를 결합한 앙상블 모델 (bagging 기반)|
|장점|대용량 데이터 처리에 강함, 카테고리형 데이터 자동 처리 지원|성능이 우수하며 다양한 설정 가능|전처리 없이 범주형 변수 지원, 빠른 학습, 안정적인 성능|과적합에 강함, 변수 중요도 제공|
|단점|소규모 데이터에서는 과적합 가능성 있음|하이퍼파라미터가 많고, 튜닝에 시간이 걸릴 수 있음|모델 해석력이 부족할 수 있음|예측이 느릴 수 있고, 매우 큰 데이터에서는 비효율적일 수 있음|

|모델| Logistic Regression| SVM (Support Vector Machine)|TabNet|
| ---------- | ---------- | ---------- | ---------- |
|특징|선형 회귀 기반의 확률적 분류 모델|마진 최대화를 통한 이진 분류, 고차원 공간 매핑 가능 (커널)|딥러닝 기반의 테이블형 데이터 전용 모델, Attention 사용|
|장점|빠르고 해석이 쉬움, 베이스라인 모델로 적합|복잡한 분류 문제에 강함, 소규모 데이터에 적합|feature selection이 자동으로 이뤄짐, end-to-end 학습|
|단점|비선형 관계를 잘 포착하지 못함|대용량 데이터에 비효율적, 커널 선택에 민감|학습 시간 오래 걸림, 데이터 정규화나 튜닝 필요|
# 7. 프로젝트 결과

# 8. 한 줄 회고

| 기현택     | 김주형     | 이소정     |
| ---------- | ---------- | ---------- |
||||

| 임가은     | 최서린     |
| ---------- | ---------- |
|안녕|이거|



