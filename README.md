# SKN15-2nd-1Team


# 1. 팀 소개
<div align="center">
<img width="256" height="256" alt="Image" src="https://github.com/user-attachments/assets/fed6f75a-fed1-4ba4-b815-351edf2ed29a" />


일단하조
  
| 기현택     | 김주형     | 이소정     | 임가은     | 최서린     |
| ---------- | ---------- | ---------- | ---------- | ---------- |
|[@mathplanet](https://github.com/mathplanet)|[@wngud09](https://github.com/wngud09)|[@leesojunghub](https://github.com/leesojunghub)|[@mars7421](https://github.com/mars7421)|[@seorinchoi](https://github.com/seorinchoi)|

</div>

# 2. 프로젝트 기간
2025년 7월 10일 ~ 2025년 7월 11일 (2일)

# 3. 프로젝트 개요

## 📕 프로젝트명
통신사 이탈 고객 방지 프로젝트

## ✅ 프로젝트 배경 및 소개

현재 국내 통신 산업은 극심한 경쟁 환경에 직면해 있습니다. 거시경제의 전반적인 성장 둔화와 통신 시장의 포화로 인해 매출 성장률은 정체되어 있으며, 이러한 현상은 국내뿐 아니라 전 세계적으로도 유사하게 나타나고 있습니다. 통신 서비스는 가입자에게 대체 가능한 일반재(commodity)로 인식되기 쉬워, 가격 외에도 고객 경험, 서비스 품질, 혜택 등 다방면에서의 차별화 전략이 기업 생존에 필수적인 요소로 떠오르고 있습니다. 이에 따라 통신사업자 간의 경쟁은 앞으로도 더욱 심화될 것으로 예상됩니다. [삼일회계법인, 「통신산업 경쟁 현황 : 경쟁우위를 위한 5가지 방안](https://www.pwc.com/kr/ko/insights/industry-focus/samilpwc_state-of-competition-and-commoditisation.pdf)

이러한 환경 속에서 통신사는 기존 고객을 유지하고 신규 가입자를 확보하기 위해 보다 정교한 데이터 기반 전략이 요구됩니다. 특히, 머신러닝 기반의 분석 기법은 고객 이탈 현상을 사전에 예측하고, 이탈에 영향을 미치는 주요 특성을 파악하는 데 효과적인 도구로 활용될 수 있습니다.

이러한 배경을 바탕으로, Kaggle에서 제공하는 [Cell2Cell Churn Prediction 데이터셋](https://www.kaggle.com/datasets/jpacse/datasets-for-churn-telecom)을 활용하여, 통신사 고객의 이탈 여부를 예측하고 이에 영향을 미치는 다양한 특성을 분석하는 웹 서비스를 제작하였습니다. 

## ❤️ 기대효과

해당 모델을 통해 이탈 가능성이 높은 고객을 조기에 파악함으로써, 기업은 고객 이탈을 방지하기 위한 장단기적인 맞춤형 마케팅 및 유지 전략을 세울 수 있습니다. 또, 기업은 자사 데이터를 활용하여 이탈 가능성이 높은 고객에 대해 맞춤형 서비스를 제공할 수도 있을 것입니다.
이에 더해, 이탈 가능성이 낮은 고객에 대한 과도한 마케팅 비용을 줄이고, 이탈 가능성이 높은 고객에게 자원을 집중함으로써 비용 대비 효율을 극대화할 수 있습니다.
추후에 해당 모델은 통신사 외에도 보험, 금융, OTT 등 고객 유지가 핵심인 산업 전반에 확장 적용할 수 있는 기반이 될 수 있습니다.

이 서비스의 예상 사용자인 통신사 CRM 및 마케팅 부서 담당자는 고객 이탈 가능성이 높은 세그먼트를 식별하여 개인화된 유지 프로모션을 제공할 수 있고, 이탈 예측 점수를 활용해 우선 대응 고객 리스트를 자동 생성되도록 할 수 있습니다. 또, CRM 개발자 및 운영자는 예측 모델 결과를 CRM 시스템에 통합하여 실시간 이탈 경고 시스템을 구현할 수 있으며, REST API 또는 배치 시스템을 통해 정지적으로 이탈 예측 점수 갱신을 자동화할 수 있습니다.


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

|최적화(전처리 전)|최적화(전처리 후)|
| ---------- | ---------- |
|<img width="500" height="300" alt="Image" src="https://github.com/user-attachments/assets/0f7b429b-e8df-48b5-9e34-fa474d0e96c5" />|<img width="500" height="300" alt="Image" src="https://github.com/user-attachments/assets/1bef460f-aa31-4c43-bf82-c2ad38e17010" />|



### 컬럼 내 전처리 방법
|0,1 이진 분류|라벨|기타|
| ---------- | ---------- | ---------- |
|MadeCallToRetentionTeam, Churn, ChildrenInHH, Homeownership, OptOutMailings, HandsetWebCapable, HandsetRefurbished, OwnsComputer, BuysViaMailOrder, HandsetPrice|ServiceAre, PrizmCode, Occupation|CreditRating-문자 제거|

### 모델 별 성능 작업
|결측치|이상치|인코딩|파라미터 튜닝|샘플링|
| ---------- | ---------- | ---------- | ---------- | ---------- |
|결측치의 갯수가 많지 않아 일괄적으로 처음에는 평균값으로 바꾸어 계산 -> 성능에 큰 영향이 없어 Dropna로 일괄 처리|이상치는 초반에는 Minmax랑 Standard 스케일러를 사용했고, 이후 Robust를 사용|초반에는 Ordinal 사용 -> 성능 문제로 label과 Frequency 사용|중반 이후, Grid-Search 튜닝 도입|Over, Under, Class_Weight 도입|



# 6. 사용모델
|모델| LightGBM| XGBoost |CatBoost|RandomForest|
| ---------- | ---------- | ---------- | ---------- | ---------- |
|특징|Gradient Boosting 기반의 트리 모델로 매우 빠르고 메모리 효율이 좋음|트리 기반의 부스팅 알고리즘, overfitting을 방지하는 정규화 포함|범주형 데이터 처리에 특화된 Gradient Boosting 모델|여러 결정트리를 결합한 앙상블 모델 (bagging 기반)|
|장점|대용량 데이터 처리에 강함, 카테고리형 데이터 자동 처리 지원|성능이 우수하며 다양한 설정 가능|전처리 없이 범주형 변수 지원, 빠른 학습, 안정적인 성능|과적합에 강함, 변수 중요도 제공|
|단점|소규모 데이터에서는 과적합 가능성 있음|하이퍼파라미터가 많고, 튜닝에 시간이 걸릴 수 있음|모델 해석력이 부족할 수 있음|예측이 느릴 수 있고, 매우 큰 데이터에서는 비효율적일 수 있음|

|모델| Logistic Regression| SVM (Support Vector Machine)|TabNet|SoftVoting|
| ---------- | ---------- | ---------- | ---------- | ---------- |
|특징|선형 회귀 기반의 확률적 분류 모델|마진 최대화를 통한 이진 분류, 고차원 공간 매핑 가능 (커널)|딥러닝 기반의 테이블형 데이터 전용 모델, Attention 사용|여러 모델의 클래스별 확률 예측값을 평균해 가장 높은 확률의 클래스로 결정|
|장점|빠르고 해석이 쉬움, 베이스라인 모델로 적합|복잡한 분류 문제에 강함, 소규모 데이터에 적합|feature selection이 자동으로 이뤄짐, end-to-end 학습|확률 정보를 활용해 더 정교하고 안정적인 예측 가능|
|단점|비선형 관계를 잘 포착하지 못함|대용량 데이터에 비효율적, 커널 선택에 민감|학습 시간 오래 걸림, 데이터 정규화나 튜닝 필요|모델들이 확률을 잘 추정하지 못하면 성능이 떨어질 수 있음|

# 7. 프로젝트 결과
<img width="990" height="645" alt="Image" src="https://github.com/user-attachments/assets/5355337f-8990-4750-9c22-ec573086d27e" />
<img width="978" height="792" alt="Image" src="https://github.com/user-attachments/assets/60a61b13-df6e-4f7e-b1b6-162ee6d886fc" />
<img width="973" height="698" alt="Image" src="https://github.com/user-attachments/assets/103faae0-f57c-4de2-88ac-8d23267abb38" />
<img width="977" height="702" alt="Image" src="https://github.com/user-attachments/assets/b7a4081d-ce65-4495-9417-f0331b5af338" />
<img width="1102" height="697" alt="Image" src="https://github.com/user-attachments/assets/5d84fa3d-3bf2-4a22-b2ac-b4f90a25ec6c" />
<img width="1116" height="381" alt="Image" src="https://github.com/user-attachments/assets/73aef602-e506-4df5-8eaa-df58bfc35852" />

| Models                  | Train Accuracy | Test Accuracy | Train F1 | Test F1 | Train Recall | Test Recall |
|-------------------------|----------------|----------------|----------|---------|---------------|--------------|
| XGBoost                 | 0.8338         | 0.7162         | 0.7564   | 0.5595  | 0.7254        | 0.5644       |
| LGBM+CatBoost+Logistic  | 0.7132         | 0.6388         | 0.6020   | 0.5029  | 0.7265        | 0.6386       |

# 8. 한 줄 회고

| 기현택     | 김주형     | 이소정     |
| ---------- | ---------- | ---------- |
|프로젝트에서 처음 팀장을 맡았는데, 떨리기도 하고 걱정도 되었지만 무사히 잘 마무리 된 것 같습니다. 팀원들과 소통하고 협업하고 문제를 발견하고 해결하는 과정 자체가 아주 소중한 경험이었고 모든 팀원들 고생하셨습니다.||기대만큼 성능이 오르지 않아 아쉬움이 남았다. 다양한 시도를 하기엔 시간이 부족했지만, 다음 프로젝트에서는 더 나은 전략과 시간 관리로 성능 개선을 이끌어내고 싶다.|

| 임가은     | 최서린     |
| ---------- | ---------- |
|모델 성능을 끌어올리기 위해 많은 경우의 수를 생각하여 돌렸지만, 시간이 이틀밖에 되지 않아 많이 아쉬웠다. 다음엔 EDA나 전처리 과정을 더 공부해서 데이터를 분석해 보고 싶다.|좋은 팀원들과 같이 해서 너무 좋았다! 모델 최적화는 정말 쉽지 않은 작업이라는 걸 다시 깨달았다... 다음엔 더 성능 좋은 모델을 만들 수 있도록 최선을 다하고 싶다!|



