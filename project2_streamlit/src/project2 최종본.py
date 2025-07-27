import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import joblib

# 저장된 모델과 전처리 객체 불러오기
model_lr = joblib.load("./model/model_lr.pkl")
model_lgb = joblib.load("./model/model_lgb.pkl")
model_cat = joblib.load("./model/model_cat.pkl")
num_features = joblib.load("./model/num_features.pkl")
feature_col = joblib.load("./model/feature_col.pkl")
scaler = joblib.load("./preprocessing/scaler.pkl")
encoder = joblib.load("./preprocessing/encoder.pkl")


# 한글 폰트 설정
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

# 페이지 설정
st.set_page_config(page_title="고객 이탈 분석", layout="wide")

# 세션 상태로 페이지 이동 관리 함수 정의
def navigate(target):
    st.session_state.page = target

# 초기 세션 상태 설정
if 'page' not in st.session_state:
    st.session_state.page = "홈"

# 사이드바 네비게이션
# 페이지 리스트
pages = ["홈", "데이터 탐색 (EDA)", "모델링", "예측 시뮬레이션", "인사이트 요약", "참고자료 / 팀 소개"]
icons = ["🏠", "📊", "🧠", "🔮", "💡", "📚"]

# 사이드바 제목
st.sidebar.markdown("### 🌐 페이지 이동")

# 버튼형 메뉴
for i, p in enumerate(pages):
    if st.sidebar.button(f"{icons[i]} {p}"):
        st.session_state.page = p

# 기본값
if "page" not in st.session_state:
    st.session_state.page = "홈"

page = st.session_state.page

# 데이터 로딩 함수 (캐시 적용)
@st.cache_data
def load_data():
    return pd.read_csv("./data/cell2celltrain.csv")

df = load_data()

def preprocess_and_predict(df_raw):
    df = df_raw.copy()
    df = df[feature_col]
    df = df.dropna().reset_index(drop=True)

    # ✅ binary encoding
    binary_cols = [
        'Churn', 'ChildrenInHH', 'HandsetRefurbished', 'HandsetWebCapable', 'TruckOwner', 'RVOwner',
        'BuysViaMailOrder', 'RespondsToMailOffers', 'OptOutMailings', 'NonUSTravel',
        'OwnsComputer', 'HasCreditCard', 'NewCellphoneUser', 'NotNewCellphoneUser',
        'OwnsMotorcycle', 'MadeCallToRetentionTeam'
    ]
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0})

    if 'Homeownership' in df.columns:
        df['Homeownership'] = df['Homeownership'].map({'Known': 1, 'Unknown': 0})

    # ✅ 라벨 인코딩
    label_cols = ['ServiceArea', 'PrizmCode', 'Occupation']
    for col in label_cols:
        if col in df.columns:
            try:
                df[col] = encoder[col].transform(df[col].astype(str))
            except Exception as e:
                raise ValueError(f"⚠ 라벨 인코딩 실패: '{col}' 컬럼 처리 중 오류\n{str(e)}")

    # ✅ 기타 인코딩 및 변환
    if 'HandsetPrice' in df.columns:
        df['HandsetPrice'] = df['HandsetPrice'].replace('Unknown', 0).astype(float)

    if 'CreditRating' in df.columns:
        df['CreditRating'] = df['CreditRating'].astype(str).str.split('-').str[0].astype(int)

    if 'MaritalStatus' in df.columns:
        df['MaritalStatus'] = df.apply(
            lambda row: 'Yes' if row['MaritalStatus'] == 'Unknown' and row['AgeHH2'] != 0 else (
                -1 if row['MaritalStatus'] == 'Unknown' else row['MaritalStatus']
            ), axis=1
        )
        df['MaritalStatus'] = df['MaritalStatus'].map({'Yes': 1, 'No': 0, -1: -1})

    # ✅ 스케일링
    df_num = df[num_features]
    df_cat = df.drop(columns=num_features)

    df_scaled = pd.DataFrame(
        scaler.transform(df_num), columns=num_features, index=df.index
    )
    df_final = pd.concat([df_scaled, df_cat], axis=1)

    # ✅ 예측
    proba_lgb = model_lgb.predict_proba(df_final)[:, 1]
    proba_cat = model_cat.predict_proba(df_final)[:, 1]
    proba_lr = model_lr.predict_proba(df_final)[:, 1]

    avg_proba = np.mean([proba_lgb, proba_cat, proba_lr], axis=0)
    pred = (avg_proba > 0.5).astype(int)

    df_result = df_raw.copy().dropna().reset_index(drop=True) 
    df_result["Churn_Probability (%)"] = np.round(avg_proba * 100, 2)
    df_result["Churn_Prediction"] = pred

    return df_result

# 1️⃣ 홈 페이지
if page == "홈":
    st.title("📱 통신사 고객 이탈 분석 대시보드")

    st.markdown("""
    ### 📌 프로젝트 개요
    - **목표:** 고객 이탈 유무 예측 및 요인 분석
    - **데이터셋:** Cell2Cell 통신사의 고객 이탈 관련 데이터
    - **활용:** 비즈니스 전략 수립 및 고객 유지 전략 개발
    """)

    st.markdown("---")
    st.subheader("🔍 페이지 구성")
    col1, col2 = st.columns(2)

    with col1:
        st.info("**1. 데이터 탐색 (EDA)**\n- 고객 분포\n- 이탈 여부별 특성 비교\n- 상관관계 분석")
        st.button("👉 EDA 페이지로 이동", on_click=navigate, args=("데이터 탐색 (EDA)",))

        st.info("**2. 모델링**\n- 로지스틱 회귀, 랜덤포레스트, XGBoost\n- 성능 평가 및 ROC Curve")
        st.button("👉 모델링 페이지로 이동", on_click=navigate, args=("모델링",))

        st.info("**3. 예측 시뮬레이션**\n- 사용자 입력 기반 이탈 예측\n- 이탈 확률과 간단한 해석 제공")
        st.button("👉 예측 시뮬레이션 이동", on_click=navigate, args=("예측 시뮬레이션",))

        st.info("**4. 인사이트 요약**\n- 주요 변수 정리\n- 비즈니스 전략 제안\n- 리포트 다운로드")
        st.button("👉 인사이트 요약 이동", on_click=navigate, args=("인사이트 요약",))

        st.info("**5. 참고자료 / 팀 소개**\n- 데이터 출처\n- 팀 구성 및 깃허브 링크")
        st.button("👉 참고자료 / 팀 소개 이동", on_click=navigate, args=("참고자료 / 팀 소개",))

# 2️⃣ 데이터 탐색 (EDA)
elif page == "데이터 탐색 (EDA)":
    st.header("📊 데이터 탐색 (EDA)")
    st.subheader("✅ 데이터 샘플")
    st.dataframe(df.head())

    tab1, tab2, tab3 = st.tabs(["📊 고객 분포", "📉 이탈 특성 비교", "🔐 상관관계 분석"])

    with tab1:
        st.subheader("고객 이탈 여부 분포 그래프")
        churn_counts = df["Churn"].value_counts()
        st.bar_chart(churn_counts)
        fig1, ax1 = plt.subplots()
        st.subheader("고객 이탈 여부 분포 비율")
        ax1.pie(churn_counts, labels=churn_counts.index, autopct="%1.1f%%", startangle=90)
        ax1.axis("equal")
        st.pyplot(fig1)

    with tab2:
        st.subheader("📊 특성별 이탈 여부 분포")

        target_cols = ['MonthsInService', 'CurrentEquipmentDays', 'TotalRecurringCharge', 'MonthlyMinutes']
        bins = 40

        for i, target_col in enumerate(target_cols):
            if target_col == 'MonthsInService':
                # ✅ 선 그래프: MonthsInService에 대한 이탈률 분포
                st.markdown("#### ✅ 서비스 개월 수에 따른 이탈 고객 분포")

                # 1. 분포 계산
                count_df = df.groupby([target_col, 'Churn']).size().reset_index(name='Count')
                pivot_df = count_df.pivot(index=target_col, columns='Churn', values='Count').fillna(0)

                # 2. 정규화
                pivot_df['Yes_pct_norm'] = pivot_df['Yes'] / pivot_df['Yes'].sum()
                pivot_df['No_pct_norm'] = pivot_df['No'] / pivot_df['No'].sum()

                # 3. 선 그래프 시각화
                fig1, ax1 = plt.subplots(figsize=(12, 6))
                ax1.plot(pivot_df.index, pivot_df['Yes_pct_norm'], label='Churn = Yes (Normalized)', linewidth=2)
                ax1.plot(pivot_df.index, pivot_df['No_pct_norm'], label='Churn = No (Normalized)', linewidth=2)
                ax1.set_title('서비스 개월 수에 따른 이탈/유지 고객 비율')
                ax1.set_xlabel('Months In Service')
                ax1.set_ylabel('Proportion (Normalized)')
                ax1.legend()
                ax1.grid(True)
                st.pyplot(fig1)

            else:
                # ✅ 막대 그래프: 나머지 변수
                st.markdown(f"#### ✅ {target_col}에 따른 이탈 고객 상대도수")

                # 구간 경계 및 라벨
                min_val = df[target_col].min()
                max_val = df[target_col].max()
                bin_edges = np.linspace(min_val, max_val, bins + 1)
                labels = [f"{int(bin_edges[i])}-{int(bin_edges[i+1])}" for i in range(len(bin_edges) - 1)]

                # 구간화
                df[f'{target_col}_bin'] = pd.cut(df[target_col], bins=bin_edges, labels=labels, include_lowest=True)

                # 상대도수 계산
                rel_freq_df = pd.crosstab(
                    df[f'{target_col}_bin'], df["Churn"], normalize='columns'
                ).reset_index()

                # x축 위치
                x = np.arange(len(rel_freq_df[f'{target_col}_bin']))
                yes_vals = rel_freq_df['Yes']
                no_vals = rel_freq_df['No']
                width = 0.4

                # 시각화
                fig, ax = plt.subplots(figsize=(14, 6))
                ax.bar(x - width / 2, yes_vals, width, label='Churn = Yes', alpha=0.8)
                ax.bar(x + width / 2, no_vals, width, label='Churn = No', alpha=0.8)
                ax.set_title(f"{target_col} 구간별 이탈/유지 고객 비율")
                ax.set_xlabel(target_col)
                ax.set_ylabel("Proportion")
                ax.set_xticks(x)
                ax.set_xticklabels(rel_freq_df[f'{target_col}_bin'], rotation=90, fontsize=8)
                ax.legend(title="Churn")
                ax.grid(axis='y', linestyle='--', alpha=0.6)
                plt.tight_layout()

                st.pyplot(fig)




    with tab3:
        numeric_df = df.select_dtypes(include=["int64", "float64"])
        corr = numeric_df.corr()
        fig4, ax4 = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, cmap="coolwarm", annot=False, ax=ax4)
        st.pyplot(fig4)

# 3️⃣ 모델링 페이지
elif page == "모델링":
    st.header("🤖 모델 성능 요약 (Soft Voting Classifier)")

    data = {
    "Models": ["XGBoost", "LGBM+CatBoost+Logistic"],
    "Train Accuracy": [0.8338, 0.7132],
    "Test Accuracy": [0.7162, 0.6388],
    "Train F1": [0.7564, 0.6020],
    "Test F1": [0.5595, 0.5029],
    "Train Recall": [0.7254, 0.7265],
    "Test Recall": [0.5644, 0.6386]
}

    # 인덱스를 제거한 DataFrame
    df1 = pd.DataFrame(data)

    # 인덱스를 제거하고 표 출력
    st.subheader("📋 모델 성능 비교")
    df1_reset = df1.reset_index(drop=True)
    st.table(df1_reset)

    st.subheader("📊 시각화 결과")

    col1, col2 = st.columns(2)

    with col1:
        st.image("./result_img/Confusion_Mateix.png", caption="Confusion Matrix - Soft Voting", use_container_width=True)

    with col2:
        st.image("./result_img/ROC.png", caption="ROC Curve - Soft Voting Classifier", use_container_width=True)



# 4️⃣ 예측 시뮬레이션 페이지
elif page == "예측 시뮬레이션":
    st.header("🔮 CSV 업로드 기반 고객 이탈 예측")
    st.markdown("""
    ### 📎 업로드 안내
    - **필수 컬럼 수:** 57개 컬럼이 포함되어야 합니다.
    - **필수 컬럼:** 모델 학습 시 사용된 컬럼들과 동일해야 합니다.
    - **결측치:** `NaN` 또는 `Unknown` 등의 결측치가 있을 경우 해당 행은 자동 제거됩니다.
    - **형식:** `csv` 파일 형식만 지원합니다. (최대 200MB)
    """)

    uploaded_file = st.file_uploader("CSV 파일 업로드", type=["csv"])
    if uploaded_file:
        try:
            df_input = pd.read_csv(uploaded_file)
            result_df = preprocess_and_predict(df_input)

            st.success("✅ 예측 완료!")
            st.dataframe(result_df.head())

            csv_result = result_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 결과 다운로드 (CSV)",
                data=csv_result,
                file_name="churn_prediction_result.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"❌ 처리 중 에러 발생: {str(e)}")
    else:
        st.info("CSV 파일을 업로드하면 이탈 예측 결과를 확인할 수 있습니다.")


# 5️⃣ 인사이트 요약 페이지
elif page == "인사이트 요약":
    st.header("📘 인사이트 요약 및 비즈니스 제안")
    st.markdown("""
    ### 📌 주요 인사이트
    - 이탈률은 가입 후 약 2~12개월 사이가 가장 높음
    - 기기 사용 일수(CurrentEquipmentDays) 가 늘어날수록 이탈률이 증가
    - 낮은 요금 고객(RecurringCharge) 이 이탈 비율이 높다.
    - 사용량이 적은 고객(MonthlyMinutes) 이 이탈 비율이 높음.

    ### 💡 비즈니스 제안
    - 가입 6~12개월 고객에 대한 리텐션 프로그램 필수
    - 기기 교체 시점 도달 고객에게 새 기기 업그레이드 제안
    - 요금이 낮고 사용량이 적은 고객을 위한 프로모션 강화
    - 고객 사용량 급감 감지 시 자동 알림 or 혜택 제공
    """)

# 6️⃣ 참고자료 / 팀 소개 페이지
elif page == "참고자료 / 팀 소개":
    st.header("📚 참고자료 & 팀 소개")
    st.markdown("""
    ### 🔗 데이터 출처
    - [Kaggle: Cell2Cell Dataset](https://www.kaggle.com/datasets)

    ### 👨‍💻 팀 구성
    - 기현택 (데이터 분석(EDA))
    - 최서린(머신러닝 모델 학습 및 평가)    
    - 이소정 (전처리 및 피처 엔지니어링)
    - 임가은 (데이터 분석(EDA))
    - 김주형 (Streamlit 웹 서비스 구현)

    ### 📬 GitHub
    - [프로젝트 저장소](https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN15-2nd-1Team)
    """)
