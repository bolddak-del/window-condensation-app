import streamlit as st
import requests
import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# -------------------------------------------------
# 0. 페이지 설정
# -------------------------------------------------

st.set_page_config(
    page_title="창호 결로 위험 예측",
    layout="wide"
)

st.title("기상청 예보 기반 단일창 결로 위험 예측")
st.markdown(
    "<div style='font-size:22px; font-weight:600;'>인하대학교 건축환경 · 열환경 프로젝트</div>",
    unsafe_allow_html=True
)
st.caption("기상청 외기온도 예보와 실내 온습도 정보를 이용해 단일창의 표면온도 변화를 예측합니다.")



# -------------------------------------------------
# 1. 노점온도 계산 함수
# -------------------------------------------------

def calculate_dew_point(Ti, RH):
    a = 17.27
    b = 237.7

    gamma = ((a * Ti) / (b + Ti)) + math.log(RH / 100)
    Tdp = (b * gamma) / (a - gamma)

    return Tdp


# -------------------------------------------------
# 2. 기상청 API 설정
# -------------------------------------------------

KMA_SERVICE_KEY = st.secrets["KMA_SERVICE_KEY"]

# 인천 미추홀구 격자좌표
nx = 54
ny = 125


def get_base_date_time():
    now = datetime.now(ZoneInfo("Asia/Seoul")) - timedelta(minutes=10)

    # 기상청 단기예보 발표시각
    base_times = ["0200", "0500", "0800", "1100", "1400", "1700", "2000", "2300"]
    current_time = now.strftime("%H%M")

    available_times = [t for t in base_times if t <= current_time]

    if available_times:
        base_date = now.strftime("%Y%m%d")
        base_time = available_times[-1]
    else:
        yesterday = now - timedelta(days=1)
        base_date = yesterday.strftime("%Y%m%d")
        base_time = "2300"

    return base_date, base_time


def fetch_kma_forecast(nx, ny):
    base_date, base_time = get_base_date_time()

    url = "https://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"

    params = {
        "serviceKey": KMA_SERVICE_KEY,
        "pageNo": "1",
        "numOfRows": "1000",
        "dataType": "JSON",
        "base_date": base_date,
        "base_time": base_time,
        "nx": nx,
        "ny": ny,
    }

    response = requests.get(url, params=params)

    st.write("응답 상태코드:", response.status_code)

    return response.json(), base_date, base_time


# -------------------------------------------------
# 3. 기상청 예보자료 파싱
# -------------------------------------------------

def parse_weather(data):
    items = data["response"]["body"]["items"]["item"]

    result = []

    for item in items:
        category = item["category"]

        if category in ["TMP", "REH", "TMN", "TMX"]:
            result.append({
                "예보날짜": item["fcstDate"],
                "예보시간": item["fcstTime"],
                "항목": category,
                "값": item["fcstValue"]
            })

    return result


def extract_5h_outdoor_temps(weather):
    # TMP = 1시간 기온
    tmp_list = [x for x in weather if x["항목"] == "TMP"]

    # 날짜와 시간 순서대로 정렬
    tmp_list = sorted(tmp_list, key=lambda x: (x["예보날짜"], x["예보시간"]))

    # 앞으로 5시간만 사용
    tmp_5h = tmp_list[:5]

    outdoor_temps = [float(x["값"]) for x in tmp_5h]

    return outdoor_temps, tmp_5h


# -------------------------------------------------
# 4. 단일창 1차원 비정상 열전도 계산
# -------------------------------------------------

def simulate_glass_temperature(Ti, Ts0, outdoor_temps):
    # -----------------------------
    # 단일창 유리 물성 및 가정값
    # -----------------------------
    d = 0.005        # m, 유리 두께 5 mm
    k = 0.937        # W/mK, 일반 유리 열전도율
    rho = 2530       # kg/m3, 일반 유리 밀도
    c = 880          # J/kgK, 일반 유리 비열

    # 표준 표면저항 기반 등가 표면열전달계수
    hi = 7.7         # W/m2K, 실내측 등가 표면열전달계수
    ho = 25.0        # W/m2K, 실외측 등가 표면열전달계수

    alpha = k / (rho * c)

    # -----------------------------
    # 수치해석 설정
    # -----------------------------
    N = 6                    # 유리 두께 방향 노드 수: T0~T5
    dx = d / (N - 1)          # 노드 간격
    dt = 1.0                 # 시간 간격 [s]
    seconds_per_hour = 3600

    # 초기조건
    # 유리 전체 온도를 현재 실내측 표면온도와 같다고 가정
    T = np.ones(N) * Ts0

    time_list = [0]
    Tsi_list = [T[0]]
    Tso_list = [T[-1]]

    # 현재 시점의 외기온도는 첫 번째 예보값으로 표시
    To_list = [outdoor_temps[0]]

    # 1시간 단위 외기온도를 적용하여 5시간 계산
    for hour_index, To in enumerate(outdoor_temps):
        for _ in range(seconds_per_hour):
            T_old = T.copy()
            T_new = T.copy()

            # 내부 노드 계산
            # 1차원 비정상 열전도 방정식의 유한차분 계산
            for j in range(1, N - 1):
                T_new[j] = T_old[j] + alpha * dt / dx**2 * (
                    T_old[j - 1] - 2 * T_old[j] + T_old[j + 1]
                )

            # 실내측 표면 노드 T0
            # 실내 공기와의 대류 + 유리 내부로의 전도
            T_new[0] = T_old[0] + dt / (rho * c * (dx / 2)) * (
                hi * (Ti - T_old[0]) +
                k * (T_old[1] - T_old[0]) / dx
            )

            # 실외측 표면 노드 T5
            # 유리 내부로부터의 전도 + 외기와의 대류
            T_new[-1] = T_old[-1] + dt / (rho * c * (dx / 2)) * (
                k * (T_old[-2] - T_old[-1]) / dx +
                ho * (To - T_old[-1])
            )

            T = T_new

        # 매 1시간마다 결과 저장
        time_list.append(hour_index + 1)
        Tsi_list.append(T[0])
        Tso_list.append(T[-1])
        To_list.append(To)

    result_df = pd.DataFrame({
        "시간 [h]": time_list,
        "외기온도 [℃]": To_list,
        "예측 실내측 유리 표면온도 [℃]": Tsi_list,
        "예측 실외측 유리 표면온도 [℃]": Tso_list,
    })

    return result_df


# -------------------------------------------------
# 5. 결로 판단 함수
# -------------------------------------------------

def judge_condensation(Tsi, Tdp):
    diff = Tsi - Tdp

    if diff <= 0:
        return "결로 위험"
    elif diff <= 1:
        return "주의"
    elif diff <= 2:
        return "관심"
    else:
        return "안전"


# -------------------------------------------------
# 6. session_state 초기화
# -------------------------------------------------

if "outdoor_temps" not in st.session_state:
    st.session_state.outdoor_temps = None

if "tmp_5h" not in st.session_state:
    st.session_state.tmp_5h = None

if "base_date" not in st.session_state:
    st.session_state.base_date = None

if "base_time" not in st.session_state:
    st.session_state.base_time = None

if "surface_df" not in st.session_state:
    st.session_state.surface_df = None

if "Tdp" not in st.session_state:
    st.session_state.Tdp = None




# -------------------------------------------------
# 7. 2행 x 2열 UI 배치
# -------------------------------------------------

top_left, top_right = st.columns(2)
bottom_left, bottom_right = st.columns(2)


# -------------------------------------------------
# 1. 실내 상태 입력
# -------------------------------------------------

with top_left:
    with st.container(border=True):
        st.subheader("1. 실내 상태 입력")

        Ti = st.number_input(
            "실내온도 (℃)",
            value=24.0,
            step=0.1
        )

        RH = st.number_input(
            "실내상대습도 (%)",
            value=60.0,
            step=1.0
        )

        Ts0 = st.number_input(
            "현재 창 실내측 표면온도 (℃)",
            value=18.0,
            step=0.1
        )

        if st.button("노점온도 계산"):
            Tdp = calculate_dew_point(Ti, RH)
            st.session_state.Tdp = Tdp

        if st.session_state.Tdp is not None:
            st.metric("현재 노점온도", f"{st.session_state.Tdp:.2f} ℃")

        st.caption("노점온도는 실내온도와 실내습도로 계산됩니다.")


# -------------------------------------------------
# 2. 기상청 예보 데이터
# -------------------------------------------------

with top_right:
    with st.container(border=True):
        st.subheader("2. 기상청 예보 데이터")

        st.write("지역: 인천 미추홀구")
        st.write(f"격자좌표: nx={nx}, ny={ny}")

        if st.button("기상청 예보 가져오기"):
            try:
                data, base_date, base_time = fetch_kma_forecast(nx, ny)

                result_code = data["response"]["header"]["resultCode"]
                result_msg = data["response"]["header"]["resultMsg"]

                weather = parse_weather(data)
                outdoor_temps, tmp_5h = extract_5h_outdoor_temps(weather)

                st.session_state.outdoor_temps = outdoor_temps
                st.session_state.tmp_5h = tmp_5h
                st.session_state.base_date = base_date
                st.session_state.base_time = base_time

                st.success("기상청 외기온도 5시간 자료를 가져왔습니다.")
                st.write(f"API 응답 코드: {result_code}")
                st.write(f"API 응답 메시지: {result_msg}")

            except Exception as e:
                st.error("기상청 데이터를 가져오는 중 오류가 발생했습니다.")
                st.write(e)

        if st.session_state.outdoor_temps is not None:
            st.write(f"조회 기준일: {st.session_state.base_date}")
            st.write(f"조회 기준시각: {st.session_state.base_time}")


            st.dataframe(
                pd.DataFrame(st.session_state.tmp_5h),
                use_container_width=True
            )

            st.caption("표면온도 계산은 마지막으로 불러온 기상청 예보값을 기준으로 수행됩니다.")
        else:
            st.info("기상청 예보를 먼저 불러오세요.")


# -------------------------------------------------
# 3. 표면온도 및 결로 위험 판단
# -------------------------------------------------

with bottom_left:
    with st.container(border=True):
        st.subheader("3. 표면온도 및 결로 위험 판단")

        if st.button("표면온도 및 결로 위험 계산"):
            if st.session_state.outdoor_temps is None:
                st.warning("먼저 '기상청 예보 가져오기' 버튼을 눌러 외기온도를 불러와야 합니다.")

            else:
                outdoor_temps = st.session_state.outdoor_temps

                surface_df = simulate_glass_temperature(Ti, Ts0, outdoor_temps)

                Tdp = calculate_dew_point(Ti, RH)
                st.session_state.Tdp = Tdp

                surface_df["노점온도 [℃]"] = Tdp
                surface_df["표면온도-노점온도 차이 [℃]"] = (
                    surface_df["예측 실내측 유리 표면온도 [℃]"] - Tdp
                )

                surface_df["결로 판단"] = surface_df["예측 실내측 유리 표면온도 [℃]"].apply(
                    lambda Tsi: judge_condensation(Tsi, Tdp)
                )

                st.session_state.surface_df = surface_df

        if st.session_state.surface_df is not None:
            surface_df = st.session_state.surface_df
            Tdp = st.session_state.Tdp

            worst_diff = surface_df["표면온도-노점온도 차이 [℃]"].min()

            if worst_diff <= 0:
                st.toast("🚨 결로 위험! 창 표면온도가 노점온도 이하로 예측됩니다.", icon="🚨")
                st.error("향후 5시간 내 결로 위험이 있습니다.")
            elif worst_diff <= 1:
                st.toast("⚠️ 결로 주의! 표면온도와 노점온도 차이가 1℃ 이하입니다.", icon="⚠️")
                st.warning("향후 5시간 내 결로 주의가 필요합니다.")
            elif worst_diff <= 2:
                st.info("향후 5시간 내 결로 관심 단계입니다.")
            else:
                st.success("향후 5시간 동안 결로 위험은 낮습니다.")

            st.write(f"실내 노점온도: {Tdp:.2f} ℃")

            st.dataframe(
                surface_df,
                use_container_width=True
            )
        else:
            st.info("표면온도 계산 버튼을 눌러 결과를 확인하세요.")


# -------------------------------------------------
# 4. 그래프 및 모델 설명
# -------------------------------------------------

with bottom_right:
    with st.container(border=True):
        st.subheader("4. 온도 변화 그래프")

        if st.session_state.surface_df is not None:
            surface_df = st.session_state.surface_df

            chart_df = pd.DataFrame({
                "time": surface_df["시간 [h]"],
                "outdoor_temp": surface_df["외기온도 [℃]"],
                "surface_temp": surface_df["예측 실내측 유리 표면온도 [℃]"],
                "dew_point": surface_df["노점온도 [℃]"]
            })

            chart_df["time"] = pd.to_numeric(chart_df["time"], errors="coerce")
            chart_df["outdoor_temp"] = pd.to_numeric(chart_df["outdoor_temp"], errors="coerce")
            chart_df["surface_temp"] = pd.to_numeric(chart_df["surface_temp"], errors="coerce")
            chart_df["dew_point"] = pd.to_numeric(chart_df["dew_point"], errors="coerce")

            chart_df = chart_df.set_index("time")

            st.line_chart(chart_df)

            st.caption(
                "outdoor_temp: 외기온도 / surface_temp: 예측 실내측 유리 표면온도 / dew_point: 실내 노점온도"
            )

        else:
            st.info("표면온도 계산 후 그래프가 표시됩니다.")

        st.divider()

        st.subheader("모델 가정")

        st.write(
            "단일창 유리를 두께 방향으로 나누고, "
            "1차원 비정상 열전도 방정식을 1초 간격으로 유한차분법으로 계산했습니다."
        )

        st.caption(
            "실내측 표면은 실내온도와 실내측 등가 열전달계수, "
            "실외측 표면은 기상청 외기온도와 실외측 등가 열전달계수를 이용한 "
            "대류 경계조건으로 처리했습니다."
        )

        st.warning(
            "본 계산은 단일창을 대상으로 한 간이 예측 모델입니다. "
            "유리 물성값과 실내·실외 표면열전달계수는 일반 조건을 가정한 값이며, "
            "실제 표면온도는 외부 풍속, 일사, 실내 기류, 커튼·블라인드, "
            "창틀 열교 등에 따라 달라질 수 있습니다."
        )