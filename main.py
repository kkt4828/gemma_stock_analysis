import torch
from transformers import pipeline
import streamlit as st
import pandas as pd

st.title('포트폴리오 분석 보고서 생성기')
st.header("1. Portfolio 정보")
list_of_investment = st.text_input(label="투자종목을 입력해주세요", placeholder="APPLE, Tesla, ...")
list_of_investment_ticker = st.text_input(label="투자종목을 입력해주세요 (ticker)")
number_of_shares = st.text_input(label="각 종목별 투자량을 입력해주세요")
purchase_price = st.text_input(label="매입한 자산 금액")
purchase_date = st.text_input(label="매입 날짜 정보 입력")
st.header("2. Portfolio 내 현금과 배당금 사용 정보")
cash_balance = st.text_input(label="cash balance")
dividend_reinvestment = st.selectbox(label="whether dividends are reinvested or paid out", options=['reinvested', 'paid out'])
st.header("3. 투자 목표와 성향")
investment_objectives = st.selectbox(label='투자 목적을 선택하세요', options=['Growth', 'Income', 'Capital preservation', 'Pension'])
risk_tolerance = st.selectbox(label='투자 성향을 선택하세요', options=['low-risk', 'medium-risk', 'high-risk'])
time_horizon = st.text_input(label='투자 기간을 입력해주세요', placeholder='short-term : 1 ~ 3 years / medium-term : 3 ~ 7 years / long-term ( 7+ years)')
st.header("4. 비교할 주요 Metrics")
performance_benchmark = st.text_input(label='비교할 지수 또는 종목을 선택하세요', placeholder='S&P 500, Nasdaq, ...')
preferred_metrics = st.text_input(label='어떤 메트릭에 집중할 지 입력해주세요', placeholder='total return, CAGR, P/E ratio, 영업이익, 매출, ...')
st.header("5. Portfolio 자산 비율")
desired_allocation = st.text_input(label='현재 포트폴리오에서 목표 자산 비율을 설정해주세요', placeholder='60% stocks, 30% bonds, 10% cash')
geographical_preferences = st.text_input(label='지역적 투자성향을 입력해주세요', placeholder='Domestic, international, ...')
sector_preferences = st.text_input(label='선호하는 섹터를 알려주세요', placeholder='Tech, healthcare, finance, AI, ...')
st.header("6. 선호하는 Report 형식")
report_frequency = st.text_input(label='보고서 주기를 선택해주세요', placeholder='Daily, weekly, monthly, quarterly reports, ...')
report_type = st.text_input(label='보고서 형태를 선택해주세요', placeholder='Summary vs detailed analysis')
st.header("7. 기타 추가 정보")
other_info = st.text_input(label="기타 정보를 입력해주세요", placeholder='transaction 정보, tax 정보, 최근이슈 등')
other_info_file = st.file_uploader(label='첨부파일을 업로드헤주세요')
last_prompt = f"""
I will give you many information about my stock analysis and portfolio.
So Can you tell me about my portfolio analysis and recommendation next actions?
the following information
1. Portfolio information
- List of investments : {list_of_investment}
- investments ticker symbol : {list_of_investment_ticker}
- number of shares : {number_of_shares}
- purchase price : {purchase_price}
- purchase date : {purchase_date}

2. Cash and Dividends
- Cash Balance : {cash_balance}
- Dividend Reinvestment : {dividend_reinvestment}

3. Investment Goals and preferences
- Investment Objectives : {investment_objectives}
- Risk Tolerance : {risk_tolerance}
- time horizon : {time_horizon}

4. Financial Metrics and Performance
- Performance Benchmark : {performance_benchmark}
- Preferred Financial Metrics : {preferred_metrics}

5. Asset Allocation Preferences
- Desired Allocation : {desired_allocation}
- Geographical Preferences : {geographical_preferences}
- Sector Preferences : {sector_preferences}

6. Reporting Preferences
- Report Frequency : {report_frequency}
- Report Type : {report_type}

7. Additional Data
- Other info : {other_info}
"""

messages = [
    {"role": "user", "content": f"{last_prompt}"},
]
button = st.button('Generate')
if button:
    pipe = pipeline(
        "text-generation",
        model="google/gemma-2-2b-it",
        model_kwargs={"torch_dtype": torch.bfloat16},
        device="cuda",  # replace with "mps" to run on a Mac device
    )
    outputs = pipe(messages, max_new_tokens=512)
    assistant_response = outputs[0]["generated_text"][-1]["content"].strip()

    st.write("---")
    st.header('분석 결과')
    st.write(assistant_response)