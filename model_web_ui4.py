import streamlit as st
import pandas as pd
import joblib



# 載入模型
# model_path = "/Users/zhengqunyao/loan_prediction_model_tuned.pkl"
# loaded_model = joblib.load(model_path)
# model_path = "loan_prediction_model_tuned.pkl"

# loaded_model = joblib.load(model_path)

import logging
logging.basicConfig(level=logging.INFO)

@st.cache_resource
def load_model():
    logging.info("載入模型中...")
    model_path = "loan_prediction_model_tuned.pkl"
    return joblib.load(model_path)

loaded_model = load_model()


# 定義應用的標題
st.title("mid-HPT e啟增貸 AI升級 增貸預測模型")

# 讓使用者輸入資料
st.header("請輸入以下資訊進行預測：")

#Education = st.number_input("Education (教育程度)", min_value=0, max_value=100, value=40, step=1)
#Employment = st.number_input("Employment (就業情況)", min_value=0, max_value=100, value=10, step=1)
#Marital = st.number_input("Marital (婚姻狀態)", min_value=0, max_value=100, value=10, step=1)
#CompanyRelationship = st.number_input("CompanyRelationship (與公司關係)", min_value=0, max_value=100, value=50, step=1)
#Industry = st.text_input("Industry (行業別)", value="I")
#Job = st.text_input("Job (職稱)",  value="0", step=1)
#Type = st.number_input("Type (類型)", min_value=0, max_value=100, value=20)
#ApprovalResult = st.text_input("ApprovalResult (審核結果)", value="A010")
#Years = st.number_input("Years (原房貸距今年份)", min_value=0, max_value=100, value=5, step=1)
#Age = st.number_input("Age (年齡)", min_value=0, max_value=120, value=38, step=1)

#Education = st.selectbox("Education (教育程度)", options=[10, 20, 30, 40, 50, 60], index=4)
#Employment = st.selectbox("Employment (就業情況)", options=[10, 20, 30, 40, 50, 60, 70], index=1)
#Marital = st.selectbox("Marital (婚姻狀態)", options=[10, 20, 30], index=1)
#CompanyRelationship = st.selectbox("CompanyRelationship (與公司關係)", options=[10, 20, 30, 40, 50, 60, 70], index=5)
#Industry = st.selectbox("Industry (行業別)", options=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"], index=8)
#Job = st.selectbox("Job (職稱)", options=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"], index=0)
#Type = st.selectbox("Type (類型)", options=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110], index=2)
#ApprovalResult = st.selectbox("ApprovalResult (審核結果)", options=["A010", "A020", "A030", "A040", "A050"], index=0)
#Years = st.number_input("Years (原房貸距今年份)", min_value=0, max_value=100, value=5, step=1)
#Age = st.number_input("Age (年齡)", min_value=0, max_value=120, value=38, step=1)
import pandas as pd


# 定義加權顯示函數
def weighted_display(prob):
    if prob < 0.20:
        # 將 0% ~ 20% 對應到 0% ~ 50%
        return prob * (50 / 0.20)
    else:
        # 將 20% ~ 100% 對應到 50% ~ 100%
        return 50 + (prob - 0.20) * (50 / 0.80)
    
    # 定義加權顯示函數
def label_display(prob):
    if prob < 0.20:
        return "非潛在增貸戶"
    else:
        return "潛在增貸戶"



education_options = {"10-博士": 10, "20-碩士": 20, "30-大學": 30, "40-專科": 40, "50-高中(職)": 50, "60-國中":60}
employment_options = {"10-上班族(含四師)": 10, "20-自由業": 20, "30-自營商": 30, "40-家管": 40, "50-學生": 50, "60-無業": 60, "70-其他": 70}
marital_options = {"10-已婚": 10, "20-未婚": 20, "30-其他": 30}
companyrelationship_options = {"10-約僱人員": 10, "20-正式人員": 20, "30-家族企業": 30, "40-股東": 40, "50-負責人": 50, "60-其他": 60, "70-房貸-STP": 70}
	
# 行業別對應表
industry_options = {
    "0-無/自由業/攤販": "0",
    "1-無、待業": "1",
    "2-家管": "2",
    "3-自由業(土木/水電包工)、SOHO(小型工作室)、靠行司機、保姆": "3",
    "4-攤販/檳榔攤": "4",
    "5-退休": "5",
    "10-保險業": "10",
    "21-人力派遣/仲介公司": "21",
    "22-地產管理人或經紀商(含仲介)": "22",
    "31-汽車買賣(車商)": "31",
    "32-二手車商": "32",
    "40-直銷": "40",
    "51-手錶、藝術品買賣業、古董店、珠寶商(含銀樓)": "51",
    "52-藝品店": "52",
    "53-貴金屬商、寶石商(如買賣未切割之原石)": "53",
    "54-當鋪、地下融資公司": "54",
    "55-地政士、代書": "55",
    "56-虛擬貨幣平台業/虛擬帳號服務": "56",
    "61-KTV/酒吧/夜總會/卡拉OK/三溫暖/俱樂部": "61",
    "62-博奕業(賭場、線上博奕)": "62",
    "63-軍火商": "63",
    "70-飯店/旅館": "70",
    "80-旅行社": "80",
    "90-保全業": "90",
    "A-農、林、漁、牧、狩獵、礦、土石採取業": "A",
    "B-水電煤氣業、加油站": "B",
    "C-運輸業": "C",
    "D-倉儲業": "D",
    "E-營造、土木、建築業": "E",
    "F-製造業": "F",
    "G-通訊業": "G",
    "H-電腦電子業": "H",
    "I-非高度專業服務業": "I",
    "I1-公益彩券/運動彩券": "I1",
    "J-自營": "J",
    "K-進出口貿易業": "K",
    "L-批發、零售業": "L",
    "M-餐飲業": "M",
    "N-大眾傳播/藝術工作/運動": "N",
    "O-金融業": "O",
    "P-證券及期貨業": "P",
    "Q-公營事業": "Q",
    "R-政府機關": "R",
    "S-教育機關": "S",
    "T-軍警消防業": "T",
    "U-非營利團體/宗教服務": "U",
    "U1-非營利團體、民意(立委)代表處": "U1",
    "U2-宗教服務(宮、廟)": "U2",
    "V-醫院": "V",
    "W-專業性服務(律師、會計師、建築師)": "W",
    "W1-專業性服務(律師、會計師事務所)": "W1",
    "W2-專業性服務(建築師事務所)": "W2",
    "X-學生": "X"
}

# 職稱對應表
job_options = {
    "0-負責人/董事長/總裁/理事長/大專校長、中研院院士": "0",
    "1-董事/股東/總監": "1",
    "2-總經理/總幹事/董監事/大專教授、副教授/院長/部長/次長": "2",
    "3-副總經理/協理/處長/大專助理教授、講師/局長/司長/首長/執行長/財務長": "3",
    "4-經理/副理/廠長/副廠長/(高中小)學校長/參事/專委/警正(監)": "4",
    "5-主任/科長/課長/襄理/隊長/巡佐": "5",
    "6-職員/科員/教師/專員/業務員/警員": "6",
    "61-職/科員/教師/國會助理/領班/專員/站長/組長": "61",
    "62-店長": "62",
    "63-業務員(非房仲)": "63",
    "64-房仲業務員": "64",
    "7-約聘人員/幼稚員老師/補習班老師": "7",
    "8-服務人員/工員(友)/生產線作業人員": "8",
    "81-生產線領班/組長": "81",
    "9-將軍": "9",
    "A-校級": "A",
    "B-尉級": "B",
    "C-士官、兵": "C",
    "D-中央民意代表": "D",
    "E-地方縣市長/地方民意代表": "E",
    "F-法官、檢察官、書記官": "F",
    "G-學生": "G",
    "H-會計師/律師/醫師/工程師/建築師": "H",
    "H1-律師": "H1",
    "H2-會計師": "H2",
    "H3-醫師(含中/牙/獸醫)": "H3",
    "H4-建築師": "H4",
    "I-專業技師": "I",
    "J-專業顧問/代書/地政士": "J",
    "J0-專業顧問/代書/地政士": "J0",
    "J1-專業顧問/代書/地政士": "J1",
    "J2-記帳士": "J2",
    "K-專業技術人員": "K",
    "L-專業技工(機電、土木、汽車修護)": "L",
    "M-廚師、汽車駕駛": "M",
    "N-保全人員、大樓管理員、清潔員、殯葬人員": "N",
    "N0-保全人員、大樓管理員、清潔員、殯葬人員": "N0",
    "N1-禮儀師(*)": "N1",
    "O-職業運動員/教練": "O",
    "P-宗教服務人員": "P",
    "Q-電視(電影)導演/製作": "Q",
    "R-音樂、戲劇、舞蹈表演、經紀人、模特兒": "R",
    "S-記者、播報、主持": "S",
    "T-服裝/造型設計人員、美容師(含SPA)、按摩師、推拿師": "T",
    "U-翻譯、寫作、攝影、圖畫": "U",
    "V-三鐵及海、空駕駛、空服員、船員": "V",
    "V0-三鐵及海、空駕駛、空服員、船員": "V0",
    "V1-飛機駕駛": "V1",
    "V2-三鐵駕駛、空服員": "V2",
    "V3-船員": "V3",
    "W-家庭主婦": "W",
    "X-退休人員": "X",
    "Y-自營商(XX企業社、小商行(號)之負責人)": "Y",
}

# 申貸性質對應表
loan_type_options = {
    "010-新買賣": 10,
    "020-增轉貸": 20,
    "030-平降轉": 30,
    "040-轉換約": 40,
    "050-變更條件": 50,
    "060-他行轉貸": 60,
    "070-舊戶增貸": 70,
    "080-原屋融資": 80,
    "090-他行平降轉": 90,
    "100-他行轉增貸": 100,
    "110-二順位貸款": 110
}

# 審核結果對應表
approval_result_options = {
    "A008-房貸-依符合規定條件核准": "A008",
    "A010-依原申貸條件承作": "A010",
    "A011-變更額度核准": "A011",
    "A012-變更額度及利率核准": "A012",
    "A013-變更額度及年限核准": "A013",
    "A014-變更利率核准": "A014",
    "A015-變更利率及年限核准": "A015",
    "A016-變更年限核准": "A016",
    "A017-變更年限、額度、利率核准": "A017",
    "A018-信貸-依規定金額核准": "A018",
    "A020-變更條件後承作": "A020",
    "A030-變更其他條件核准(請說明)": "A030",
    "A031-原額度及加信用科目核准": "A031",
    "A032-變更額度及加信用科目核准": "A032",
    "A051-房貸-依原申請條件核准": "A051",
    "A052-房貸-依符合規定條件核准": "A052",
    "A053-房貸-變更額度核准": "A053",
    "A054-房貸-變更年限核准": "A054",
    "A055-房貸-變更額度及年限核准": "A055",
    "A056-房貸-變更額度及利率核准": "A056",
    "A057-房貸-變更年限及利率核准": "A057",
    "A058-房貸-變更年限、額度、利率核准": "A058",
    "A059-房貸-變更利率核准": "A059",
    "A060-房貸-變更其他條件核准(請說明)": "A060",
    "A062-房貸-僅變更額度逾規定": "A062",
    "A063-房貸-僅變更額度符合規定": "A063",
    "A064-房貸-僅變更額度低於規定": "A064",
    "A065-房貸-僅變更年限逾規定": "A065",
    "A066-房貸-僅變更年限符合規定": "A066",
    "A067-房貸-僅變更年限低於規定": "A067",
    "A068-房貸-變更(額度及年限均逾規定)": "A068",
    "A069-房貸-變更(額度逾規定，年限符合規定)": "A069",
    "A070-房貸-變更(額度逾規定，年限低於規定)": "A070",
    "A071-房貸-變更(額度符合規定，年限逾規定)": "A071",
    "A072-房貸-變更(額度及年限均符合規定)": "A072",
    "A073-房貸-變更(額度符合規定，年限低於規定)": "A073",
    "A074-房貸-變更(額度低於規定，年限逾規定)": "A074",
    "A075-房貸-變更(額度低於規定，年限符合規定)": "A075",
    "A076-房貸-變更(額度及年限均低於規定)": "A076",
    "A077-房貸-STP": "A077",
    "A078-信貸-依專案額度核准": "A078",
    "B010-婉拒": "B010",
    "B011-房貸-婉拒": "B011",
    "B020-信貸-依評分建議婉拒": "B020",
    "B030-信貸-政策性婉拒": "B030",
    "B040-信貸-排除規則婉拒": "B040",
    "B050-信貸-婉拒": "B050",
    "B060-信貸-依評分(額度)建議婉拒": "B060",
    "B070-信貸-依專案額度核准": "B070"
}

Education = st.selectbox("Education (學歷)", options=list(education_options.keys()), index=2)
Employment = st.selectbox("Employment (就業情況)", options=list(employment_options.keys()), index=0)
Marital = st.selectbox("Marital (婚姻狀態)", options=list(marital_options.keys()), index=0)
CompanyRelationship = st.selectbox("CompanyRelationship (與公司關係)", options=list(companyrelationship_options.keys()), index=1)
Industry = st.selectbox("Industry (行業別)", options=list(industry_options.keys()), index=29)
Job = st.selectbox("Job (職稱)", options=list(job_options.keys()), index=6)
Type = st.selectbox("Type (申貸性質)", options=list(loan_type_options.keys()), index=0)
ApprovalResult = st.selectbox("ApprovalResult (審核結果)", options=list(approval_result_options.keys()), index=14)
Years = st.number_input("Years (原房貸距今年份)", min_value=0, max_value=15, value=9, step=1)
Age = st.number_input("Age (當時年齡)", min_value=0, max_value=80, value=39, step=1)

Income = st.number_input("Income (月收入)", min_value=0, max_value=1000000, value=68000, step=1000)
LoanIncomeRatio = st.number_input("LoanIncomeRatio (負債比)", min_value=0.0, max_value=10000.0, value=50.0, step=0.01, format="%.2f")
Adjust = st.number_input("Adjust (碼數)", min_value=0.0, max_value=10.0, value=0.68, step=0.01, format="%.2f")


# 當使用者按下預測按鈕時
if st.button("預測"):
    # 構建單筆資料
    single_data = {
        "Education": education_options[Education],
        "Employment": employment_options[Employment],
        "Marital": marital_options[Marital],
        "CompanyRelationship": companyrelationship_options[CompanyRelationship],
        "Industry": industry_options[Industry],
        "Job": job_options[Job],
        "Type": loan_type_options[Type],
        "ApprovalResult": approval_result_options[ApprovalResult],
        "Years": Years,
        "Age": Age,
        "Income": Income,
        "LoanIncomeRatio": LoanIncomeRatio,
        "Adjust": Adjust,
        
    }
    print(single_data)

    
    # 將單筆資料轉為 DataFrame
    single_data_df = pd.DataFrame([single_data])
    
    # 類別型特徵進行編碼（與訓練時一致）
    for col in single_data_df.select_dtypes(include=['object']).columns:
        single_data_df[col] = single_data_df[col].astype('category').cat.codes

    # 預測類別和概率
    predicted_class = loaded_model.predict(single_data_df)
    predicted_prob = loaded_model.predict_proba(single_data_df)[:, 1]

    label = label_display(predicted_prob)
    predicted_prob = weighted_display(predicted_prob)

    # 將類別轉換為更易理解的文字
    predicted_label = label

    # 顯示預測結果
    st.subheader("預測結果：")

    # 使用 HTML 標籤放大文字和改格式
    st.markdown(f"""
        <div style="font-size: 24px; font-weight: bold;">
            預測類別：<span style="color: blue;">{predicted_label}</span>
        </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
        <div style="font-size: 24px; font-weight: bold;">
            增貸概率：<span style="color: green;">{predicted_prob[0] :.0f}%</span>
        </div>
    """, unsafe_allow_html=True)
