import streamlit as st
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from scipy import signal
import hmpreprocessing2  # 导入自制的预处理库
import joblib  # 用于加载训练好的模型

plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "axes.labelsize": 16,
        "axes.titlesize": 17,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 13,
    }
)


def render_table(df: pd.DataFrame, show_index: bool = True) -> None:
    html = df.to_html(index=show_index, classes="custom-table", border=0)
    st.markdown(f'<div class="table-wrapper">{html}</div>', unsafe_allow_html=True)


# 页面设置
st.set_page_config(
    page_title="NIR-based MDD Diagnosis and treatment outcome prediction", layout="wide"
)
# 设置页面标题
st.title("NIR-Based Diagnostic and Treatment Outcome Prediction System for MDD")

# 提示用户上传红外光谱数据
st.sidebar.header("Upload the spectral file")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

col1, col2 = st.columns([1.1, 1], gap="large")


def Pre(X, smooth_method, transform_method):
    if smooth_method == "No":
        X_smooth = X
    elif smooth_method == "SG":
        X_smooth = signal.savgol_filter(X, 13, 3, deriv=1, axis=1)
    elif smooth_method == "WT":
        X_smooth = hmpreprocessing2.WT(X, wavelet="sym8", threshold=0.005)
    elif smooth_method == "MWA":
        data = pd.DataFrame(X).apply(pd.to_numeric, errors="coerce")
        X_smooth = hmpreprocessing2.MWA(data, window_size=23)
    else:
        raise ValueError("Invalid smooth method")

    if transform_method == "No":
        return X_smooth
    elif transform_method == "MC":
        return hmpreprocessing2.mc(X_smooth)
    elif transform_method == "MSC":
        return hmpreprocessing2.msc(X_smooth)
    elif transform_method == "SNV":
        return hmpreprocessing2.snv(X_smooth)
    elif transform_method == "Stand":
        return hmpreprocessing2.stand(X_smooth)
    elif transform_method == "Norm":
        return hmpreprocessing2.norm(X_smooth)
    elif transform_method == "Norm1":
        return preprocessing.normalize(X_smooth, norm="l1", copy=True)
    elif transform_method == "Norm2":
        return preprocessing.normalize(X_smooth, norm="l2", copy=True)
    elif transform_method == "diff1":
        return np.diff(X_smooth)
    elif transform_method == "diff2":
        return np.diff(X_smooth, n=2)
    else:
        raise ValueError("Invalid transform method")


# 加载训练好的模型（替换为你的模型路径）
model_d = joblib.load("Final diagnosic model.joblib")
model_t = joblib.load("Final treatment outcome prediction model.joblib")
model_dm = joblib.load("Final diagnosic model mapper.joblib")
model_tm = joblib.load("Final treatment outcome prediction model mapper.joblib")


if uploaded_file:
    input_data = pd.read_csv(uploaded_file, index_col=0, header=0)
    with col1:
        st.markdown('<p class="section-title">Near-infrared Spectral Data：</p>', unsafe_allow_html=True)
        render_table(input_data, show_index=True)

    if input_data.shape[1] != 1582:
        st.error(
            f"数据维度与模型输入不一致。模型需要 1582 个特征，但上传的数据有 {input_data.shape[1]} 个特征。"
        )
    else:
        data = pd.read_csv("MDD&HC_average.csv", header=0)
        x = data.iloc[:, 0]
        y = np.array(data.iloc[:, 1])
        input_draw = input_data.values.T
        input_drawminmax = preprocessing.MinMaxScaler().fit_transform(input_draw)

        y_minmax = preprocessing.MinMaxScaler().fit_transform(y.reshape(-1, 1)).flatten()
        fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=240)
        ax.plot(x, y_minmax, label="参考光谱")
        ax.plot(x, input_drawminmax)
        ax.set_xlabel("Wavenumber (nm)", fontsize=16, fontname="Times New Roman")
        ax.set_ylabel("Intensity", fontsize=16, fontname="Times New Roman")
        ax.tick_params(axis="both", labelsize=14)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontname("Times New Roman")

        with col2:
            st.markdown('<p class="section-title">Near-infrared Spectra：</p>', unsafe_allow_html=True)
            st.pyplot(fig, use_container_width=True)

        if st.button("Predict"):
            X_trs_d = Pre(input_data, "SG", "No")
            selected_features_d = [
                339,238,338,109,88,260,34,8,345,239,158,81,344,135,261,280,110,119,148,234,370,98,80,120,628,692,134,491,894,716,65,216,526,721,421,703,510,959,452,1205,1108,1082,1414,430,756,511,492,771,1039,1305,976,1329,1204,1328,509,1429,778,1100,456,1327,503,1558
            ]
            proba_d = model_d.predict(X_trs_d[:, selected_features_d])
            predictions_d = (proba_d > 0.5268778234383523).astype(int)
            probabilities_d = model_dm.predict_proba(proba_d.flatten().reshape(-1, 1))[:, 1]

            X_trs_t = Pre(input_data, "MWA", "No")
            selected_features_t = [
                36,14,258,121,89,324,140,91,670,784,575,627,629,630,623,846,848,634,537,1150,1164,1186,1428,1429,1260,1460,1463,1293,963,964,961
            ]
            proba_t = model_t.predict(X_trs_t[:, selected_features_t])
            predictions_t = (proba_t > 0.49122806236829564).astype(int)
            probabilities_t = model_tm.predict_proba(proba_t.flatten().reshape(-1, 1))[:, 1]

            st.markdown('<p class="section-title">Results：</p>', unsafe_allow_html=True)
            results = pd.DataFrame(
                {
                    "Sample ID": range(1, len(predictions_d) + 1),
                    "Probability of MDD": abs(probabilities_d).round(2),
                    "Diagnostic results (0:HC,1:MDD)": predictions_d,
                }
            )

            treatment_results = []
            cure_probabilities = []
            for i in range(len(predictions_d)):
                if predictions_d[i] == 1:
                    treatment_results.append(predictions_t[i])
                    cure_probabilities.append(abs(probabilities_t[i]).round(2))
                else:
                    treatment_results.append("/")
                    cure_probabilities.append("/")

            results["Probability of non-remission"] = cure_probabilities
            results["Treatment results (0:remission,1:non-remission)"] = treatment_results
            render_table(results, show_index=False)

            healthy_count = (predictions_d == 0).sum()
            diseased_count = (predictions_d == 1).sum()
            remitter_count = (results["Treatment results (0:remission,1:non-remission)"] == 0).sum()
            non_remitter_count = (results["Treatment results (0:remission,1:non-remission)"] == 1).sum()

            st.markdown(f'<p class="result-text">Number of healthy samples：{healthy_count}</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="result-text">Number of MDD samples：{diseased_count}</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="result-text">Number of remitters in MDD samples：{remitter_count}</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="result-text">Number of non-remitters in MDD samples：{non_remitter_count}</p>', unsafe_allow_html=True)
else:
    st.markdown('<p class="section-title">Upload NIR spectral data file on the left.</p>', unsafe_allow_html=True)


st.markdown(
    """
    <style>
    html,
    body,
    [class*="css"],
    [data-testid="stAppViewContainer"],
    [data-testid="stSidebar"] {
        font-family: "Times New Roman", serif;
        font-size: 19px;
    }

    p,
    label,
    .stMarkdown,
    .stButton button {
        font-family: "Times New Roman", serif !important;
        font-size: 22px !important;
        font-weight: 700 !important;
        padding: 0.4rem 1.1rem !important;
        border-radius: 8px !important;
    }

    h1, h2, h3 {
        font-family: "Times New Roman", serif !important;
    }

    h1 {
        font-size: 38px !important;
    }

    [data-testid="stSidebar"] * {
        font-family: "Times New Roman", serif !important;
        font-size: 18px !important;
    }

    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] div {
        margin-left: 0 !important;
        padding-left: 0 !important;
        text-indent: 0 !important;
    }

    .section-title {
        font-family: "Times New Roman", serif !important;
        font-size: 24px !important;
        font-weight: 600;
        margin: 0 0 8px 0 !important;
        padding: 0 !important;
        text-indent: 0 !important;
    }

    .result-text {
        font-family: "Times New Roman", serif !important;
        font-size: 22px !important;
        margin: 2px 0;
    }

    .table-wrapper {
        width: 100%;
        overflow-x: auto;
        margin-bottom: 14px;
    }

    .custom-table {
        width: 100%;
        border-collapse: collapse;
        font-family: "Times New Roman", serif !important;
        font-size: 20px !important;
    }

    .custom-table th,
    .custom-table td {
        border: 1px solid #d9d9d9;
        padding: 6px 10px;
        text-align: left;
        font-family: "Times New Roman", serif !important;
        font-size: 20px !important;
    }

    .custom-table thead th {
        background-color: #f7f7f7;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)





