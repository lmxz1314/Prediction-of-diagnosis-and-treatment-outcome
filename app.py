import streamlit as st
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from scipy import signal
import hmpreprocessing2#导入自制的预处理库
import joblib  # 用于加载训练好的模型

# 页面设置
st.set_page_config(page_title="NIR-based MDD Diagnosis and treatment outcome prediction",layout="wide")
# 设置页面标题
st.title("NIR-Based Diagnostic and Antidepressant Treatment Response Prediction System for MDD")

# 提示用户上传红外光谱数据
st.sidebar.header("Upload the spectral file")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

col1, col2 = st.columns([1, 1])

def Pre(X, smooth_method, transform_method):
    if smooth_method == "No":
        X_smooth = X
    elif smooth_method == "SG":
        X_smooth = signal.savgol_filter(X, 13, 3, deriv=1, axis=1)
    elif smooth_method == "WT":
        X_smooth = hmpreprocessing2.WT(X,wavelet='sym8', threshold=0.005)  
    elif smooth_method == "MWA":
        data = pd.DataFrame(X).apply(pd.to_numeric, errors='coerce')
        X_smooth = hmpreprocessing2.MWA(data,window_size=23)
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
        return preprocessing.normalize(X_smooth, norm='l1', copy=True)
    elif transform_method == "Norm2":
        return preprocessing.normalize(X_smooth, norm='l2', copy=True)
    elif transform_method == "diff1":
        return np.diff(X_smooth)
    elif transform_method == "diff2":
        return np.diff(X_smooth, n=2)
    else:
        raise ValueError("Invalid transform method")
        
# 加载训练好的模型（替换为你的模型路径）
model_d = joblib.load("Final diagnosic model.joblib")  # 替换为你的模型文件路径
model_t = joblib.load("Final treatment outcome prediction model.joblib") 
model_dm = joblib.load("Final diagnosic model mapper.joblib")  # 替换为你的模型文件路径
model_tm = joblib.load("Final treatment outcome prediction model mapper.joblib") 



# 如果用户上传了文件
if uploaded_file:
    # 读取上传的文件
    input_data = pd.read_csv(uploaded_file,index_col=0,header=0)
    with col1:
        st.write("Near-infrared Spectral Data：")
        st.dataframe(input_data, use_container_width=True, height=400)

    # 确保数据维度与模型的输入一致
    if input_data.shape[1] != 1582:
        st.error(f"数据维度与模型输入不一致。模型需要 1582 个特征，但上传的数据有 {input_data.shape[1]} 个特征。")
    else:
        data = pd.read_csv("MDD&HC_average.csv",header=0)
        # 提取横纵坐标
        x = data.iloc[:, 0]  # 波数（cm⁻¹）
        y = np.array(data.iloc[:, 1])  # 平均光谱值
        input_draw=input_data.values.T
        input_drawminmax=preprocessing.MinMaxScaler().fit_transform(input_draw)
        # 绘图
        y_minmax = preprocessing.MinMaxScaler().fit_transform(y.reshape(-1, 1)).flatten()
        fig, ax = plt.subplots(figsize=(5, 4)) 
        ax.plot(x, y_minmax, label="参考光谱")
        ax.plot(x, input_drawminmax)
        ax.set_xlabel("Wavenumber (nm)")
        ax.set_ylabel("Intensity")
        #ax.invert_xaxis()  # 红外图谱通常从高波数向低波数
        #ax.legend()
        #ax.grid(True)
        # 在 Streamlit 中显示图像
        with col2:
            st.write("Near-infrared Spectra：")
            st.pyplot(fig)
        # 进行预测
        if st.button("Predict"):
            X_trs_d=Pre(input_data,"SG","No")
            selected_features_d=[339,238,338,109,88,260,34,8,345,239,158,81,344,135,261,280,110,119,148,234,370,98,80,120,628,692,134,491,894,716,65,216,526,721,421,703,510,959,452,1205,1108,1082,1414,430,756,511,492,771,1039,1305,976,1329,1204,1328,509,1429,778,1100,456,1327,503,1558]
            #predictions = model.predict(X_trs)  # 预测类别
            proba_d = model_d.predict(X_trs_d[:,selected_features_d])  # 预测概率
            predictions_d= (proba_d > 0.5268778234383523).astype(int)
            probabilities_d = model_dm.predict_proba(proba_d.flatten().reshape(-1, 1))[:, 1]

            X_trs_t=Pre(input_data,"MWA","No")
            selected_features_t=[36,14,258,121,89,324,140,91,670,784,575,627,629,630,623,846,848,634,537,1150,1164,1186,1428,1429,1260,1460,1463,1293,963,964,961]
            #predictions = model.predict(X_trs)  # 预测类别
            proba_t = model_t.predict(X_trs_t[:,selected_features_t])  # 预测概率
            predictions_t= (proba_t > 0.49122806236829564).astype(int)
            probabilities_t = model_tm.predict_proba(proba_t.flatten().reshape(-1, 1))[:, 1]
              
            # 显示预测结果
            st.write("Results：")
            results = pd.DataFrame({
                "Sample ID": range(1, len(predictions_d) + 1),
                "Predicted probability of MDD": abs(probabilities_d).round(2),
                "Predicted diagnostic results": predictions_d                
            })
            # 初始化空列
            treatment_results = []
            cure_probabilities = []
            
            # 根据 predictions_d 添加治疗信息
            for i in range(len(predictions_d)):
                if predictions_d[i] == 1:
                    treatment_results.append(predictions_t[i])
                    cure_probabilities.append(abs(probabilities_t[i]).round(2))
                else:
                    treatment_results.append("/")
                    cure_probabilities.append("/")
            
            # 添加到 DataFrame
            results["Predicted probability of treatment response"] = cure_probabilities
            results["Treatment outcome prediction results"] = treatment_results            
            #st.write(results)
            st.dataframe(results, use_container_width=True)

            # 根据预测类别统计健康和患病人数
            healthy_count = (predictions_d == 0).sum()
            diseased_count = (predictions_d == 1).sum()
            remitter_count = (results["Treatment outcome prediction results"] == 0).sum()
            
            st.write(f"Number of healthy samples：{healthy_count}")
            st.write(f"Number of MDD samples：{diseased_count}")
            st.write(f"Number of MDD samples with positive response to antidepressant treatment：{remitter_count}")
else:
    st.write("Upload NIR spectral data file on the left.")

