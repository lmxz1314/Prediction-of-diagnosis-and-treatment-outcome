import pywt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
#X每行为一条数据，每列为一个特征

def msc(X):
    q,p = X.shape
    X_msc = np.ones((q,p))
    mean = np.mean(X,axis=0)
    for i in range(q):
        j = X[i,:]
        liner = LinearRegression()
        liner.fit(mean.reshape(-1, 1),j.reshape(-1, 1))
        k = liner.coef_
        b = liner.intercept_
        X_msc[i,:] = (j-b)/k
    return X_msc

def snv(X):#SNV,类似标准化Standardization,是对行进行z-score。
    X_sn = np.array(X,dtype=float).copy()
    m,n = X_sn.shape
    X_std = np.std(X_sn,axis=1)#axis=1对每行操作
    X_mean = np.mean(X_sn,axis=1)
    X_snvlist = [[((X_sn[i][j] - X_mean[i]) / X_std[i]) for j in range(n)] for i in range(m)]
    X_snv = np.array(X_snvlist)
    return X_snv

def mc(X):#meancenter
    X_mmmean = np.mean(X,axis=0)
    X_mmean = X-X_mmmean
    return X_mmean

def stand(X):#标准化Standardization,对列进行z-score
    scaler = preprocessing.StandardScaler(copy=True)
    X_stand = scaler.fit_transform(X)
    return X_stand

def norm(X):#对行（0,1）最大最小归一化
    min_max_scaler = preprocessing.MinMaxScaler(copy=True)
    #min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    # #设置数据压缩范围为[-1,1]，对应均值归一化。默认是[0,1]，对应最大最小归一化。
    X_mminmax = min_max_scaler.fit_transform(X.T)#MinMaxScaler默认对列操作
    X_minmax = X_mminmax.T
    return X_minmax

    
def norm1(X):#归一化Normalization,对行进行L1范式操作
    X_norm1 = preprocessing.normalize(X, norm='l1',copy=True) #L1范式和L2范式,默认L2
    return X_norm1

def norm2(X):#归一化Normalization,对行进行L2范式操作
    X_norm2 = preprocessing.normalize(X, norm='l2',copy=True) #L1范式和L2范式,默认L2
    return X_norm2

def MWA(data, window_size=3):
    """
    对近红外光谱数据应用移动窗口平均预处理。
    
    参数:
    - data: 2D numpy array, 每一行表示一个样本的光谱数据
    - window_size: int, 移动窗口的大小，默认为 3
    
    返回:
    - 经过移动窗口平均预处理后的数据
    """
    # 使用 pandas 提供的 rolling 方法对每一行进行移动窗口平均
    data_smoothed = pd.DataFrame(data).T.rolling(window=window_size, min_periods=1, center=True).mean().T.values
    return data_smoothed

def WT(data_x, wavelet='db8', threshold=0.04):
    """
    对NIR数据进行小波变换的预处理。

    参数:
    - data_x: 输入的numpy数组，每行表示一条光谱数据
    - wavelet: 小波类型，默认为'db8'
    - threshold: 去噪处理的阈值，默认为0.04

    返回:
    - 经过小波变换和去噪处理的NIR数据
    """

    def process_signal(signal, wavelet, threshold):
        """
        对单个信号进行小波变换和去噪处理。
        """
        w = pywt.Wavelet(wavelet)
        maxlev = pywt.dwt_max_level(len(signal), w.dec_len)
        coeffs = pywt.wavedec(signal, wavelet, level=maxlev)
        for i in range(1, len(coeffs)):
            coeffs[i] = pywt.threshold(coeffs[i], threshold * np.max(coeffs[i]))
        processed_signal = pywt.waverec(coeffs, wavelet)
        return processed_signal

    # 预分配数组以提高性能
    processed_data = np.zeros_like(data_x)

    for i in range(data_x.shape[0]):
        processed_data[i, :] = process_signal(data_x[i], wavelet, threshold)

    return processed_data




    

    
