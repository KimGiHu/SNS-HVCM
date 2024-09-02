import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import roc_curve, auc, f1_score, precision_score, recall_score, confusion_matrix
import random
import seaborn as sns
import argparse
from proposed_model import *
from baseline_model import *

print(f'GPU Available : {torch.cuda.is_available()}')
# cuda 캐시 정리
torch.cuda.empty_cache()

# 시드 값 설정
seed = 42

# 기본 시드 고정
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# CUDA 사용 시 추가 설정
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 멀티-GPU 사용 시
    # CuDNN 결정론적 및 비결정론적 동작 설정
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 데이터 준비
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# 손실 함수 정의 - 라이브러리
def loss_function(x_recon, x, mu, logvar):    
    MSE_lib = nn.functional.mse_loss(x_recon, x, reduction='sum') # divided into batch size
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())# see Appendix B from VAE paper:
                                                                 # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    return MSE_lib + 1.0*KLD + 1e-12

# 고유 파형들 정의
features=['A+IGBT-I', 'A+*IGBT-I', 'B+IGBT-I', 'B+*IGBT-I', 'C+IGBT-I', 
          'C+*IGBT-I', 'A-FLUX', 'B-FLUX', 'C-FLUX', 'CB-I','CB-V',
          'MOD-V', 'MOD-I','DV/DT']

# 모듈별 변수들 정의
system1='RFQ'
system2='DTL'
system3='CCL'
system4='SCL'

# 데이터 로드 및 라벨 인코딩
def load_data(system):
    X = np.load(f'../hvcm/data/hvcm/{system}.npy')
    Y = np.load(f'../hvcm/data/hvcm/{system}_labels.npy', allow_pickle=True)
    return X, Y

X1, Y1 = load_data(system1)
X2, Y2 = load_data(system2)
X3, Y3 = load_data(system3)
X4, Y4 = load_data(system4)

time = np.arange(X1.shape[1]) * 400e-9 # 타임스텝 : 1.8ms (4500개 샘플씩, 한 샘플당 400ns)

# 시스템 인덱스 생성
system_indices = {
    system1: 0,
    system2: 1,
    system3: 2,
    system4: 3
}

def create_labels(Y, system):
    labels = np.array([system_indices[system]] * len(Y))
    return labels

labels1 = create_labels(Y1, system1)
labels2 = create_labels(Y2, system2)
labels3 = create_labels(Y3, system3)
labels4 = create_labels(Y4, system4)

# 배열 X,Y의 정상 및 오류 데이터들을 분리함
fault_indices_RFQ, normal_indices_RFQ = np.where(Y1[:,1] == 'Fault')[0], np.where(Y1[:,1] == 'Run')[0] 
fault_indices_DTL, normal_indices_DTL = np.where(Y2[:,1] == 'Fault')[0], np.where(Y2[:,1] == 'Run')[0]
fault_indices_CCL, normal_indices_CCL = np.where(Y3[:,1] == 'Fault')[0], np.where(Y3[:,1] == 'Run')[0]
fault_indices_SCL, normal_indices_SCL = np.where(Y4[:,1] == 'Fault')[0], np.where(Y4[:,1] == 'Run')[0]

Xnormal_RFQ, Xanomaly_RFQ = X1[normal_indices_RFQ,:,:], X1[fault_indices_RFQ,:,:]
Xnormal_DTL, Xanomaly_DTL = X2[normal_indices_DTL,:,:], X2[fault_indices_DTL,:,:]
Xnormal_CCL, Xanomaly_CCL = X3[normal_indices_CCL,:,:], X3[fault_indices_CCL,:,:]
Xnormal_SCL, Xanomaly_SCL = X4[normal_indices_SCL,:,:], X4[fault_indices_SCL,:,:]

Ynormal_RFQ, Yanomaly_RFQ = Y1[normal_indices_RFQ,:], Y1[fault_indices_RFQ,:]
Ynormal_DTL, Yanomaly_DTL = Y2[normal_indices_DTL,:], Y2[fault_indices_DTL,:]
Ynormal_CCL, Yanomaly_CCL = Y3[normal_indices_CCL,:], Y3[fault_indices_CCL,:]
Ynormal_SCL, Yanomaly_SCL = Y4[normal_indices_SCL,:], Y4[fault_indices_SCL,:]

############################### 15가지 모듈의 정상신호 모음 ###############################
def return_indicies(data, module_name):
    indices = np.where(np.char.find(data, module_name) != -1)[0]
    return indices

def return_len(data, module_name):
    data_len = len(np.where(np.char.find(data, module_name) != -1)[0])
    return data_len

# 라벨링된 데이터 불러오기(?)
column_to_search_DTL_normal = Ynormal_DTL[:, 0].astype(str)
column_to_search_CCL_normal = Ynormal_CCL[:, 0].astype(str)
column_to_search_SCL_normal = Ynormal_SCL[:, 0].astype(str)

# RFQ 정상신호 
normal_RFQ = Xnormal_RFQ[:,:,:]

# DTL 정상신호 - DTL03, DTL05
normal_DTL3 = Xnormal_DTL[return_indicies(column_to_search_DTL_normal, 'DTL3'),:,:]
normal_DTL5 = Xnormal_DTL[return_indicies(column_to_search_DTL_normal, 'DTL5'),:,:]

# CCL 정상신호 - CCL01, CCL02, CCL03, CCL04
normal_CCL1 = Xnormal_CCL[return_indicies(column_to_search_CCL_normal, 'CCL1'),:,:]
normal_CCL2 = Xnormal_CCL[return_indicies(column_to_search_CCL_normal, 'CCL2'),:,:]
normal_CCL3 = Xnormal_CCL[return_indicies(column_to_search_CCL_normal, 'CCL3'),:,:]
normal_CCL4 = Xnormal_CCL[return_indicies(column_to_search_CCL_normal, 'CCL4'),:,:]

# SCL 정상신호 - SCL01, SCL05, SCL09, SCL12, SCL14, SCL15, SCL18, SCL21
normal_SCL1 = Xnormal_SCL[return_indicies(column_to_search_SCL_normal, 'SCL1'),:,:]
normal_SCL5 = Xnormal_SCL[return_indicies(column_to_search_SCL_normal, 'SCL5'),:,:]
normal_SCL9 = Xnormal_SCL[return_indicies(column_to_search_SCL_normal, 'SCL9'),:,:]
normal_SCL12 = Xnormal_SCL[return_indicies(column_to_search_SCL_normal, 'SCL12'),:,:]
normal_SCL14 = Xnormal_SCL[return_indicies(column_to_search_SCL_normal, 'SCL14'),:,:]
normal_SCL15 = Xnormal_SCL[return_indicies(column_to_search_SCL_normal, 'SCL15'),:,:]
normal_SCL18 = Xnormal_SCL[return_indicies(column_to_search_SCL_normal, 'SCL18'),:,:]
normal_SCL21 = Xnormal_SCL[return_indicies(column_to_search_SCL_normal, 'SCL21'),:,:]

############################## 15가지 모듈의 이상신호 모음 ###############################
column_to_search_RFQ = Yanomaly_RFQ[:, 0].astype(str)
column_to_search_DTL = Yanomaly_DTL[:, 0].astype(str)
column_to_search_CCL = Yanomaly_CCL[:, 0].astype(str)
column_to_search_SCL = Yanomaly_SCL[:, 0].astype(str)

# RFQ 이상신호
abnormal_RFQ = Xanomaly_RFQ[:,:,:]

# DTL 이상신호 - DTL03, DTL05
abnormal_DTL3 = Xanomaly_DTL[return_indicies(column_to_search_DTL, 'DTL3'),:,:]
abnormal_DTL5 = Xanomaly_DTL[return_indicies(column_to_search_DTL, 'DTL5'),:,:]

# CCL 이상신호 - CCL01, CCL02, CCL03, CCL04
abnormal_CCL1 = Xanomaly_CCL[return_indicies(column_to_search_CCL, 'CCL1'),:,:]
abnormal_CCL2 = Xanomaly_CCL[return_indicies(column_to_search_CCL, 'CCL2'),:,:]
abnormal_CCL3 = Xanomaly_CCL[return_indicies(column_to_search_CCL, 'CCL3'),:,:]
abnormal_CCL4 = Xanomaly_CCL[return_indicies(column_to_search_CCL, 'CCL4'),:,:]

# SCL 이상신호 - SCL01, SCL05, SCL09, SCL12, SCL14, SCL15, SCL18, SCL21
abnormal_SCL1 = Xanomaly_SCL[return_indicies(column_to_search_SCL, 'SCL1'),:,:]
abnormal_SCL5 = Xanomaly_SCL[return_indicies(column_to_search_SCL, 'SCL5'),:,:]
abnormal_SCL9 = Xanomaly_SCL[return_indicies(column_to_search_SCL, 'SCL9'),:,:]
abnormal_SCL12 = Xanomaly_SCL[return_indicies(column_to_search_SCL, 'SCL12'),:,:]
abnormal_SCL14 = Xanomaly_SCL[return_indicies(column_to_search_SCL, 'SCL14'),:,:]
abnormal_SCL15 = Xanomaly_SCL[return_indicies(column_to_search_SCL, 'SCL15'),:,:]
abnormal_SCL18 = Xanomaly_SCL[return_indicies(column_to_search_SCL, 'SCL18'),:,:]
abnormal_SCL21 = Xanomaly_SCL[return_indicies(column_to_search_SCL, 'SCL21'),:,:]

# 테스트 : normal / abnormal 설정
ratio_target = 0.85
ratio_test = 1.0
data_test = np.concatenate( 
    (normal_RFQ[int(len(normal_RFQ)*ratio_target):int(len(normal_RFQ)*ratio_test),:,:],
     # DTL module
     normal_DTL3[int(return_len(column_to_search_DTL_normal,'DTL3')*ratio_target):int(return_len(column_to_search_DTL_normal,'DTL3')*ratio_test),:,:],
     normal_DTL5[int(return_len(column_to_search_DTL_normal,'DTL5')*ratio_target):int(return_len(column_to_search_DTL_normal,'DTL5')*ratio_test),:,:],
     # CCL module
     normal_CCL1[int(return_len(column_to_search_CCL_normal,'CCL1')*ratio_target):int(return_len(column_to_search_CCL_normal,'CCL1')*ratio_test),:,:],
     normal_CCL2[int(return_len(column_to_search_CCL_normal,'CCL2')*ratio_target):int(return_len(column_to_search_CCL_normal,'CCL2')*ratio_test),:,:],
     normal_CCL3[int(return_len(column_to_search_CCL_normal,'CCL3')*ratio_target):int(return_len(column_to_search_CCL_normal,'CCL3')*ratio_test),:,:],
     normal_CCL4[int(return_len(column_to_search_CCL_normal,'CCL4')*ratio_target):int(return_len(column_to_search_CCL_normal,'CCL4')*ratio_test),:,:],
     # SCL module
     normal_SCL1[int(return_len(column_to_search_SCL_normal,'SCL1')*ratio_target):int(return_len(column_to_search_SCL_normal,'SCL1')*ratio_test),:,:],
     normal_SCL5[int(return_len(column_to_search_SCL_normal,'SCL5')*ratio_target):int(return_len(column_to_search_SCL_normal,'SCL5')*ratio_test),:,:],
     normal_SCL9[int(return_len(column_to_search_SCL_normal,'SCL9')*ratio_target):int(return_len(column_to_search_SCL_normal,'SCL9')*ratio_test),:,:],
     normal_SCL12[int(return_len(column_to_search_SCL_normal,'SCL12')*ratio_target):int(return_len(column_to_search_SCL_normal,'SCL12')*ratio_test),:,:],
     normal_SCL14[int(return_len(column_to_search_SCL_normal,'SCL14')*ratio_target):int(return_len(column_to_search_SCL_normal,'SCL14')*ratio_test),:,:],
     normal_SCL15[int(return_len(column_to_search_SCL_normal,'SCL15')*ratio_target):int(return_len(column_to_search_SCL_normal,'SCL15')*ratio_test),:,:],
     normal_SCL18[int(return_len(column_to_search_SCL_normal,'SCL18')*ratio_target):int(return_len(column_to_search_SCL_normal,'SCL18')*ratio_test),:,:],
     normal_SCL21[int(return_len(column_to_search_SCL_normal,'SCL21')*ratio_target):int(return_len(column_to_search_SCL_normal,'SCL21')*ratio_test)+1,:,:]), axis=0 )

fault_ratio = 0.0
fault_end_ratio = 1.0
data_anomaly = np.concatenate( 
    (abnormal_RFQ[int(len(abnormal_RFQ)*fault_ratio):int(len(abnormal_RFQ)*fault_end_ratio),:,:],
     # DTL module
     abnormal_DTL3[int(return_len(column_to_search_DTL,'DTL3')*fault_ratio):int(return_len(column_to_search_DTL,'DTL3')*fault_end_ratio),:,:],
     abnormal_DTL5[int(return_len(column_to_search_DTL,'DTL5')*fault_ratio):int(return_len(column_to_search_DTL,'DTL5')*fault_end_ratio),:,:],
     # CCL module
     abnormal_CCL1[int(return_len(column_to_search_CCL,'CCL1')*fault_ratio):int(return_len(column_to_search_CCL,'CCL1')*fault_end_ratio),:,:],
     abnormal_CCL2[int(return_len(column_to_search_CCL,'CCL2')*fault_ratio):int(return_len(column_to_search_CCL,'CCL2')*fault_end_ratio),:,:],
     abnormal_CCL3[int(return_len(column_to_search_CCL,'CCL3')*fault_ratio):int(return_len(column_to_search_CCL,'CCL3')*fault_end_ratio),:,:],
     abnormal_CCL4[int(return_len(column_to_search_CCL,'CCL4')*fault_ratio):int(return_len(column_to_search_CCL,'CCL4')*fault_end_ratio),:,:],
     # SCL module
     abnormal_SCL1[int(return_len(column_to_search_SCL,'SCL1')*fault_ratio):int(return_len(column_to_search_SCL,'SCL1')*fault_end_ratio),:,:],
     abnormal_SCL5[int(return_len(column_to_search_SCL,'SCL5')*fault_ratio):int(return_len(column_to_search_SCL,'SCL5')*fault_end_ratio),:,:],
     abnormal_SCL9[int(return_len(column_to_search_SCL,'SCL9')*fault_ratio):int(return_len(column_to_search_SCL,'SCL9')*fault_end_ratio),:,:],
     abnormal_SCL12[int(return_len(column_to_search_SCL,'SCL12')*fault_ratio):int(return_len(column_to_search_SCL,'SCL12')*fault_end_ratio),:,:],
     abnormal_SCL14[int(return_len(column_to_search_SCL,'SCL14')*fault_ratio):int(return_len(column_to_search_SCL,'SCL14')*fault_end_ratio),:,:],
     abnormal_SCL15[int(return_len(column_to_search_SCL,'SCL15')*fault_ratio):int(return_len(column_to_search_SCL,'SCL15')*fault_end_ratio),:,:],
     abnormal_SCL18[int(return_len(column_to_search_SCL,'SCL18')*fault_ratio):int(return_len(column_to_search_SCL,'SCL18')*fault_end_ratio),:,:],
     abnormal_SCL21[int(return_len(column_to_search_SCL,'SCL21')*fault_ratio):int(return_len(column_to_search_SCL,'SCL21')*fault_end_ratio),:,:]), axis=0 )

# Min-Max 스케일러 인스턴스 초기화
# scaler = MinMaxScaler()
scaler_IGBT = MinMaxScaler()
scaler_FLUX = MinMaxScaler()
scaler_CAP_I = MinMaxScaler()
scaler_CAP_V = MinMaxScaler()
scaler_MOD_V = MinMaxScaler()
scaler_MOD_I = MinMaxScaler()
scaler_dvdt = MinMaxScaler()

# 테스트 데이터셋 min-max 스케일링.
for i in range(len(features)):
    if i>=0 and i<=5:
        data_test[:,:,i] = scaler_IGBT.fit_transform(data_test[:,:,i].reshape(-1, 1)).reshape(data_test[:,:,i].shape)
        data_anomaly[:,:,i] = scaler_IGBT.fit_transform(data_anomaly[:,:,i].reshape(-1, 1)).reshape(data_anomaly[:,:,i].shape)
    if i>=6 and i<=8:
        data_test[:,:,i] = scaler_FLUX.fit_transform(data_test[:,:,i].reshape(-1, 1)).reshape(data_test[:,:,i].shape)
        data_anomaly[:,:,i] = scaler_FLUX.fit_transform(data_anomaly[:,:,i].reshape(-1, 1)).reshape(data_anomaly[:,:,i].shape)
    if (i==9) :
        data_test[:,:,i] = scaler_CAP_I.fit_transform(data_test[:,:,i].reshape(-1, 1)).reshape(data_test[:,:,i].shape)
        data_anomaly[:,:,i] = scaler_CAP_I.fit_transform(data_anomaly[:,:,i].reshape(-1, 1)).reshape(data_anomaly[:,:,i].shape)
    if (i==10) :
        data_test[:,:,i] = scaler_CAP_V.fit_transform(data_test[:,:,i].reshape(-1, 1)).reshape(data_test[:,:,i].shape)
        data_anomaly[:,:,i] = scaler_CAP_V.fit_transform(data_anomaly[:,:,i].reshape(-1, 1)).reshape(data_anomaly[:,:,i].shape)
    if (i==11) :
        data_test[:,:,i] = scaler_MOD_V.fit_transform(data_test[:,:,i].reshape(-1, 1)).reshape(data_test[:,:,i].shape)
        data_anomaly[:,:,i] = scaler_MOD_V.fit_transform(data_anomaly[:,:,i].reshape(-1, 1)).reshape(data_anomaly[:,:,i].shape)
    if (i==12) :
        data_test[:,:,i] = scaler_MOD_I.fit_transform(data_test[:,:,i].reshape(-1, 1)).reshape(data_test[:,:,i].shape)
        data_anomaly[:,:,i] = scaler_MOD_I.fit_transform(data_anomaly[:,:,i].reshape(-1, 1)).reshape(data_anomaly[:,:,i].shape)
    if (i==13) :
        data_test[:,:,i] = scaler_dvdt.fit_transform(data_test[:,:,i].reshape(-1, 1)).reshape(data_test[:,:,i].shape)
        data_anomaly[:,:,i] = scaler_dvdt.fit_transform(data_anomaly[:,:,i].reshape(-1, 1)).reshape(data_anomaly[:,:,i].shape)
    
# source and target unique waveform slicing
index_slice_start = 0
index_slice_end = 14
data_test = data_test[:,:,index_slice_start:index_slice_end]
data_anomaly = data_anomaly[:,:,index_slice_start:index_slice_end]

# 테스트셋 전치(transpose)
data_test = data_test.transpose(0,2,1)
data_anomaly = data_anomaly.transpose(0,2,1)
# 테스트셋 tensor위에 두기
data_test = torch.tensor(data_test, dtype=torch.float32)
data_anomaly = torch.tensor(data_anomaly, dtype=torch.float32)

# one-hot encoding label
normal_labels = np.concatenate((
    # RFQ module
    labels1[normal_indices_RFQ[int(len(normal_RFQ)*(ratio_target)):int(len(normal_RFQ)*(ratio_test))]],
    # DTL module
    labels2[return_indicies(column_to_search_DTL_normal, 'DTL3')[int(return_len(column_to_search_DTL_normal,'DTL3')*(ratio_target)):int(return_len(column_to_search_DTL_normal,'DTL3')*(ratio_test))]],
    labels2[return_indicies(column_to_search_DTL_normal, 'DTL5')[int(return_len(column_to_search_DTL_normal,'DTL5')*(ratio_target)):int(return_len(column_to_search_DTL_normal,'DTL5')*(ratio_test))]], 
    # CCL module
    labels3[return_indicies(column_to_search_CCL_normal, 'CCL1')[int(return_len(column_to_search_CCL_normal,'CCL1')*(ratio_target)):int(return_len(column_to_search_CCL_normal,'CCL1')*(ratio_test))]],
    labels3[return_indicies(column_to_search_CCL_normal, 'CCL2')[int(return_len(column_to_search_CCL_normal,'CCL2')*(ratio_target)):int(return_len(column_to_search_CCL_normal,'CCL2')*(ratio_test))]],
    labels3[return_indicies(column_to_search_CCL_normal, 'CCL3')[int(return_len(column_to_search_CCL_normal,'CCL3')*(ratio_target)):int(return_len(column_to_search_CCL_normal,'CCL3')*(ratio_test))]],
    labels3[return_indicies(column_to_search_CCL_normal, 'CCL4')[int(return_len(column_to_search_CCL_normal,'CCL4')*(ratio_target)):int(return_len(column_to_search_CCL_normal,'CCL4')*(ratio_test))]],
    # SCL module
    labels4[return_indicies(column_to_search_SCL_normal, 'SCL1')[int(return_len(column_to_search_SCL_normal,'SCL1')*(ratio_target)):int(return_len(column_to_search_SCL_normal,'SCL1')*(ratio_test))]],
    labels4[return_indicies(column_to_search_SCL_normal, 'SCL5')[int(return_len(column_to_search_SCL_normal,'SCL5')*(ratio_target)):int(return_len(column_to_search_SCL_normal,'SCL5')*(ratio_test))]],
    labels4[return_indicies(column_to_search_SCL_normal, 'SCL9')[int(return_len(column_to_search_SCL_normal,'SCL9')*(ratio_target)):int(return_len(column_to_search_SCL_normal,'SCL9')*(ratio_test))]],
    labels4[return_indicies(column_to_search_SCL_normal, 'SCL12')[int(return_len(column_to_search_SCL_normal,'SCL12')*(ratio_target)):int(return_len(column_to_search_SCL_normal,'SCL12')*(ratio_test))]],
    labels4[return_indicies(column_to_search_SCL_normal, 'SCL14')[int(return_len(column_to_search_SCL_normal,'SCL14')*(ratio_target)):int(return_len(column_to_search_SCL_normal,'SCL14')*(ratio_test))]],
    labels4[return_indicies(column_to_search_SCL_normal, 'SCL15')[int(return_len(column_to_search_SCL_normal,'SCL15')*(ratio_target)):int(return_len(column_to_search_SCL_normal,'SCL15')*(ratio_test))]],
    labels4[return_indicies(column_to_search_SCL_normal, 'SCL18')[int(return_len(column_to_search_SCL_normal,'SCL18')*(ratio_target)):int(return_len(column_to_search_SCL_normal,'SCL18')*(ratio_test))]],
    labels4[return_indicies(column_to_search_SCL_normal, 'SCL21')[int(return_len(column_to_search_SCL_normal,'SCL21')*(ratio_target)):int(return_len(column_to_search_SCL_normal,'SCL21')*(ratio_test))+1]]), axis=0)

tmp_normal_labels = np.eye(15)[normal_labels]
labels = torch.tensor(tmp_normal_labels, dtype=torch.float32)

anomaly_labels = np.concatenate((
    # RFQ module
    labels1[fault_indices_RFQ[int(len(abnormal_RFQ)*fault_ratio):int(len(abnormal_RFQ)*fault_end_ratio)]],
    # DTL module
    labels2[return_indicies(column_to_search_DTL, 'DTL3')[int(return_len(column_to_search_DTL,'DTL3')*fault_ratio):int(return_len(column_to_search_DTL, 'DTL3')*fault_end_ratio)]],
    labels2[return_indicies(column_to_search_DTL, 'DTL5')[int(return_len(column_to_search_DTL,'DTL5')*fault_ratio):int(return_len(column_to_search_DTL, 'DTL5')*fault_end_ratio)]],
    # CCL module
    labels3[return_indicies(column_to_search_CCL, 'CCL1')[int(return_len(column_to_search_CCL,'CCL1')*fault_ratio):int(return_len(column_to_search_CCL, 'CCL1')*fault_end_ratio)]],
    labels3[return_indicies(column_to_search_CCL, 'CCL2')[int(return_len(column_to_search_CCL,'CCL2')*fault_ratio):int(return_len(column_to_search_CCL, 'CCL2')*fault_end_ratio)]],
    labels3[return_indicies(column_to_search_CCL, 'CCL3')[int(return_len(column_to_search_CCL,'CCL3')*fault_ratio):int(return_len(column_to_search_CCL, 'CCL3')*fault_end_ratio)]],
    labels3[return_indicies(column_to_search_CCL, 'CCL4')[int(return_len(column_to_search_CCL,'CCL4')*fault_ratio):int(return_len(column_to_search_CCL, 'CCL4')*fault_end_ratio)]],
    # SCL module
    labels4[return_indicies(column_to_search_SCL, 'SCL1')[int(return_len(column_to_search_SCL,'SCL1')*fault_ratio):int(return_len(column_to_search_SCL, 'SCL1')*fault_end_ratio)]],
    labels4[return_indicies(column_to_search_SCL, 'SCL5')[int(return_len(column_to_search_SCL,'SCL5')*fault_ratio):int(return_len(column_to_search_SCL, 'SCL5')*fault_end_ratio)]],
    labels4[return_indicies(column_to_search_SCL, 'SCL9')[int(return_len(column_to_search_SCL,'SCL9')*fault_ratio):int(return_len(column_to_search_SCL, 'SCL9')*fault_end_ratio)]],
    labels4[return_indicies(column_to_search_SCL, 'SCL12')[int(return_len(column_to_search_SCL,'SCL12')*fault_ratio):int(return_len(column_to_search_SCL, 'SCL12')*fault_end_ratio)]],
    labels4[return_indicies(column_to_search_SCL, 'SCL14')[int(return_len(column_to_search_SCL,'SCL14')*fault_ratio):int(return_len(column_to_search_SCL, 'SCL14')*fault_end_ratio)]],
    labels4[return_indicies(column_to_search_SCL, 'SCL15')[int(return_len(column_to_search_SCL,'SCL15')*fault_ratio):int(return_len(column_to_search_SCL, 'SCL15')*fault_end_ratio)]],
    labels4[return_indicies(column_to_search_SCL, 'SCL18')[int(return_len(column_to_search_SCL,'SCL18')*fault_ratio):int(return_len(column_to_search_SCL, 'SCL18')*fault_end_ratio)]],
    labels4[return_indicies(column_to_search_SCL, 'SCL21')[int(return_len(column_to_search_SCL,'SCL21')*fault_ratio):int(return_len(column_to_search_SCL, 'SCL21')*fault_end_ratio)]]), axis=0)

tmp_fault = np.eye(15)[anomaly_labels]
anomaly_labels = torch.tensor(tmp_fault, dtype=torch.float32)

# GPU 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
# 모델 및 학습파라미터 설정
input_dim = (index_slice_end-index_slice_start) # default : 14
hidden_dim = 4500 # default : 4500
latent_dim = 512
condition_dim = 15 # RFQ(0), DTL(1), CCL(2), SCL(3) 총 4가지의 시스템 모듈 이름을 조건으로 사용.
dropout = 0.2
num_epochs = 100

# 베이스라인 학습 모델 초기화
baseline_model = CVAE_baseline(input_dim=input_dim, latent_dim=latent_dim, hidden_dim=hidden_dim, condition_dim=condition_dim, dropout_prob=dropout).to(device)
baseline_model.load_state_dict(torch.load('./model/baseline.pth'))

# 제안한 학습 모델 초기화
proposed_model = CVAE_rev(input_dim=input_dim, latent_dim=latent_dim, hidden_dim=hidden_dim, condition_dim=condition_dim, dropout_prob=dropout).to(device)
proposed_model.load_state_dict(torch.load('./model/proposal.pth'))

# 손실 함수 및 옵티마이저
optimizer_baseline = optim.Adam(baseline_model.parameters(), lr=1e-5)
optimizer_pretrain = optim.Adam(proposed_model.parameters(), lr=1e-3)

# Target encoder pretraining function
# to train discriminator
baseline_model.eval()
proposed_model.eval()


import pandas as pd

# 모델 분석하기.

########################################################################################################################################################################
######################################################################## 1. 모델 파라미터 수 비교 ########################################################################
########################################################################################################################################################################

baseline_model_params = sum(p.numel() for p in baseline_model.parameters())
proposed_model_params = sum(p.numel() for p in proposed_model.parameters())

print(f"Baseline Model parameters: {baseline_model_params}")
print(f"Proposed Model parameters: {proposed_model_params}")

############################################################################################################################################################################################################
######################################################################## 2. 모델 구조(각 모델의 레이어와 레이어의 파라미터 수 출력) 비교 ########################################################################
############################################################################################################################################################################################################

from torchinfo import summary

# 입력 데이터 생성
x_input = torch.ones(2, 14, 4500)  # 첫 번째 입력 텐서: x
c_input = torch.ones(2, 15)        # 두 번째 입력 텐서: c

# 입력 데이터를 GPU로 이동
x_input = x_input.to(device)
c_input = c_input.to(device)

print("Baseline Summary:")
summary(baseline_model, input_data=(x_input, c_input))

print("\nProposed 2 Summary:")
summary(proposed_model, input_data=(x_input, c_input))

############################################################################################################################################################################################################
######################################################################## 3. 모델 내 활용되는 연산량 비교(모델의 계산복잡도 비교) ###############################################################################
############################################################################################################################################################################################################

# Use a library like thop to calculate FLOPs
from thop import profile

flops_1, params_1 = profile(baseline_model, inputs=(x_input,c_input))
flops_2, params_2 = profile(proposed_model, inputs=(x_input,c_input))

print(f"Baseline Model FLOPs: {flops_1}, Parameters: {params_1}")
print(f"Proposed Model FLOPs: {flops_2}, Parameters: {params_2}")
