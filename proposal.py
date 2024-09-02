import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import random
from proposed_model import *
from tqdm import tqdm
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--paths', type = str,
                    default = 'model', help='Enter the stored model path')
args = parser.parse_args()

def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")

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
    system4: 3,
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

# to load the trained data
Xnormal_RFQ, Xanomaly_RFQ = X1[normal_indices_RFQ,:,:], X1[fault_indices_RFQ,:,:]
Xnormal_DTL, Xanomaly_DTL = X2[normal_indices_DTL,:,:], X2[fault_indices_DTL,:,:]
Xnormal_CCL, Xanomaly_CCL = X3[normal_indices_CCL,:,:], X3[fault_indices_CCL,:,:]
Xnormal_SCL, Xanomaly_SCL = X4[normal_indices_SCL,:,:], X4[fault_indices_SCL,:,:]

# to load the labeled data
Ynormal_RFQ, Yanomaly_RFQ = Y1[normal_indices_RFQ,:], Y1[fault_indices_RFQ,:]
Ynormal_DTL, Yanomaly_DTL = Y2[normal_indices_DTL,:], Y2[fault_indices_DTL,:]
Ynormal_CCL, Yanomaly_CCL = Y3[normal_indices_CCL,:], Y3[fault_indices_CCL,:]
Ynormal_SCL, Yanomaly_SCL = Y4[normal_indices_SCL,:], Y4[fault_indices_SCL,:]

# 모듈이름과 그에 대한 정상 및 비정상신호 출력
def search_and_print(data, module_name, search_terms):
    print(f"Module-{module_name}")
    for term in search_terms:
        count = len(np.where(np.char.find(data, term) != -1)[0])
        print(f'{term} : {count}')

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

column_to_search_DTL = Yanomaly_DTL[:, 0].astype(str)
column_to_search_CCL = Yanomaly_CCL[:, 0].astype(str)
column_to_search_SCL = Yanomaly_SCL[:, 0].astype(str)

# # 모듈이름별 정상신호 갯수 출력
# search_and_print(column_to_search_DTL_normal, 'DTL Normal', ['DTL3', 'DTL5'])
# search_and_print(column_to_search_CCL_normal, 'CCL Normal', ['CCL1', 'CCL2', 'CCL3', 'CCL4'])
# search_and_print(column_to_search_SCL_normal, 'SCL Normal', ['SCL1', 'SCL5', 'SCL9', 'SCL12', 'SCL14', 'SCL15', 'SCL18', 'SCL21'])

# # 모듈이름별 비정상신호 갯수 출력
# search_and_print(column_to_search_DTL, 'DTL Fault', ['DTL3', 'DTL5'])
# search_and_print(column_to_search_CCL, 'CCL Fault', ['CCL1', 'CCL2', 'CCL3', 'CCL4'])
# search_and_print(column_to_search_SCL, 'SCL Fault', ['SCL1', 'SCL5', 'SCL9', 'SCL12', 'SCL14', 'SCL15', 'SCL18', 'SCL21'])
 

############################### 15가지 모듈의 정상신호 모음 ###############################
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

############################### 15가지 모듈의 이상신호 모음 ###############################
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

ratio = 0.8
Xnormal_concat = np.concatenate( 
    (normal_RFQ[:int(len(normal_RFQ)*ratio),:,:],
     normal_DTL3[:int(return_len(column_to_search_DTL_normal,'DTL3')*ratio),:,:],
     normal_DTL5[:int(return_len(column_to_search_DTL_normal,'DTL5')*ratio),:,:],
     normal_CCL1[:int(return_len(column_to_search_CCL_normal,'CCL1')*ratio),:,:],
     normal_CCL2[:int(return_len(column_to_search_CCL_normal,'CCL2')*ratio),:,:],
     normal_CCL3[:int(return_len(column_to_search_CCL_normal,'CCL3')*ratio),:,:],
     normal_CCL4[:int(return_len(column_to_search_CCL_normal,'CCL4')*ratio),:,:],
     normal_SCL1[:int(return_len(column_to_search_SCL_normal,'SCL1')*ratio),:,:],
     normal_SCL5[:int(return_len(column_to_search_SCL_normal,'SCL5')*ratio),:,:],
     normal_SCL9[:int(return_len(column_to_search_SCL_normal,'SCL9')*ratio),:,:],
     normal_SCL12[:int(return_len(column_to_search_SCL_normal,'SCL12')*ratio),:,:],
     normal_SCL14[:int(return_len(column_to_search_SCL_normal,'SCL14')*ratio),:,:],
     normal_SCL15[:int(return_len(column_to_search_SCL_normal,'SCL15')*ratio),:,:],
     normal_SCL18[:int(return_len(column_to_search_SCL_normal,'SCL18')*ratio),:,:],
     normal_SCL21[:int(return_len(column_to_search_SCL_normal,'SCL21')*ratio),:,:]), axis=0 )

Xnormal_concat = np.array(Xnormal_concat)

# 정상신호 N개 학습
data1 = Xnormal_concat[:,:,:] # 15가지 모듈(RFQ, DTL3, DTL5, CCL1, CCL2, CCL3, CCL4, SCL1, SCL5, SCL9, SCL12, SCL14, SCL15, SCL18, SCL21)

# Min-Max 스케일러 인스턴스 초기화
# scaler = MinMaxScaler()
scaler_IGBT = MinMaxScaler()
scaler_FLUX = MinMaxScaler()
scaler_CAP_I = MinMaxScaler()
scaler_CAP_V = MinMaxScaler()
scaler_MOD_V = MinMaxScaler()
scaler_MOD_I = MinMaxScaler()
scaler_dvdt = MinMaxScaler()

# 데이터 Min-Max 스케일링
for i in range(len(features)):
    if i<=5 or (i>=14 and i<=19):
        data1[:,:,i] = scaler_IGBT.fit_transform(np.array(data1[:,:,i]).reshape(-1,1)).reshape(np.array(data1[:,:,i]).shape)
    if (i>=6 and i<=8) :
        data1[:,:,i] = scaler_FLUX.fit_transform(data1[:,:,i].reshape(-1,1)).reshape(data1[:,:,i].shape)
    if (i==9) :
        data1[:,:,i] = scaler_CAP_I.fit_transform(data1[:,:,i].reshape(-1,1)).reshape(data1[:,:,i].shape)
    if (i==10) :
        data1[:,:,i] = scaler_CAP_V.fit_transform(data1[:,:,i].reshape(-1,1)).reshape(data1[:,:,i].shape)    
    if (i==11):
        data1[:,:,i] = scaler_MOD_V.fit_transform(data1[:,:,i].reshape(-1,1)).reshape(data1[:,:,i].shape)
    if (i==12):
        data1[:,:,i] = scaler_MOD_I.fit_transform(data1[:,:,i].reshape(-1,1)).reshape(data1[:,:,i].shape)
    if (i==13):
        data1[:,:,i] = scaler_dvdt.fit_transform(data1[:,:,i].reshape(-1,1)).reshape(data1[:,:,i].shape)

# 고유파형별 인덱스 슬라이싱
index_slice_start = 0
index_slice_end = 14
data1 = data1[:,:,index_slice_start:index_slice_end]
data1 = data1.transpose(0,2,1)  # (N,14,4500) 형태로 변경하여 Conv1D 입력 형식에 맞춤 
data1 = torch.tensor(data1, dtype=torch.float32)


# 원-핫 인코딩
labels_concat = np.concatenate((labels1[normal_indices_RFQ[:int(len(normal_RFQ)*ratio)]],
                                
                                labels2[return_indicies(column_to_search_DTL_normal, 'DTL3')[:int(return_len(column_to_search_DTL_normal,'DTL3')*ratio)]],
                                labels2[return_indicies(column_to_search_DTL_normal, 'DTL5')[:int(return_len(column_to_search_DTL_normal,'DTL5')*ratio)]], 

                                labels3[return_indicies(column_to_search_CCL_normal, 'CCL1')[:int(return_len(column_to_search_CCL_normal,'CCL1')*ratio)]],
                                labels3[return_indicies(column_to_search_CCL_normal, 'CCL2')[:int(return_len(column_to_search_CCL_normal,'CCL2')*ratio)]],
                                labels3[return_indicies(column_to_search_CCL_normal, 'CCL3')[:int(return_len(column_to_search_CCL_normal,'CCL3')*ratio)]],
                                labels3[return_indicies(column_to_search_CCL_normal, 'CCL4')[:int(return_len(column_to_search_CCL_normal,'CCL4')*ratio)]],

                                labels4[return_indicies(column_to_search_SCL_normal, 'SCL1')[:int(return_len(column_to_search_SCL_normal,'SCL1')*ratio)]],
                                labels4[return_indicies(column_to_search_SCL_normal, 'SCL5')[:int(return_len(column_to_search_SCL_normal,'SCL5')*ratio)]],
                                labels4[return_indicies(column_to_search_SCL_normal, 'SCL9')[:int(return_len(column_to_search_SCL_normal,'SCL9')*ratio)]],
                                labels4[return_indicies(column_to_search_SCL_normal, 'SCL12')[:int(return_len(column_to_search_SCL_normal,'SCL12')*ratio)]],
                                labels4[return_indicies(column_to_search_SCL_normal, 'SCL14')[:int(return_len(column_to_search_SCL_normal,'SCL14')*ratio)]],
                                labels4[return_indicies(column_to_search_SCL_normal, 'SCL15')[:int(return_len(column_to_search_SCL_normal,'SCL15')*ratio)]],
                                labels4[return_indicies(column_to_search_SCL_normal, 'SCL18')[:int(return_len(column_to_search_SCL_normal,'SCL18')*ratio)]],
                                labels4[return_indicies(column_to_search_SCL_normal, 'SCL21')[:int(return_len(column_to_search_SCL_normal,'SCL21')*ratio)]]), axis=0)
labels_one_hot = np.eye(15)[labels_concat]
labels_one_hot = torch.tensor(labels_one_hot, dtype=torch.float32) # to use the output of conditional VAE
# print(labels_one_hot.shape)

dataset1 = CustomDataset(data1, labels_one_hot)
dataloader1 = DataLoader(dataset1, batch_size=16, shuffle=True) # batch_size default : 16

# 손실 함수 정의 - my proposal
def loss_function_sum(x_recon, x, mu, logvar):    

    MSE_lib = nn.functional.mse_loss(x_recon, x,  reduction='sum') # divided into batch size, time steps
    KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1), dim=0)# see Appendix B from VAE paper:
                                                                                           # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    
    return MSE_lib + 1.0*KLD + 1e-12

# 손실함수 정의 (MSE(reduction='mean') + KLD)
def loss_function_mean(x_recon, x, mu, logvar):    

    MSE_lib = nn.functional.mse_loss(x_recon, x, reduction='sum') # divided into batch size, time steps
    KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1), dim=0)# see Appendix B from VAE paper:
                                                                                           # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    
    return MSE_lib + 1.0*KLD + 1e-12

# GPU 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 및 학습파라미터 설정
input_dim = (index_slice_end - index_slice_start) # default : 14
hidden_dim = 4500 # default : 4500
latent_dim = 512 # default : 512
condition_dim = 15 # RFQ(0), DTL(1), CCL(2), SCL(3) 총 4가지의 시스템 모듈 이름을 조건으로 사용.
dropout = 0.2

num_epochs = 100
num_trials = 1   # default : 150
tmp = 100000000

# main 함수
if __name__ =='__main__':

    # 모델 초기화
    source_model = CVAE_rev(
        input_dim=input_dim, 
        latent_dim=latent_dim,
        hidden_dim=hidden_dim, 
        condition_dim=condition_dim,
        dropout_prob=dropout
    ).to(device)

    source_model.train()
    # 손실 함수 및 옵티마이저
    optimizer_source = optim.Adam(source_model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_source, mode='min', factor=0.3, patience=3, verbose=True)

    for epoch in range(num_epochs):
        total_loss = 0
        middle_loss = 0
        # tqdm을 사용하여 학습 진행 상황을 시각적으로 표시
        pbar = tqdm(dataloader1, desc=f'Epoch {epoch + 1}/{num_epochs}', total=len(dataloader1), ncols=100)

        for x, c in pbar:
            x = x.to(device)
            c = c.to(device)
            optimizer_source.zero_grad()

            x_recon, mu, logvar = source_model(x, c)
            # loss = loss_function_mean(x_recon, x, mu, logvar)
            loss = loss_function_sum(x_recon, x, mu, logvar)
            loss.backward()
            middle_loss += loss.item()
            optimizer_source.step()

            # total_loss += middle_loss / (4500 * (index_slice_end - index_slice_start)) # default total losses
            total_loss += middle_loss / ((index_slice_end - index_slice_start)) # tunnedd total losses

            # tqdm의 진행 표시줄에 손실 값 업데이트
            pbar.set_postfix(loss=total_loss)
        scheduler.step(total_loss)
        avg_loss = total_loss / (len(dataloader1)+1) # 데이터로더의 길이는 전체 샘플수 / 배치사이즈 한 값이다.
        print(f' Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}')
    print("사전 모델 학습 완료!")
    # torch.save(source_model.state_dict(), "./model/0826/proposed_activation_gelu_reduction_mean.pth")
    createDirectory(args.paths)
    torch.save(source_model.state_dict(), "./%s"%args.paths+"/proposal.pth")
