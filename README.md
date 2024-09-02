# SNS-HVCM

이 프로젝트는 논문 "Multi-module-based CVAE to predict HVCM faults in the SNS accelerator"에서 사용된 모델을 기반으로 하여, SNS accelerator의 운행 데이터셋에 대해서 개선된 성능을 보이는 학습방식 및 모델을 구현한 내용입니다.

프로젝트의 주요 기여도(contribution)으로는 다음과 같이 있습니다.

- 이상치 탐지 성능 향상 (AUC : 0.73-->1.00, F1-score : 0.48 --> 0.99)
- 개선 사항 :
    - 손실함수 수정 : Mean Square Error --> Square Error (복원손실을 더 정확하게 학습함)
    - 모델 구조 개선 : 활성화 함수 ReLU--> GELU (모델의 학습성능 향상)
    - 학습률 튜닝 및 학습 스케쥴러 추가 : 1e-5 --> 1e-3 상향 후 스케쥴링 추가. (최적의 손실값을 학습하게 됨)

결론 
- 개선된 모델 및 학습 하이퍼 파라미터를 이용하여, 실제 가속기 운영 중에 발생한 사전오류(Pre-Fault) 데이터에도 적용해 볼 수 있는 실용성이 높은 모델을 학습해내었다.

# Usage

1. 베이스라인 모델 및 제안된 모델 학습하기  
    $ python baseline.py --path "저장할 모델 경로"  
    $ python proposal.py --path "저장할 모델 경로"  

2-1. 저장된 모델 테스트하기(ROC-Curve, AUC, F1-score, Confusion Matrix)  
    $ python plot_f1_auc.py --name "Baseline or Proposal 중 보고 싶은 혼동행렬 선택" --figure_path "plot 결과를 저장할 경로"  

2-2. 저장된 모델 테스트하기(Histogram, Kernel Density Estimation plot, Boxplot)  
    $ python plot_kde_box_tsne.py --figure_path  "plot 결과를 저장할 경로"  


