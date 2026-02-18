# 🧠 심리 성향 예측 AI 경진대회 (Psychological Trait Prediction)

안녕하세요, **심리 성향 예측 AI 경진대회**에 오신 것을 환영합니다!

이번 프로젝트는 **성향 문항 응답(Qa~Qt)**, **응답 특성(문항별 시간 지표)**, 그리고 **인구통계·환경 정보**를 활용해 개인의 **심리/행동 성향(voted)** 을 예측하는 모델을 개발하는 것을 목표로 합니다.  
데이터에서 인사이트를 발굴하고, 더 일반화 성능이 뛰어난 알고리즘을 설계하며, “사람을 이해하는 AI”의 가능성을 확장해 보세요.

---

## 📌 핵심 아이디어 요약

- **전처리**
  - 이상치 제거: `familysize`가 비정상적으로 큰 샘플 제거 (모델별 기준 상이)
  - 역채점(Reverse coding): 특정 문항(`QaA, QdA, ...`)을 `6 - 응답값`으로 변환
  - 시간 지표(`Q*E`)는 모델에 따라 **제거**하거나 **집계 특징**으로 사용

- **특징 공학(Feature Engineering)**
  - `mach_score`: Qa~Qt 응답 평균
  - `ans_var`: Qa~Qt 응답 분산
  - `total_time`: 시간 지표 합을 `log1p(sum)`로 변환

- **모델링**
  - ResNet 계열 MLP, Wide&Deep + SEBlock, 1D-CNN, Denoising AutoEncoder(DAE), TabNet 등 **다양한 tabular 딥러닝 모델** 구성
  - 다중 Seed/Repeated Stratified K-Fold로 일반화 성능 강화

- **앙상블(Ensemble)**
  - 모델별 예측 확률을 **가중 평균** 후, 간단한 **보정(calibration)** 및 변환 적용

---

## 🧱 Repository Structure

> 데이터 파일(train/test/sample_submission)은 저작권/대회 규정상 포함하지 않습니다. 실행 전 동일 경로에 배치해주세요.

```bash
.
├── Model1.ipynb          # ResNet-style MLP (SiLU) + Repeated Stratified K-Fold
├── Model2.ipynb          # ResNet-style MLP (dropout/WD/scheduler 변경)
├── Model3.ipynb          # Wide&Deep + SEBlock 하이브리드
├── Model4.ipynb          # TabNet + Unsupervised Pretraining + CV
├── Model5.ipynb          # Tabular 1D-CNN
├── Model6.ipynb          # DAE(denoising autoencoder) pretrain + frozen encoder classifier
├── Model7.ipynb          # (제출 생성용) TabNet 예측값(tabnet_preds) 기반 CSV 저장
├── Model8.ipynb          # CNN 기반 분류기 (expansion + conv)
├── Model9.ipynb          # GELU 기반 ResNet MLP
└── Ensemble.ipynb        # Model1~9 결과를 가중 앙상블하여 최종 제출 파일 생성
