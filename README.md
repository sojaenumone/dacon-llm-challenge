# dacon-llm-challenge
dacon-llm-challenge
# 🧠 AI-Generated Text Classifier with XGBoost

이 프로젝트는 생성형 AI가 쓴 글을 판별하기 위해 `TF-IDF`와 `XGBoost`를 활용한 머신러닝 모델입니다. 해당 코드는 [DACON "생성형 AI와 인간: 텍스트 판별 챌린지"](https://dacon.io/competitions/official/236473/overview/description) 참가를 위해 제작되었습니다.

## 📁 프로젝트 구성

```bash
📦 프로젝트 루트
├── main.py                     # 메인 실행 코드
├── train.csv                   # 학습 데이터 (로컬 필요)
├── test.csv                    # 테스트 데이터 (로컬 필요)
├── sample_submission.csv       # 제출 템플릿 (로컬 필요)
└── submission_YYYYMMDD_HHMMSS.csv  # 실행 시 자동 생성되는 결과 파일
