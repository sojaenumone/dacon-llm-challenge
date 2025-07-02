# 🤖 AI 텍스트 판별기 (TF-IDF + XGBoost)

이 프로젝트는 생성형 AI 텍스트 판별 챌린지를 위한 **베이스라인 모델**로, `TF-IDF` 기반 벡터화와 `XGBoost` 분류기를 활용하여 입력된 문장이 인간이 작성한 것인지 AI가 생성한 것인지를 이진 분류합니다.

---

## 📁 프로젝트 구조

```
.
├── main.py                # 실행 코드 (모델 학습 및 예측)
├── train.csv              # 학습 데이터
├── test.csv               # 테스트 데이터
├── sample_submission.csv # 제출 양식 파일
└── README.md              # 프로젝트 설명
```

---

## 🗂️ 데이터 설명

### 🔹 train.csv
| 컬럼명       | 설명                                   |
|--------------|----------------------------------------|
| `full_text`  | 전체 문단 (입력 텍스트)                |
| `generated`  | AI 생성 여부 (1: AI 생성, 0: 인간 작성) |

### 🔹 test.csv
| 컬럼명         | 설명                            |
|----------------|---------------------------------|
| `paragraph_text` | 테스트용 문단 (예측 대상)         |

### 🔹 sample_submission.csv
| 컬럼명       | 설명                            |
|--------------|---------------------------------|
| `id`         | 고유 ID                         |
| `generated`  | 예측 결과 (0 또는 1로 이진 분류) |

---

## ⚙️ 실행 방법

### 1. 의존성 설치

```bash
pip install pandas scikit-learn xgboost
```

### 2. 파일 구조 구성

아래 세 파일이 다음 경로에 있어야 합니다:

```
/Users/nextweb/Downloads/open/
├── train.csv
├── test.csv
└── sample_submission.csv
```

### 3. 코드 실행

```bash
python main.py
```

실행 후 결과 제출 파일이 다음 위치에 생성됩니다:

```
/Users/nextweb/Downloads/open/submission_YYYYMMDD_HHMMSS.csv
```

---

## 🧠 사용된 모델 및 설정

- **TF-IDF 벡터화**
  - `max_features=5000`
- **XGBoost 분류기**
```python
XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='logloss'
)
```

---

## 📈 예측 방식

- `predict_proba`를 통해 AI 생성 확률을 추정
- 확률값이 `0.5 이상`이면 1 (AI 생성), 아니면 0 (인간 작성)

---

## ✅ 결과 예시

| id  | generated |
|-----|-----------|
| 0   | 1         |
| 1   | 0         |
| 2   | 1         |

---

## 📄 라이선스

MIT License

---

## 🙋‍♀️ 기여

기여를 원하신다면 Pull Request나 Issues를 열어주세요!

