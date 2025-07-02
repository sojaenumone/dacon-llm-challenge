import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import datetime

# 현재 작업 디렉터리 출력 (디버깅용)
print("Current directory:", os.getcwd())
print("Files:", os.listdir())

# 데이터 파일 절대 경로 지정
train_path = "/Users/nextweb/Downloads/open/train.csv"
test_path = "/Users/nextweb/Downloads/open/test.csv"
submission_path = "/Users/nextweb/Downloads/open/sample_submission.csv"

# 파일 존재 여부 확인 (없으면 경고)
for fpath in [train_path, test_path, submission_path]:
    if not os.path.isfile(fpath):
        raise FileNotFoundError(f"{fpath} 파일이 존재하지 않습니다.")

# 데이터 불러오기
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
submission = pd.read_csv(submission_path)

# TF-IDF 벡터화 (train full_text 컬럼 사용)
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train_df['full_text'])
y_train = train_df['generated']

# 모델 학습 (random_state 미지정 → 랜덤성 유지)
model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='logloss'
)
model.fit(X_train, y_train)

# 테스트 데이터의 paragraph_text 컬럼 확인
if 'paragraph_text' not in test_df.columns:
    raise KeyError("'paragraph_text' 컬럼이 test 데이터에 없습니다.")

# 테스트 데이터 전처리 및 예측
X_test = vectorizer.transform(test_df['paragraph_text'])
test_preds = model.predict_proba(X_test)[:, 1]  # AI 생성일 확률

# 확률을 0 또는 1로 변환 (임계값 0.5 기준)
submission['generated'] = (test_preds >= 0.5).astype(int)

# 현재 시간(년월일_시분초) 문자열 생성
now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# 제출 파일 저장 (시간 포함 파일명)
submission.to_csv(f"/Users/nextweb/Downloads/open/submission_{now}.csv", index=False, encoding='utf-8')

print(f"✅ 제출용 submission_{now}.csv 생성 완료!")