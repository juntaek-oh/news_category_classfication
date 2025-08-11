# 📰 News Category Classification

**뉴스 기사 제목을 기반으로 카테고리를 분류하는 머신러닝 프로젝트**  
CountVectorizer, TF-IDF, Word2Vec 등 다양한 벡터화 기법과 모델을 비교 적용합니다.

---

## 📸 프로젝트 구성 흐름
1. **뉴스 데이터 수집** — 헤드라인 기사 크롤링  
2. **데이터 병합·전처리** — 텍스트 정제, 불용어 제거  
3. **벡터화 & 모델 학습** — 머신러닝/딥러닝 모델 비교  
4. **카테고리 예측** — 평가 및 예측 결과 확인  

---

## ✨ 주요 기능
- 📰 **헤드라인 뉴스 크롤링**: 카테고리별 기사 수집  
- 🔤 **텍스트 벡터화**: TF-IDF, CountVectorizer, Word2Vec 지원  
- 🤖 **분류 모델**: NB, SVM, RandomForest, LSTM 등  
- 📊 **성능 평가**: 정확도, F1-score 등 주요 평가지표 출력  

---

## 🛠 설치 & 실행
필수 라이브러리 설치
pip install -r requirements.txt

뉴스 카테고리 예측 실행
python job06_section_predict.py


---

## 📂 파일 구조
news_category_classfication/
├── crawling_data/ # 뉴스 데이터
├── models/ # 저장된 모델
├── job01_crawling_headline.py # 뉴스 헤드라인 크롤링
├── job03_concat_data.py # 데이터 병합
├── job04_preprocess.py # 전처리
├── job05_model_learning.py # 모델 학습
└── job06_section_predict.py # 카테고리 예측 실행


---

## 🚨 트러블 슈팅
| 문제 | 해결 방법 |
|------|-----------|
| 특정 뉴스 페이지 크롤링 실패 | HTML 구조 변경 시 CSS Selector, XPath 수정 |
| Word2Vec 학습 속도 저하 | 학습 벡터 차원 축소 or 데이터 샘플링 |
| 예측 정확도 불안정 | 데이터 불균형 해결 (oversampling/undersampling) |
| 모델 학습 중 메모리 부족 | 배치 학습 적용, 메모리 최적화 |

---

## 📋 기술 스택
- **언어**: Python  
- **크롤링**: Selenium, BeautifulSoup  
- **분석/모델링**: Pandas, scikit-learn, Gensim, TensorFlow/Keras  
- **환경**: Python 3.10  
