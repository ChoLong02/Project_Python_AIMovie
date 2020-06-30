# PJ_Python_AIMovie

### 사용한 데이터셋
데이터셋: Naver Sentiment Movie Corpus(https://github.com/e9t/nsmc/)  
네이버 영화 리뷰 중 영화당 100개의 리뷰를 모아  
총 200,000개의 리뷰(훈련: 15만개, 테스트: 5만개)로  
이루어져있고, 1-10점까지의 평점 중 중립적인 평점(5-8)은  
제외하고 1-4점을 긍정, 9~10점을 부정으로 동일한 비율로  
데이터에 포함시킴  


### 사용한 프로그램
1.Python  
2.BeautifulSoup  
3.Selenium  
4.MongoDB  
5.Numpy  
6.Konlpy  
7.Tensorflow  


### 파이썬기반의 AI를 활용한 감성분석
1.BeautifulSoup, Selenium을 활용한 네이버와 다음 영화평 수집  
2.MongoDB에 저장  
3.AI학습  
4.수집된 영화평 전처리 후 AI에 입력  
5.AI가 감성분석을 통해 ?%확률로 긍정, 부정 판별  
