     #################
     # Dataset Intro #
     #################

# 데이터셋: Naver Sentiment Movie Corpus(https://github.com/e9t/nsmc/)
# >> 네이버 영화 리뷰 중 영화당 100개의 리뷰를 모아
# >> 총 200,000개의 리뷰(훈련: 15만개, 테스트: 5만개)로
# >> 이루어져있고, 1~10점까지의 평점 중 중립적인 평점(5~8)은
# >> 제외하고 1~4점을 긍정, 9~10점을 부정으로 동일한 비율로
# >> 데이터에 포함시킴

# >> 데이터는 id, document, label 세개의 열로 이루어져있음
# >> id: 리뷰의 고유한 Key값
# >> document: 리뷰의 내용
# >> label: 긍정(1)인지 부정(0)인지 나타냄
#           평점이 긍정(9~10점), 부정(1~4점), 5~8점은 제거

import json
import os
import nltk
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager, rc
from pprint import pprint
from konlpy.tag import Okt
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics

     #############
     # File Open #
     #############

# *.txt 파일에서 데이터를 불러오는 메서드
def read_data(filename):
    with open(filename, 'r', encoding='UTF-8') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
        data = data[1:] # id는 제외
    return data

# nsmc 데이터를 불러와서 python 변수에 담기
train_data = read_data('ratings_train.txt') # 트레이닝 데이터 Open
test_data = read_data('ratings_test.txt') # 테스트 데이터 Open

# 데이터 확인
# print(len(train_data))
# print(train_data[0])
# print(len(test_data))
# print(test_data[0])

     #################
     # PreProcessing #
     #################

# 데이터를 학습하기에 알맞게 처리해보자. konlpy 라이브러리를 사용해서
# 형태소 분석 및 품사 태깅을 진행한다. 네이버 영화 데이터는
# 맞춤법이나 띄어쓰기가 제대로 되어있지 않은 경우가 있기 때문에
# 정확한 분류를 위해서 konlpy를 사용한다.
# konlpy에는 여러 클래스가 존재하지만 그중 okt(open korean text)를
# 사용하여 간단한 문장분석을 실행한다.
okt = Okt()

#print(okt.pos('이 밤 그날의 반딧불을 당신의 창 가까이 보낼게요'))

# 트레인, 테스트 데이터셋에 형태소 분석을 통해 품사태깅 작업 진행

def tokenize(doc):
    # norm은 정규화, stem은 근어로 표시하기를 나타냄
    return ['/'.join(t) for t in okt.pos(doc, norm=True, stem=True)]

if os.path.isfile('train_docs.json'):
    with open('train_docs.json', 'r', encoding='UTF-8') as f:
        train_docs = json.load(f)
    with open('test_docs.json', 'r', encoding='UTF-8') as f:
        test_docs = json.load(f)
else:
    train_docs = [(tokenize(row[1]), row[2]) for row in train_data]
    test_docs = [(tokenize(row[1]), row[2]) for row in test_data]
    # JSON 파일로 저장
    with open('train_docs.json', 'w', encoding='UTF-8') as make_file:
        json.dump(train_docs, make_file, ensure_ascii=False, indent='\t')
    with open('test_docs.json', 'w', encoding='UTF-8') as make_file:
        json.dump(test_docs, make_file, ensure_ascii=False, indent='\t')

# 전처리 작업 데이터 확인
# pprint(train_docs[0])
# pprint(test_docs[0])

# 분석한 데이터의 토큰(문자열 분석을 위한 작은 단위)의 갯수를 확인
tokens = [t for d in train_docs for t in d[0]]
# print(len(tokens))

# 이 데이터를 nltk 라이브러리를 통해서 전처리,
# vocab().most_common를 이용해서 가장 자주 사용되는 단어 빈도수 확인
text = nltk.Text(tokens, name='NSMC')

# 전체 토큰의 개수
#print(len(text.tokens))

# 중복을 제외한 토큰의 개수
#print(len(set(text.tokens)))

# 출현 빈도가 높은 상위 토큰 10개
#pprint(text.vocab().most_common(10))

# 자주 출현하는 단어 50개를 matplotlib을 통해 그래프로 그리기
# 한글폰트를 로드해야 깨지지 않고 출력 됨
font_fname = 'C:\Windows\Fonts\gulim.ttc'
font_name = font_manager.FontProperties(fname=font_fname).get_name()
rc('font', family=font_name)

# plt.figure(figsize=(20,10))
# text.plot(50)
# plt.show()

# 자주 사용되는 토큰 5000개를 사용해서 데이터를 벡터화 시킨다
# 문서 집합에서 단어 토큰을 생성하고 각 단어의 수를 세어
# BOW(Bag of Words) 인코딩한 벡터를 만드는 역할을 한다.
select_words = [f[0] for f in text.vocab().most_common(5000)]
print('type:', type(select_words))
print('len:', len(select_words))
print('data:', select_words[:10])

f = open('selectword.txt', 'w', encoding='UTF-8')
print('>> selectword 파일 저장 시작')
for i in select_words:
    i += '\n'
    f.write(i)
f.close()
print('>> 파일 저장 완료')


def term_frequency(doc):
    return [doc.count(word) for word in select_words]

train_x = [term_frequency(d) for d, _ in train_docs]
test_x = [term_frequency(d) for d, _ in test_docs]
train_y = [c for _, c in train_docs]
test_y = [c for _, c in test_docs]

# 이제 데이터를 float로 형 변환 시켜주면 데이터 전처리 과정을 끝
print('>> 형변환 시작')
x_train = np.asarray(train_x).astype('float32')
x_test = np.asarray(test_x).astype('float32')

y_train = np.asarray(train_y).astype('float32')
y_test = np.asarray(test_y).astype('float32')


     #################
     # Deep Learning #
     #################

# 모델 정의 및 학습하기
# Tensorflow-Keras
# - keras는 층을 조합하여 모델을 생성함
# - 차례대로 층을 쌓아 만든 모델을 tf.keras.Sqquential 모델

model = models.Sequential() # 모델 생성

# 모델 층을 구성
model.add(layers.Dense(64, activation='relu', input_shape=(5000,))) # 1층 생성
model.add(layers.Dense(64, activation='relu')) # 2층 생성
model.add(layers.Dense(1, activation='sigmoid')) # 3층 생성

# 모델 훈련 준비(훈련과정을 설정)
# 1) optimizer: 훈련과정을 설정
# 2) loss: 최적화 과정에서 최소화될 손실 함수를 설정
# 3) metrics = 훈련을 모니터링하기위해 사용
model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss=losses.binary_crossentropy,
              metrics=[metrics.binary_accuracy])

# 모델 훈련
model.fit(x_train, y_train, epochs=10, batch_size=512)

# 모델평가와 예측
results = model.evaluate(x_test, y_test)

# 학습모델 저장
# 모델 아키텍처 + 모델 가중치
print('Trained Model Saved.')
model.save('my_model.h5')

















