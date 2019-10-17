# modellearning.py의 설명서
# 텐서플로우, 케라스 모델 트레이닝 및 예측 등
# 함수에 관련된 설명






# 데이터셋은 Naver sentiment movie corpus를 사용했습니다. (https://github.com/e9t/nsmc/)
# 이 데이터셋은 네이버 영화의 리뷰 중 영화당 100개의 리뷰를 모아 총 200,000개의 리뷰(train: 15만, test: 5만)로 이루어져있고,
# 1~10점까지의 평점 중에서 중립적인 평점(5~8점)은 제외하고 1~4점을 긍정으로, 9~10점을 부정으로 동일한 비율로 데이터에 포함시켰습니다.

# 데이터는 id, document, label 세 개의 열로 이루어져있습니다.
# id는 리뷰의 고유한 key 값이고, document는 리뷰의 내용, label은 긍정(0)인지 부정(1)인지를 나타냅니다.
# txt로 저장된 데이터를 처리하기 알맞게 list 형식으로 받아서 사용하겠습니다.
import json
import os
import nltk
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from pprint import pprint
from konlpy.tag import Okt
import numpy as np
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics


# nltk.download('all')



# *.txt 파일에서 데이터를 불러는 메서드
def read_data(filename):
    with open(filename, 'r', encoding='UTF8') as f:
        # "r" : Opens a file for reading only.
        # "r+" : Opens a file for both readin g and writing.
        # "rb" : Opens a file for reading only in binary format.
        # "rb+" : Opens a file for both reading and writing in binary format.
        # "w" : Opens a file for writing only.

        # cp949코덱으로 인고딩 된 파일을 불러들일 때 UnicodeDecodeError:'cp949' Error발생
        # open('파일경로.txt', 'rt', encoding='UTF8') << 문제 해결가능
        data = [line.split('\t') for line in f.read().splitlines()]
        # txt 파일의 헤더(id document label)는 제외하기
        data = data[1:]
    return data

# nsmc 데이터를 불러와서 Python 변수에 담기
train_data = read_data('ratings_train.txt') # 트레이닝 데이터(훈련)
test_data = read_data('ratings_test.txt') # 테스트 데이터(실험)

# 데이터가 제대로 불러와졌는지 확인하기
print('>> 데이터 확인')
print(len(train_data))
print(len(train_data[0]))
print(len(test_data))
print(len(test_data[0]))

# (Warning) numpy 관련된 error가 발생하면 이는 numpy 버전이 달라서 생기는 문제
# Anaconda Prompt에서 pip install -U numpy로 numpy 최신버전으로 다시 설치
# 이제 데이터를 학습하기에 알맞게 처리를 해볼텐데요, KoNLPy 라이브러리를 이용해서 형태소 분석 및 품사 태깅을 하겠습니다.
# imdb 리뷰 분석 예제처럼 주어진 단어의 빈도만을 사용해서 처리해도 되지만 한국어는 영어와는 달리 띄어쓰기로 의미를 구분짓기에는 한계가 있고,
# 네이버 영화 데이터에는 맞춤법이나 띄어쓰기가 제대로 되어있지 않은 경우가 있기 때문에 정확한 분류를 위해서 KoNLPy를 이용하겠습니다.
# KoNLPy는 띄어쓰기 알고리즘과 정규화를 이용해서 맞춤법이 틀린 문장도 어느 정도 고쳐주면서 형태소 분석과 품사를 태깅해주는 여러 클래스를 제공합니다. (링크 참조)
# 그 중에서 Okt(Open Korean Text) 클래스를 이용하겠습니다.
# 먼저 Okt를 이용해서 간단한 문장을 분석해보겠습니다.

okt = Okt()
print(okt.pos(u'이 밤 그날의 반딧불을 당신의 창 가까이 보낼게요'))

# 이제 아까 불러온 데이터에 형태소 분석을 통해서 품사를 태깅해주는 작업을 하겠습니다.
# 데이터의 양이 큰 만큼 시간이 오래 걸리기 때문에 이 작업을 반복하지 않도록 한 번 태깅을 마친 후에는 json 파일로 저장하는 것을 추천합니다.
# 여기에서는 이미 태깅이 완료된 train_docs.json 파일이 존재하면 반복하지 않도록 만들었습니다.


def tokenize(doc):
    # norm은 정규화, stem은 근어로 표시하기를 나타냄
    return ['/'.join(t) for t in okt.pos(doc, norm=True, stem=True)]


if os.path.isfile('train_docs.json'):
    with open('train_docs.json', 'r', encoding='UTF8') as f:
         train_docs = json.load(f)
    with open('test_docs.json', 'r', encoding='UTF8') as f:
        test_docs = json.load(f)
else:
    train_docs = [(tokenize(row[1]), row[2]) for row in train_data]
    test_docs = [(tokenize(row[1]), row[2]) for row in test_data]
    # JSON 파일로 저장
    with open('train_docs.json', 'w', encoding="utf-8") as make_file:
        json.dump(train_docs, make_file, ensure_ascii=False, indent="\t")
    with open('test_docs.json', 'w', encoding="utf-8") as make_file:
        json.dump(test_docs, make_file, ensure_ascii=False, indent="\t")


# 예쁘게(?) 출력하기 위해서 pprint 라이브러리 사용
pprint(train_docs[0])
pprint(test_docs[0])

# 분석한 데이터의 토큰(문자열을 분석을 위한 작은 단위)의 갯수를 확인해봅시다.
tokens = [t for d in train_docs for t in d[0]]
print(len(tokens))

# 이제 이 데이터를 nltk 라이브러리를 통해서 전처리를 해볼텐데요, Text 클래스는 문서를 편리하게 탐색할 수 있는 다양한 기능을 제공합니다.
# 여기에서는 vocab().most_common 메서드를 이용해서 데이터에서 가장 자주 사용되는 단어를 가져올 때 사용하겠습니다.


# (Warling) _sqlite3 DLL Import Error 가 발생하면
# https://www.sqlite.org/download.html 접속해서 해당 OS에 맞는 파일을 다운로드후
# C:\Users\PC\Anaconda3\envs\bigdata\DLLs PC의 anaconda evn의 Dlls 폴더에 넣어주고 실행하면 문제 해결 됨
text = nltk.Text(tokens, name='NMSC')

# 전체 토큰의 개수
print(len(text.tokens))

# 중복을 제외한 토큰의 개수
print(len(set(text.tokens)))

# 출현 빈도가 높은 상위 토큰 10개
pprint(text.vocab().most_common(10))

# 자주 나오는 단어 50개를 matplotlib 라이브러리를 통해서 그래프로 나타내보겠습니다.
# 한편 한글 폰트를 로드해야 글씨가 깨지지 않고 출력이 되는데요,
# 윈도우에서는 font_fname 을 'c:/windows/fonts/gulim.ttc',
# 리눅스에서는 /usr/share/fonts/nanumfont/NanumGothic.ttf 등 한글 폰트를 지정해줘야 합니다.

# (Warling) import ft2font Error가 발생한다면
# Anaconda Prompt에서 pip install matplotlib --force-reinstall로 matplotlib 재설치 후 실행
font_fname = 'C:\Windows\Fonts\gulim.ttc'
font_name = font_manager.FontProperties(fname=font_fname).get_name()
rc('font', family=font_name)

plt.figure(figsize=(20,10))
text.plot(50)
plt.show()

# 이제 자주 사용되는 토큰 10000개를 사용해서 데이터를 벡터화를 시키겠습니다.
# 여기서는 원 핫 인코딩 대신에 CountVectorization을 사용했습니다.
# 이는 문서 집합에서 단어 토큰을 생성하고 각 단어의 수를 세어 BOW(Bag of Words) 인코딩한 벡터를 만드는 역할을 합니다.

# 시간이 꽤 걸립니다! 시간을 절약하고 싶으면 most_common의 매개변수를 줄여보세요.
selected_words = [f[0] for f in text.vocab().most_common(5000)] # 10000
print('type:', type(selected_words))
print('len:', len(selected_words))
print('data:', selected_words[:10])


f = open('selectword.txt', 'w', encoding='UTF8')
print('>> List 파일 저장 시작')
for i in selected_words:
    i += '\n'
    f.write(i)
f.close()
print('>> List 파일 저장 완료')

def term_frequency(doc):
    return [doc.count(word) for word in selected_words]


train_x = [term_frequency(d) for d, _ in train_docs]
test_x = [term_frequency(d) for d, _ in test_docs]
train_y = [c for _, c in train_docs]
test_y = [c for _, c in test_docs]

# 이제 데이터를 float로 형 변환 시켜주면 데이터 전처리 과정은 끝납니다.
print('>> 형변환 시작')
x_train = np.asarray(train_x).astype('float32')
x_test = np.asarray(test_x).astype('float32')

y_train = np.asarray(train_y).astype('float32')
y_test = np.asarray(test_y).astype('float32')





# 모델 정의 및 학습하기
# 두 개의 Dense 층은 64개의 유닛을 가지고 활성화 함수로는 relu를 사용했으며, 마지막 층은 sigmoid 활성화 함수를 사용해서 긍정의 리뷰일 확률을 출력합니다.
# 손실 함수로는 binary_crossentropy를 사용했고 RMSProp 옵티마이저를 통해서 경사하강법을 진행했습니다.
# 또한 배치 사이즈를 512로, 에포크를 10번으로 학습시켰습니다.

# 층 설정하기
#  - 케라스에서는 층을 조합하여 모델을 생성함.
#  - 가장 흔한 모델은 층을 차례대로 쌓은 tf.keras.Sequential 모델

# activation = 층의 활성화 함수를 설정(기본값: 활성화함수 적지 않음)
# kernel_initializer, bias_initializer = 층의 가중치(커널(kernel)과 절편(bias))을 초기화하는 방법(기본값: glorot_uniform)
# kernel_regularizer, bias_regularizer = L1 또는 L2 규제와 같이 층의 가중치(커널과 절편)에 적용할 규제 방법을 지정(기본값: 규제를 적용하지 않음)
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(5000,))) # 10000
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


# 모델훈련 준비
# tf.keras.Model.compile()에 사용되는 3개의 매개변수
# 1) optimizer = 훈련과정을 설정(Adam, SGD 등을 사용)
# 2) loss = 최적화 과정에서 최소화될 손실 함수(loss function)를 설정
#   - 평균제곱 오차(mse), categorical_crossentropy, binary_crossentropy 등이 자주 사용
# 3) metrics = 훈련을 모니터링하기위해 사용
# +) 추가적으로 훈련과 평가를 즉시 실행할 때는 run_eagerly=True 매개변수를 전달 가능
model.compile(optimizer=optimizers.RMSprop(lr=0.001),
             loss=losses.binary_crossentropy,
             metrics=[metrics.binary_accuracy])

# 모델훈련
# tf.keras.Model.fit()에 사용되는 매개변수들
# 1) epochs = 훈련은 에포크로 구성, 한 에포크는 전체 입력데이터를 한번 순회하는 것(작은 배치로 나누어 수행)
# 2) batch_size = 넘파이 데이터를 전달하면 모델은 데이터를 작은 배치로 나누고 훈련 과정에서 이 배치를 순회함.
#                 이정수값은 배치의 크기를 지정, 전체 샘플 개수가 배치 크기로 나누어 떨어지지 않으면 마지막 배치의 크기는 더 작을 수 있음.
# 3) validation_data = 모델의 프로토타입을 만들 때 검증데이터에서 간편하게 성능을 모니터링 해야함. 입력과 레이블의 튜플을
#                      이 매개변수로 전달하면 에포크가 끝날 때마다 추론 모드에서 전달된 데이터의 손실과 측정 지표를 출력함
model.fit(x_train, y_train, epochs=10, batch_size=512)

# 모델평가와 예측
results = model.evaluate(x_test, y_test)

# 학습모델 저장
# 모델은 크게 모델 아키텍처와 모델 가중치로 구성됩니다.
# 기본적으로 모델의 가중치는 텐서플로 체크포인트 파일 포맷으로 저장됩니다.
# 케라스의 HDF5 포맷으로 가중치를 저장할 수도 있습니다(다양한 백엔드를 지원하는 케라스 구현에서는 HDF5가 기본 설정입니다):

print('Trained Model Saved.')
model.save('my_model.h5')


def predict_pos_neg(review):
    token = tokenize(review)
    tf = term_frequency(token)
    data = np.expand_dims(np.asarray(tf).astype('float32'), axis=0)
    score = float(model.predict(data))
    if(score > 0.5):
        print("[{}]는 {:.2f}% 확률로 긍정 리뷰이지 않을까 추측해봅니다.^^\n".format(review, score * 100))
    else:
        print("[{}]는 {:.2f}% 확률로 부정 리뷰이지 않을까 추측해봅니다.^^;\n".format(review, (1 - score) * 100))

predict_pos_neg("올해 최고의 영화! 세 번 넘게 봐도 질리지가 않네요.")
predict_pos_neg("배경 음악이 영화의 분위기랑 너무 안 맞았습니다. 몰입에 방해가 됩니다.")

# 가중치를 HDF5 파일로 저장합니다.
#model.save_weights('my_model.h5', save_format='h5')

# 모델의 상태를 복원합니다.
# model.load_weights('my_model.h5')

# 설정
# 모델 설정을 저장하면 가중치는 제외하고 모델의 구조를 직렬화합니다.
# 원본 모델을 정의한 코드가 없어도 저장된 설정을 사용하여 동일한 구조를 만들고 초기화할 수 있습니다.
# 케라스는 JSON과 YAML 직렬화 포맷을 지원합니다.

# 모델을 JSON 포맷으로 직렬화합니다.
# json_string = model.to_json()
# json_string

# import json
# import pprint
# pprint.pprint(json.loads(json_string))




# 전체모델 저장
# 간단한 모델을 만듭니다.
# model = tf.keras.Sequential([
#   layers.Dense(10, activation='softmax', input_shape=(32,)),
#   layers.Dense(10, activation='softmax')
# ])
# model.compile(optimizer='rmsprop',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
# model.fit(data, labels, batch_size=32, epochs=5)
#
#
# # 전체 모델을 HDF5 파일로 저장합니다.
# model.save('my_model.h5')
#
# # 가중치와 옵티마이저를 포함하여 정확히 같은 모델을 다시 만듭니다.
# model = tf.keras.models.load_model('my_model.h5')