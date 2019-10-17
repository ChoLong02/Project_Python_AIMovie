import tensorflow as tf
import numpy as np
from konlpy.tag import Okt
from pymongo import MongoClient
from pprint import pprint
import persistence.MongoDAO as DAO

class PredictReview:
    reply_list = []              # MongoDB Document를 담을 List
    selected_words = []          # selectword.txt를 담을 List
    model = None                 # 트레이닝 된 모델을 담을 변수
    okt = Okt()                  # Okt() 형태소 분석기 객체 생성
    mDao = DAO.MongoDAO()        # MongoDAO 객체생성
    all_count = 0  # 전체갯수
    pos_count = 0                # 긍정갯수

    def __init__(self):
        try:
            PredictReview.reply_list = PredictReview.mDao.mongo_selectAll()
            print('>> Success: MongoDB Data Open Complete:)')
            PredictReview.all_count = len(PredictReview.reply_list)
            # print(len(PredictReview.reply_list))
        except Exception as e:
            print('>> Error: MongoDB Data Open Fail:(')
            print(e)
        finally:
            pass
        PredictReview.selected_words = self.read_data('C:\\Users\master\PycharmProjects\movieday\deeplearning\selectword.txt')  # 트레이닝 데이터(훈련)
        # print(self.selected_words[:10])

        # 트레이닝된 모델(AI) 불러오기
        PredictReview.model = tf.keras.models.load_model('C:\\Users\master\PycharmProjects\movieday\deeplearning\my_model.h5')
        # print('model(type):', type(PredictReview.model))

    # 데이터 전처리에 필요한 SelectWords 데이터를 불러오는 메서드
    def read_data(self, filename):
        words_data = []
        with open(filename, 'r', encoding='UTF8') as f:
            while True:
                line = f.readline()[:-1]
                if not line: break
                words_data.append(line)
        return words_data

    # 예측할 데이터의 전처리를 진행할 메서드
    def tokenize(self, doc):
        # norm은 정규화, stem은 근어로 표시하기를 나타냄
        return ['/'.join(t) for t in self.okt.pos(doc, norm=True, stem=True)]

    # 예측할 데이터의 벡터화를 진행할 메서드(임베딩)
    def term_frequency(self, doc):
        return [doc.count(word) for word in PredictReview.selected_words]

    # 모델로 예측하는 메서드 구현
    def predict_pos_neg(self, review):
        token = self.tokenize(review)
        tf = self.term_frequency(token)
        data = np.expand_dims(np.asarray(tf).astype('float32'), axis=0)
        score = float(PredictReview.model.predict(data))
        if (score > 0.5):
            PredictReview.pos_count += 1
            print("[{}]는 {:.2f}% 확률로 긍정 리뷰이지 않을까 추측해봅니다.^^\n".format(review, score * 100))
        else:
            print("[{}]는 {:.2f}% 확률로 부정 리뷰이지 않을까 추측해봅니다.^^;\n".format(review, (1 - score) * 100))

    # 8. 예측시작
    def predict(self):
        for one in PredictReview.reply_list:
            self.predict_pos_neg(one[1])

        aCount = PredictReview.all_count
        pCount = PredictReview.pos_count
        pos_pct = (pCount*100)/aCount
        neg_pct = 100-pos_pct
        # print(aCount, pCount, pos_pct, neg_pct)
        print('▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒')
        print('▒▒({}) 댓글 {}개를 감성분석한 결과'.format(PredictReview.reply_list[0][0], aCount))
        print('▒▒긍정적인 의견{:.2f}% / 부정적인 의견{:.2f}% '.format(pos_pct, neg_pct))
        print('▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒')
