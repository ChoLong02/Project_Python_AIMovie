from pymongo import MongoClient
# MongoDB에 계정이 있거나 외부 IP인 경우
# DB_HOST = 'xxx.xx.xx.xxx:27017'
# DB_ID = 'root'
# DB_PW = 'pw'
# client = MongoClient('mongodb://%s:%s@%s' % (DB_ID, DB_PW, DB_HOST))

class MongoDAO:
    reply_list = []  # MongoDB Document를 담을 List

    def __init__(self):
        # >> MongoDB Connection
        self.client = MongoClient('localhost', 27017) # 클래스 객체 할당(ip주소, port번호)
        self.db = self.client['local']  # MongoDB의 'local' DB를 할당
        # self.collection = self.db.movie
        self.collection = self.db.get_collection('movie')  # 동적으로 Collection 선택

    def mongo_write(self, data):
        print('>> MongoDB write data!')
        self.collection.insert(data) # JSON Type = Dict Type(python)

    def mongo_update(self, data):
        pass

    def mongo_selectAll(self):
        for one in self.collection.find({}, {'_id': 0, 'title':1, 'content': 1, 'score': 1}):  # 제목, 내용, 평점만 DB에서 조회
            self.reply_list.append([one['title'], one['content'], one['score']])  # dict에서 Value와 Score만 추출
        return self.reply_list

    def mongo_view(self, data):
        pass

    def mongo_delete(self, data):
        pass