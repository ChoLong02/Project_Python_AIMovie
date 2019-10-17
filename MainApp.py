import collector.DaumCrawler as collector
import deeplearning.modeluse as model

code = '125049' # 다음 영화코드 입력

# 다음영화 수집 Start!!!

# __name__ == '__main__'을 사용하는 이유
# MainApp을 실행하지 않고 import만 했는데도 실행되는 경우가 있음
# __name__ == '__main__'을 사용하면 실제 실행했을 때만 조건이 True가 되어 실행되고
# 인터프리터나 다른 파일에서 이 모듈을 불러서 사용하면 False가 되어 실행이 되지 않는다

# __name__는 파이썬이 내부적으로 사용하는 특별한 변수 이름
# C:\doit>python mod1.py처럼 직업 mod1.py를 실행한 경우
# mod1.py의 __name__변수에는 __main__값이 저장됨
if __name__ == '__main__':
    # 크롤링 및 MongoDB에 저장
    try:
        scrap = collector.DaumCrawler() # 객체 생성
        scrap.crawler(code)
    except Exception as e:
        print('>> Exception :(')
        print('>>', e)
    finally:
        pass

    # MongoDB에서 데이터 불러온 후 트레이닝된 AI로 감성분석(예측)
    pr = model.PredictReview()
    pr.predict()



