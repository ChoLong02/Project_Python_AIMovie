from bs4 import BeautifulSoup
from selenium import webdriver
import persistence.MongoDAO as DAO
import requests

class DaumCrawler:
    def __init__(self):
        self.mDao = DAO.MongoDAO() # 객체생성

    def crawler(self, code):

        # 페이지 존재유무 체크
        doc = requests.get('https://movie.daum.net/moviedb/main?movieId={}'.format(code))
        if doc.status_code != 200: # 200(success)
            print('>> Not Found Page:/')
            return

        print('>> Start Crawling!')
        count = 0
        page = 1

        path = 'E:\Bigdata\webdriver\chromedriver.exe'
        driver = webdriver.Chrome(path)

        while True:
            url = 'https://movie.daum.net/moviedb/grade?movieId={}&type=netizen&page={}'.format(code, page)

            # selenium 설정
            driver.get(url)  # http://까지 적어야함

            soup = BeautifulSoup(driver.page_source, 'html.parser')
            reply_list = soup.select('div.review_info')
            title = soup.select('h2.tit_rel')[0].text
            if len(reply_list) == 0:
                break

            for reply in reply_list:
                score = reply.select('em.emph_grade')[0].text
                writer = reply.select('em.link_profile')[0].text
                content = reply.select('p.desc_review')[0].text.strip()
                reg_date = reply.select('span.info_append')[0].text.strip()[:10]

                data = {'title': title,
                        'score': int(score), # 평점계산을 위해 정수형으로 변환
                        'writer': writer,
                        'content': content,
                        'reg_date': reg_date}

                # MongoDB에 댓글 저장
                self.mDao.mongo_write(data)

                print('▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒')
                print('제목:', title)
                print('평점:', score)
                print('작성자:', writer)
                print('댓글:', content)
                print('작성일자', reg_date)
                count += 1
            page += 1
        driver.close() # driver 자원 반납
        print('▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒')
        print('수집한 게시글 수는 {}건입니다.'.format(count))
