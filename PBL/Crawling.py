#####Day_1#####
import urllib.request
import requests
from bs4 import BeautifulSoup


url = 'https://www.naver.com'
html = urllib.request.urlopen(url)
html.read()

url = 'path.image.jpg'
urllib.request.urlretrieve(url, 'save_name')

url = 'https://www.naver.com'
html = urllib.request.urlopen(url)
soup = BeautifulSoup(html, 'html.parser')

soup.html.body.h1
soup.html.body.p

soup.find('ul')
soup.find('ul', {'class':'reply'})
soup.find('ul', {'class':'reply'}).findAll('li')

soup.findAll('ul')
soup.findAll('ul', {'class':'title'})

#string: 태그에 포함된 단어가 한개만 있을시 출력 (enter, tap 포함)
#text: 태그에 포함된 단어 전부 출력 (enter, tap 포함)
for item in soup.find('ul', {'class':'list_nav type_fix'}).findAll('li'):
    print(item.string)
for item in soup.find('ul', {'class':'list_nav type_fix'}).findAll('li'):
    print(item.text)

#strip(): enter, tap 제거
for item in soup.findAll('div', {'class':'tit3'}):
    print(item.text.strip())


#####Day_2#####
import requests

#가져오기 성공 200 리턴
#가져오기 실패 404(URL오타), 500(서버오류)
response = requests.get('http://www.naver.com')

#response
#response(성공여부, content)
response
response.content
response.text

soup = BeautifulSoup(response.text, 'html.parser')

#csv파일 생성 후 데이터 입력
#file.write() 데이터 입력
#효율성을 위해 format선호
with open('weather.csv', 'w') as file:
    print('파일 저장')
    file.write('point, temp, hum\n')
    for i in lst:
        file.write('{0}, {1}, {2}\n'.format(i[0], i[1], i[2]))


#Selenium
pip install selenium
from selenium import webdriver as wd


url = 'http://tour.interpark.com'
searchBox_id = 'SearchGNBText'
keyword = '스위스'
searchBtn_class = 'search-btn'
infoBtn_id = 'li_R'

#Chrome실행, url접속
driver = wd.Chrome(executable_path='chromedriver.exe')
driver.get(url)

#url 접속 시간 확보
driver.implicitly_wait(10)

#원하는 페이지 접속
driver.find_element_by_id(searchBox_id).send_keys(keyword)
driver.find_element_by_class_name(searchBtn_class).click()
driver.find_element_by_id(infoBtn_id).click()

#페이지를 바꿔가며 스크래핑
#time.sleep()으로 스크래핑 시간 확보
for page in range(1,8):
    print('########### {0} page ###########'.format(page))
    driver.execute_script(
    'searchModule.SetCategoryList({}, '')'.format(page))
    time.sleep(2)     
    box_items = driver.find_elements_by_css_selector('.boxList>li')
    for li in box_items:
        product = li.find_element_by_css_selector('h5.proTit').text
        price = li.find_element_by_css_selector('.proPrice').text.split('원')[0]
        print(product, price)


#####Day_3#####
import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup

url='https://www.youtube.com/channel/UCyn-K7rZLXjGl7VXGweIlcA/videos'
SCROLL_PAUSE_TIME=0.5
now_height = 0

driver = webdriver.Chrome('chromedriver.exe')
driver.get(url)

body = driver.find_element_by_tag_name('body')
while True:
    #javascript function
    last_height = driver.execute_script('return document.documentElement.scrollHeight')
    #스크롤
    body.send_keys(Keys.END)
    time.sleep(3)
    if now_height == last_height: break
    else: now_height = last_height

page = driver.page_source
soup = BeautifulSoup(page, 'html.parser')
all_videos = soup.find_all(id='dismissible')


#title_lst생성
title_lst = []
for video in all_videos:
    title = video.find(id='video-title')
    title_lst.append(title.text)

#video_lst생성
video_time_lst = []
for video in all_videos:
    video_time = video.find('span', {'class':'style-scope ytd-thumbnail-overlay-time-status-renderer'})
    split_time = video_time.text.strip().split(':')
    total_time = int(split_time[0]) * 60 + int(split_time[1])
    video_time_lst.append(total_time)

#view_lst생성
view_num_lst = []
for video in all_videos:
    view_num = video.find('span', {'class':'style-scope ytd-grid-video-renderer'})
    split_view = view_num.text.split(' ')
    split = split_view[1]
    if split[-2:] == '만회': view = float(split[:-2]) * 10000
    else: view = float(split[:-2]) * 1000
    view_num_lst.append(int(view))

#youtube_df생성
youtube_dict = {'title':title_lst, 
               'video_time':video_time_lst, 
               'view_num':view_num_lst}
youtube_df = pd.DataFrame(youtube_dict)