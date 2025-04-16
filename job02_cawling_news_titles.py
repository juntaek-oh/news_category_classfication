from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions
from webdriver_manager.chrome import ChromeDriverManager
import time
import pandas as pd
import datetime

category = ['Politiccs', 'Economic', 'Social', 'Culture', 'World', 'IT']
df_titles = pd.DataFrame()


options = ChromeOptions()

options.add_argument('lang=ko_KR')

service = ChromeService(executable_path=ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)

url = 'https://news.naver.com/section/100' # 100 수정
driver.get(url)
titles = []
button_xpath = '//*[@id="newsct"]/div[4]/div/div[2]' # div[4]/div/div[2] 수정
for i in range(5):
    time.sleep(0.5)
    try:
        driver.find_element(By.XPATH, button_xpath).click()
    except:
        print(i)
time.sleep(5)

for i in range(1, 5):
    for j in range(1, 7):
        title_path = '//*[@id="newsct"]/div[4]/div/div[1]/div[{}]/ul/li[{}]/div/div/div[2]/a/strong'.format(i, j)  # div[4]/div/div[1]/ 수정
        try:
            title = driver.find_element(By.XPATH, title_path).text
            titles.append(title)
            print(title)
        except:
            print('error', i, j)

df_section_titles = pd.DataFrame(titles, columns=['titles'])
df_section_titles['category'] = category[0]     #category[0] 수정
df_titles = pd.concat([df_titles, df_section_titles],
                      axis='rows', ignore_index=True)
df_titles.to_csv('./crawling_data/naver_headline_Politiccs_{}.csv'.format(
    datetime.datetime.now().strftime('%y%m%d')), index=False) #Politiccs수정
