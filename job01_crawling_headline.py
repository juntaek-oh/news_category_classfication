from bs4 import BeautifulSoup
import requests
import re
import pandas as pd
import datetime

category = ['Politics', 'Economic', 'Social', 'Culture', 'World', 'IT']
df_titles = pd.DataFrame()

for i in range(6):

    url = 'https://news.naver.com/section/10{}'.format(i) #for문 i값 들어가면서 100,101,102 순으로 뉴스페이지 바꿈

    resp = requests.get(url)
    # print(list(resp))
    soup = BeautifulSoup(resp.text, 'html.parser')
    # print(list(soup))

    title_tags = soup.select('.sa_text_strong')
    titles = []
    for tag in title_tags:
        titles.append(tag.text)
    print(titles)

    df_section_titles = pd.DataFrame(titles, columns=['titles'])
    df_section_titles['category'] = category[i]
    df_titles = pd.concat([df_titles, df_section_titles],
                          axis='rows', ignore_index=True)
print(df_titles.head())
df_titles.info()
print(df_titles['category'].value_counts())
df_titles.to_csv('./crawling_data/naver_headline_news_{}.csv'.format(
    datetime.datetime.now().strftime('%y%m%d')), index=False)