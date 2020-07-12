---
title: "Basic Data Scrapping"
date: 2020-07-12
tags: [data scrapping, data science, messy data]
header:
  image: "/images/perceptron/percept.jpg"
excerpt: "Data Wrangling, Data Science, Messy Data"
mathjax: "true"
---

# Scrape IMDB Ranking Data:


```python
# import libraries
import urllib.request
from bs4 import BeautifulSoup
import csv
import pandas as pd

#page url
pageurl="https://www.imdb.com/search/title/?count=100&groups=top_1000&sort=user_rating"
```

### Iterate over all the pages and capture the data until reached the last page.


```python
# Get the parent div section {use inspect element to know the attributes}
rows=[]
rows.append(['Movie_Name','Year','Rating','Description','Director','Stars','Certificate','Runtime','Genre','Votes','Gross'])
while True:
    page = urllib.request.urlopen(pageurl)

    soup=BeautifulSoup(page,'html.parser')
    parent_div=soup.find('div',attrs={'class':'article'})
    atricle_nav=parent_div.find('div',attrs={'class':'nav'}).find('div',attrs={'class':'desc'}).find('a',attrs={'class':'lister-page-next next-page'})
    #print(type(atricle_nav))
    # if NoneType then break
    #print(atricle_nav.get('href'))
    child_div_list=soup.find('div',attrs={'class':'lister-list'})

    content_list_result=child_div_list.find_all('div',attrs={'class':'lister-item mode-advanced'})
    len(content_list_result)
    for content_fetch in content_list_result:
        
        ch_list=[]
        content_div=content_fetch.find('div',attrs={'class':'lister-item-content'})
        name=content_div.find('h3').find('a').getText()
        year=content_div.find('h3').find('span',attrs={'class':'lister-item-year text-muted unbold'}).getText()
        ch_list.append(name)
        ch_list.append(year)
        rating=content_div.find('div').find('div',attrs={'class':'inline-block ratings-imdb-rating'}).get('data-value')
        ch_list.append(rating)
        description=content_div.find_all('p',attrs={'class':'text-muted'})[1].getText().strip()
        ch_list.append(description)
        dir_start_list=content_div.find_all('p')[2].getText().replace('\n','').strip().split('|')
        director=dir_start_list[0].split(':')[1].strip()
        ch_list.append(director)
        stars=dir_start_list[1].split(':')[1].strip()
        ch_list.append(stars)
        movie_feature=content_div.find('p').getText().replace('\n','').split('|')
        try:
            
            certificate=content_div.find('p').find('span',attrs={'class':"certificate"}).getText().strip()
        except:
            certificate=''
        #grade=movie_feature[0].strip()
        ch_list.append(certificate)
        runtime=content_div.find('p').find('span',attrs={'class':"runtime"}).getText()
        ch_list.append(runtime)
        try:
            genre=content_div.find('p').find('span',attrs={'class':"genre"}).getText().replace('\n','').strip()
        except:
            genre=''
        ch_list.append(genre)
        votes=content_div.find('p',attrs={'class':'sort-num_votes-visible'}).find_all('span',attrs={'name':'nv'})[0].get('data-value')
        try:
            gross=content_div.find('p',attrs={'class':'sort-num_votes-visible'}).find_all('span',attrs={'name':'nv'})[1].get('data-value')
        except:
            gross=''
        ch_list.append(votes)
        ch_list.append(gross)
        rows.append(ch_list)
    # If the page is having the next page link then loop over to next page else we are on the last webpage.
    if atricle_nav is None:break
    else:pageurl='https://www.imdb.com'+atricle_nav.get('href').strip()
```


```python
## Create csv and write rows to output file
with open('imdb-data.csv','w', newline='') as f_output:
    csv_output = csv.writer(f_output)
    csv_output.writerows(rows)
```
