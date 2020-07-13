---
title: "Understanding Data Scrapping"
date: 2020-07-12
tags: []
header:
  image: "/images/web-scrapping.jpg"
excerpt: "Data Scrapping, BeautifulSoup, Selenium"
mathjax: "true"
---

# Scrape IMDB Ranking Data:

Extracting the data from a webpage as per the business needs is web-scrapping.A basic intution would be picking up few balls from a bucket which we need for our purpose.
As we know that a webpage is nothing but a bunch of html tags wrapped around the useful information that could be rendered through a web browser. In web-scrapping we are concerned about the data inside the tags and the whole process would be ignoring the html tags and extracting the information.
BeautifulSoup is the most popular library in python for web-scrapping but with some limitations which we talk later as we proceed.

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

Get the parent div section {use inspect element to know the attributes} on the browser through the developer tools.
In this example we have total 10 webpages each page of 100 records but url of the first page is only provided.
We will implement  --> If the page is having the next page link then loop over to next page else we are on the last webpage.
```python

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

Finally once we extract the data and stored into our list then we will write into a csv file for further use and analysis.

```python
## Create csv and write rows to output file
with open('imdb-data.csv','w', newline='') as f_output:
    csv_output = csv.writer(f_output)
    csv_output.writerows(rows)
```
Remember we talk about limitation of BeautifulSoup ? Yes, there is a limitation. Suppose you have a webpage where the contents are not static and the data is loaded dynamically through a js function. In this case your script cannot parse the dynamic content as it doesnot have js engine.
Therefore in these type of cases we must use 'Selenium' library to overcome this problem. As Selenium will be able to create a browser object and load the dynamic content and then we can parse it to start our extraction process.