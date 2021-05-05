import csv
from bs4 import BeautifulSoup
import requests
import re
from urllib.parse import urljoin
from os.path import basename
import urllib.request


URL='https://www.humanprogress.org/datasets/'
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.14; rv:65.0) Gecko/20100101 Firefox/65.0"
headers = {"user-agent": USER_AGENT} # adding the user agent
page = requests.get(URL, headers=headers)
soup = BeautifulSoup(page.content, "html.parser") # use this if you want to scrap the site

# print(page)
# print(page.status_code)
# print(page.content)

pages = soup.find_all('a', class_='dataset-link')

# print(pages[0])
results = []
for i in pages:
    par = str(i)
    res = re.search("(?P<url>https?://[^\s]+)", par).group("url")
    res = res[:-2]
    res = res + "data-table/"
    results.append(res)

print(len(results))
print(results[6])

#Data scraping that goes over each HumanProgress page and download csv files.
for j in results:
    try:
        URL = j
        USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.14; rv:65.0) Gecko/20100101 Firefox/65.0"
        headers = {"user-agent": USER_AGENT}  # adding the user agent
        url = requests.get(URL, headers=headers)
        soup = BeautifulSoup(url.content, "html.parser")

        for link in soup.findAll("a"):
            current_link = link.get("href")
            if current_link.endswith('csv'):
                print('Found CSV: ' + current_link)
                print('Downloading %s' % current_link)

                URL2 = current_link
                USER_AGENT2 = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.14; rv:65.0) Gecko/20100101 Firefox/65.0"
                headers2 = {"user-agent": USER_AGENT2}  # adding the user agent
                response = requests.get(URL2, headers=headers2)

                # with open("out.csv", 'w') as f:
                #     writer = csv.writer(f)
                #     for line in response.iter_lines():
                #         writer.writerow(line.decode('utf-8').split(','))


                url_content = response.content
                nam = current_link.rsplit('/', 1)
                name = str(nam[1])
                print(name)



                csv_file = open('all_indicators/' + name, 'wb')

                csv_file.write(url_content)
                csv_file.close()
    except:
        print("error:", j)






