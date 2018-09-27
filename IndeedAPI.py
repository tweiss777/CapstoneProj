import json
from bs4 import BeautifulSoup
import urllib3
import urllib
import certifi
import time
# vjk=66c82be83c389f01
# test method that returns a list of jobs
# limit should be in increments of 10
def retrieve_urls(query,zipcode,page_limit):

    page = 0
    total_job_links = []
    while page <= page_limit:
        params = {
            'q' : query,
            'l' : str(zipcode),
            'start': str(page)
        }

        link = "https://www.indeed.com/jobs?" + urllib.parse.urlencode(params)
        http = urllib3.PoolManager(cert_reqs = 'CERT_REQUIRED', ca_certs = certifi.where())
        response = http.request('GET', link)
        html = response.data
        soup = BeautifulSoup(html, 'html.parser')

        job_links = [a_tags for a_tags in soup.find_all('a', class_='turnstileLink', href=True)]
        for job_link in job_links:
            total_job_links.append(job_link['href'])

        page +=10

    for job in total_job_links:
        print(job)
    print("Total job links returned %s" % len(total_job_links))
    return total_job_links

# tesmethod that returns individual job description
def getJob(href):
    # link = "https://indeed.com/rc/clk?jk=d2dcaad5873c0ccd&amp;fccid=cad0c703787d24ae&amp;vjs=3"

    #site url contatinated with the href taken from the dom
    link = "https://indeed.com" + str(href)

    # retrieve the url even if its a redirect
    initial_response = urllib.request.urlopen(link)

    # final link of the redirect if there is one
    finalLink = initial_response.geturl()

    http = urllib3.PoolManager(cert_reqs = 'CERT_REQUIRED', ca_certs = certifi.where())
    response = http.request('GET', finalLink)
    html = response.data

    soup = BeautifulSoup(html, 'html.parser')
    job_title = soup.title.string
    job_title = job_title.split('-')[0].rstrip()
    #return the markup for the specific jd
    # print(soup.prettify())

    #find the tag that holds the description.
    jdMarkup = soup.find_all('div',class_='jobsearch-JobComponent-description')
    print(job_title)

    if len(jdMarkup) < 1:
        return

    return(job_title, jdMarkup[0].get_text())


#test method that returns the job ids


def main():

    hrefs = retrieve_urls('Java developer', 11590,10)
    time.sleep(5)
    for i in range(0,10,1):
        getJob(hrefs[i])
main()

