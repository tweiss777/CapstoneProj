import json
from bs4 import BeautifulSoup
import urllib3
import urllib
import certifi

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
            total_job_links.append(job_link)

        page +=10

    for job in total_job_links:
        print(job)
    print("Total jobs returned %s" % len(total_job_links))
    return total_job_links

# tesmethod that returns individual job
def getJD(query,location,page_limit,job_id):
    link = "https://indeed.com/company/Source-Infotech-Inc./jobs/Java-Developer-d9cc01025a737866?fccid=25d10be0b57433ae&amp;vjs=3"
    http = urllib3.PoolManager(cert_reqs = 'CERT_REQUIRED', ca_certs = certifi.where())
    response = http.request('GET', link)
    html = response.data

    soup = BeautifulSoup(html, 'html.parser')

    #return the markup for the specific jd
    print(soup.prettify())

#test method that returns the job ids
def getJobIDS(query,location,page_limit):
    page = 0
    totalJobIds = []

    while page <= page_limit:
        params = {'q':query,
                  'l':str(location),
                  'start':str(page_limit)}

        link = "https://www.indeed.com/jobs?" + urllib.parse.urlencode(params)
        http = urllib3.PoolManager(cert_reqs='CERT_REQUIRED', ca_certs=certifi.where())
        response = http.request('GET', link)
        html = response.data
        soup = BeautifulSoup(html, 'html.parser')

        #retrieve div tags that have the job ids
        divs = soup.find_all('div')


        page += 10


    return totalJobIds


def main():
    retrieve_urls('Java developer', 11590,10)


main()

