import json
from bs4 import BeautifulSoup
import urllib3
import urllib.request
import certifi
import time

# Class for indeed api
class IndeedAPi:

    # limit should be in increments of 10
    def retrieve_urls(self,query,zipcode,page_limit):

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

        return total_job_links

    # test method that returns individual job description
    def getJob(self,href):

        #site url contatinated with the href taken from the dom
        link = "https://indeed.com" + str(href)

        # retrieve the url even if its a redirect
        initial_response = urllib.request.urlopen(link)

        # final link of the redirect if there is one
        finalLink = initial_response.geturl()

        # initialize pool manager and request the html data from url
        http = urllib3.PoolManager(cert_reqs = 'CERT_REQUIRED', ca_certs = certifi.where())
        response = http.request('GET', finalLink)
        html = response.data

        soup = BeautifulSoup(html, 'html.parser')
        job_title = soup.title.string
        job_title = job_title.split('-')[0].rstrip()


        #find the tag that holds the description.
        jdMarkup = soup.find_all('div',class_='jobsearch-JobComponent-description')

        # when the job description cannot be found
        if len(jdMarkup) < 1:
            return

        return(job_title, jdMarkup[0].get_text())

    #method that filters out NoneType indices in the jobs list
    def filterJobs(self,jobs):
        filtered_jobs = [job for job in jobs if job is not None]
        return filtered_jobs


