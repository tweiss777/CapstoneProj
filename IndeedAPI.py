import json
from bs4 import BeautifulSoup
import urllib



# test method that returns a list of jobs
def test1():
    # link = "https://www.indeed.com/jobs?q=java+developer&l=11590&start=0"
    link = 'https://www.indeed.com/jobs?q=python&l=11590&start=0&limit=10'
    with urllib.request.urlopen(link) as response:
        html = response.read()


    soup = BeautifulSoup(html, 'html.parser')

    job_links = [a_tags for a_tags in soup.find_all('a', class_='turnstileLink', href=True)]
    for jl in job_links:
        print(jl)

    print("the number of jobs returned %s" % len(job_links))

# tesmethod that returns individual job
def test2():
    link = "https://indeed.com/company/Source-Infotech-Inc./jobs/Java-Developer-d9cc01025a737866?fccid=25d10be0b57433ae&amp;vjs=3"
    with urllib.request.urlopen(link) as response:
        html = response.read()
    soup = BeautifulSoup(html,'html.parser')

    #return the markup for the specific jd
    print(soup.prettify())




def main():
    test2()

main()

