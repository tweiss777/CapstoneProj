from bs4 import BeautifulSoup
import urllib
import urllib3
import certifi
import json



def retrieveData(query,numJobs):
    apikey = 'e4309ca9a7136eca715e51b52947b62d'
    base_url = 'https://authenticjobs.com/api/?'
    method = 'aj.jobs.search'
    params = {
        'api_key': apikey,
        'method': method,
        'keywords':query.replace(" ",","),
        'perpage': str(numJobs),
        'format': 'json'
    }

    final_url = base_url + urllib.parse.urlencode(params)
    http = urllib3.PoolManager(cert_reqs = 'CERT_REQUIRED', ca_certs = certifi.where())
    response = http.request('GET', final_url)
    data = response.data
    print(data)
    return data






def main():
    json_data = retrieveData('python', 1)
main()