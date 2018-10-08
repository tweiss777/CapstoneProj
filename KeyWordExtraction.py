# Test file that extracts

from IndeedAPI import *
import json



indeedApi = IndeedAPi()
job_urls = indeedApi.retrieve_urls("Java developer",11590,10)

#this list holds jobs that may have no content.
unfiltered_jobs = [indeedApi.getJobMarkup(url) for url in job_urls]

#filtered jobs are jobs in list that have content
filtered_jobs = indeedApi.filterJobs(unfiltered_jobs)

json_data = indeedApi.generateJson(filtered_jobs)