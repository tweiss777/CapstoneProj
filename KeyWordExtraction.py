# Test file that extracts

from IndeedAPI import *
indeedApi = IndeedAPi()



job_urls = indeedApi.retrieve_urls("Java developer",11590,10)

unfiltered_jobs = [indeedApi.getJob(url) for url in job_urls]
filtered_jobs = indeedApi.filterJobs(unfiltered_jobs)


job_keywords = []

for title,job in filtered_jobs:
    job_keywords.append((title,set(job.split(" "))))