# Test file that extracts
import re

from docx import Document
from nltk.corpus import stopwords

from IndeedAPI import *

#data needs to be exported to json.


#Initialize the necessary objects
indeedApi = IndeedAPi()
# rfunctions = RakeFunctions()


job_urls = indeedApi.retrieve_urls("Java developer",11590,10)

#this list holds jobs that may have no content.
unfiltered_jobs = [indeedApi.getJobMarkup(url) for url in job_urls]

# filtered jobs are jobs in list that have content Note: this holds the markup (html) of the job
filtered_jobs = indeedApi.filterJobs(unfiltered_jobs)

#takes the filtered jobs and creates a dictionary that holds all the jobs
# key = {job_id: {title: "string", description: "string", keywords: ["String"]}}
json_data = indeedApi.generateJson(filtered_jobs)

# extract text from documents
file = open("TalWeissResume.docx", 'rb')
d = Document(file)
fullText = []

for p in d.paragraphs:
    fullText.append(p.text)

resumetext = ""  # empty string which should hold the text of the resume
# iterate through the fullText array and strip \n and \t
for i in range(len(fullText)):
    if len(fullText[i] > 0):
        fullText[i] = re.sub("\s+", " ", fullText[i])  # if there is text remove the extra space and the and the tabs
        resumetext = resumetext + fullText[i] + " "  # concatinate part of the resume text to the job description
    else:
        fullText.pop(i)  # no text? remove it from the list

resumeNoStopWords = [word for word in resumetext.split() if word not in stopwords.words('english')]

# iterate through json_data job title and description and strip the stopwords from the description
jobsNoStopWords = {}  # job_id: {title: "string", description:
for i in range(len(json_data)):

# now we will run tf-idf on both the job descriptions and the resume & strip the stopwords
