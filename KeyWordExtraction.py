# Test file that extracts
import re

import nltk
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

# Iterate through the array removing indices that have length of 0
for i in range(len(fullText)):
    try:
        if len(fullText[i]) == 0:
            fullText.pop(i)
    except IndexError:
        print("Could not pop indice...")
# iterate through the fullText array and strip \n and \t
for i in range(len(fullText)):
    fullText[i] = re.sub("\s+", " ", fullText[i])  # if there is text remove the extra space and the and the tabs
    resumetext = resumetext + fullText[i] + " "  # concatinate part of the resume text to the job description

resumeNoStopWords = [word for word in resumetext.split() if word not in stopwords.words('english')]

# iterate through the words in the resume and remove the terms that are less than 2 and contain any special characters
for i in range(len(resumeNoStopWords)):
    try:
        if len(resumeNoStopWords[i]) < 2 and resumeNoStopWords[i][0] in '()~><?'';":\/|{},[]@6&*-_':
            resumeNoStopWords.pop(i)
    except IndexError:
        print("Could not pop indice...")

# iterate through json_data job title and description and strip the stopwords from the description
jobsNoStopWords = {}  # job_id: {title: "string", descriptionNoStopwords: []
# job_id = 0
for i in range(len(json_data)):
    # Create an empty dictionary
    jobsNoStopWords[i] = {}

    # Store title from json_data
    jobsNoStopWords[i]["title"] = json_data[i]["title"]

    # Store the words with the stopwords stripped out represented as a list
    jobsNoStopWords[i]["description"] = [w for w in json_data[i]["description"].split() if
                                         w not in stopwords.words('english')]

# retrieve parts of speech from each term in the resume, keeping only nouns verbs and adjectives
resumePOSTags = nltk.pos_tag(resumeNoStopWords)
POSToKeep = ["JJ", "JJR", "JJS", "NN", "NNS", "NNP", "NNPS", "VB", "VBD", "VBG", "VBN", "VBP",
             "VBZ"]  # These are the nouns, verbs, and adjectives we want to keep
punctuation = '~><?'';:\/|{}[]@6&*-_,.'
# iterate through the part of speech tags and strip out the conjunction and prepositions.
# keep the verbs, nouns, and adjectives
for w in resumePOSTags:
    if w[1] not in POSToKeep:
        resumePOSTags.remove(w)

resumeNoStopWordsUpdated = [w[0] for w in resumePOSTags]

# snippet that strips punctuation from the words in the resume.
for word in resumeNoStopWordsUpdated:
    if word[-1] in punctuation:
        word = word.replace(word[-1], "")

# they layout of this dictionary will be the same as jobsNoStopWords
# this has certain parts of speech eliminated
jobsNoStopWordsUpdated = {}

# do the same for the jobs
# something is wrong with this loop (its stripping words we don't want but not all of them)
for i in range(len(jobsNoStopWords)):

    jobsNoStopWordsUpdated[i] = {}
    jobsNoStopWordsUpdated[i]["title"] = jobsNoStopWords[i]["title"]

    # store the job part of speech
    jobPOS = nltk.pos_tag(jobsNoStopWords[i]["description"])
    for w in jobPOS:
        if w[1] not in POSToKeep:
            print("found %s" % w[1])
            jobPOS.remove(w)
    jobsNoStopWordsUpdated[i]["description"] = [w[0] for w in jobPOS]

# Remove special punctuation from the last indice in the jobsNoStopWordsUpdated dictionary description key
for i in range(len(jobsNoStopWordsUpdated)):
    for j in range(len(jobsNoStopWordsUpdated[i]["description"])):
        if jobsNoStopWordsUpdated[i]["description"][j][-1] in punctuation:
            jobsNoStopWordsUpdated[i]["description"][j] = jobsNoStopWordsUpdated[i]["description"][j].replace(
                jobsNoStopWordsUpdated[i]["description"][j][-1], "")

# the query will be the job description
# now we will run tf-idf on both the job descriptions and the resume & strip the stopwords
# for now assume the resume will be the query as i have currently one resume and multiple job descriptions
