# Import statements
import re

import nltk
import numpy as np
from docx import Document
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from IndeedAPI import *


class DataProcessor:
    # function to get paragraphs / sections from word document
    def gen_paragraph(self, document):
        # holds the document paragraphs
        resume = [line.text for line in document.paragraphs]

        # True/False array
        isEmptyArr = []

        # Array that will hold the paragraphs
        paragraphs = []

        section = []
        # array that will populate the true false array
        for line in resume:
            if len(line) == 0 or bool(re.match('^\s+$', line)) or line in '()~><?'';":\/|{},[]@6&*-_.':
                isEmptyArr.append(True)
            else:
                isEmptyArr.append(False)

        for i in range(len(resume)):
            # print("%s " % resume[i] + "||| % s" % isEmptyArr[i])
            if isEmptyArr[i] is not True:
                section += resume[i]
            else:
                paragraphs.append(section)
                section = ""
        paragraphs.append(section)
        return paragraphs

    # input: job title as a string, location as int, page limit as int
    # Returns a dictionary of {job_id: {title: "string", description: "string", keywords: ["String"]}}
    def get_jobs(self, title, location, page_limit):
        indeedApi = IndeedAPi()
        job_urls = indeedApi.retrieve_urls(title, location, page_limit)

        # this list holds jobs that may have no content.
        unfiltered_jobs = [indeedApi.getJobMarkup(url) for url in job_urls]

        # filtered jobs are jobs in list that have content Note: this holds the markup (html) of the job
        filtered_jobs = indeedApi.filterJobs(unfiltered_jobs)

        # takes the filtered jobs and creates a dictionary that holds all the jobs
        # key = {job_id: {title: "string", description: "string", keywords: ["String"]}}
        json_data = indeedApi.generateJson(filtered_jobs)
        return json_data

    # takes in resume and bool to indicate whether to return list of paragraphs or the entire resume as a string
    def process_resume(self, resume, split_paragraphs):
        file = open(resume, 'rb')
        d = Document(file)
        fullText = []
        resumeText = ""

        for p in d.paragraphs:
            fullText.append(p.text)
        if split_paragraphs is True:
            paragraphs = self.gen_paragraph(d)
            return paragraphs
        else:
            # Iterate through the array removing indices that have length of 0
            for i in range(len(fullText)):
                try:
                    if len(fullText[i]) == 0:
                        fullText.pop(i)
                except IndexError:
                    print("Could not pop indice...")
            # iterate through the fullText array and strip \n and \t
            for i in range(len(fullText)):
                fullText[i] = re.sub("\s+", " ",
                                     fullText[i])  # if there is text remove the extra space and the and the tabs
                resumeText = resumeText + fullText[
                    i] + " "  # concatinate part of the resume text to the job description
            return resumeText

    # pass in a string or list of paragraphs
    # Strips the stopwords for the resume and removes any punctuation that may be found at the last chr of the str
    # remove parts of speech
    def strip_resume_stopwords_punctuation_pos(self, resume):
        punctuation = '~><?'';:\/|{}[]@6&*-_,.'
        POSToKeep = ["JJ", "JJR", "JJS", "NN", "NNS", "NNP", "NNPS", "VB", "VBD", "VBG", "VBN", "VBP",
                     "VBZ"]  # These are the nouns, verbs, and adjectives we want to keep

        # if string is passed, then return a list of words from the resume with no stopwords
        if isinstance(resume, str):
            resumeNoStopWords = [word for word in resume.split() if word not in stopwords.words('english')]

            # snippet that strips punctuation from the words in the resume.
            for i in range(len(resumeNoStopWords)):
                if resumeNoStopWords[i][-1] in punctuation:
                    resumeNoStopWords[i] = resumeNoStopWords[i].replace(resumeNoStopWords[i][-1], "")
            # iterate through the words in the resume and remove the terms that are less than 2 and contain any special characters
            for i in range(len(resumeNoStopWords)):
                try:
                    if len(resumeNoStopWords[i]) < 2 and resumeNoStopWords[i][0] in '()~><?'';":\/|{},[]@6&*-_':
                        resumeNoStopWords.pop(i)
                except IndexError:
                    print("Could not pop indice...")
            # retrieve parts of speech from each term in the resume, keeping only nouns verbs and adjectives
            resumeNoStopWords = " ".join(w for w in resumeNoStopWords if len(w) > 0)

            resumePOSTags = nltk.pos_tag(word_tokenize(resumeNoStopWords))
            for w in resumePOSTags:
                if w[1] not in POSToKeep:
                    resumePOSTags.remove(w)
            resumeNoStopWordsUpdated = [w[0] for w in resumePOSTags]
            return resumeNoStopWordsUpdated


        # if list is passed in, then return a 2d list, which are the paragraphs consisting of stopwords
        elif isinstance(resume, list):
            resumeNoStopWords = []
            for p in resume:
                if len(p) < 1:
                    continue
                else:
                    # Strip the stopwords
                    # add each word to the paragraph and add the paragraph to the list
                    paragraph = [word for word in p.split() if word not in stopwords.words('english')]
                    resumeNoStopWords.append(paragraph)

            # Remove puncuation
            for i in range(len(resumeNoStopWords)):
                for j in range(len(resumeNoStopWords[i])):
                    if resumeNoStopWords[i][j][-1] in punctuation:
                        resumeNoStopWords[i][j] = resumeNoStopWords[i][j].replace(resumeNoStopWords[i][j][-1], "")

            # iterate through the words in the resume and remove the terms that are less than 2 and contain any special characters
            for i in range(len(resumeNoStopWords)):
                for j in range(len(resumeNoStopWords[i])):
                    try:
                        if len(resumeNoStopWords[i][j]) < 2 and resumeNoStopWords[i][j][
                            0] in '()~><?'';":\/|{},[]@6&*-_':
                            resumeNoStopWords[i].pop(j)
                    except IndexError:
                        print("Could not pop indice...")

            resumePOSTags = [nltk.pos_tag(word_tokenize(" ".join(w for w in section))) for section in resumeNoStopWords]
            for i, line in enumerate(resumePOSTags):
                for w in line:
                    if w[1] not in POSToKeep:
                        resumePOSTags[i].remove(w)

            resumeNoStopWordsUpdated = [[w[0] for w in line] for line in resumePOSTags]
            return resumeNoStopWordsUpdated



        else:
            return Exception("list or str not passed in")

    # Pass in jobs dictionary, strip stopwords, remove certain parts of speech, and strip punctuation
    def process_jobs(self, jobs):
        POSToKeep = ["JJ", "JJR", "JJS", "NN", "NNS", "NNP", "NNPS", "VB", "VBD", "VBG", "VBN", "VBP",
                     "VBZ"]  # These are the nouns, verbs, and adjectives we want to keep
        punctuation = '~><?'';:\/|{}[]@6&*-_,.'
        jobsNoStopWords = {}  # job_id: {title: "string", descriptionNoStopwords: []
        # job_id = 0
        for i in range(len(jobs)):
            # Create an empty dictionary
            jobsNoStopWords[i] = {}

            # Store title from json_data
            jobsNoStopWords[i]["title"] = jobs[i]["title"]

            # Store the words with the stopwords stripped out represented as a list
            jobsNoStopWords[i]["description"] = [w for w in jobs[i]["description"].split() if
                                                 w not in stopwords.words('english')]

        jobsNoStopWordsUpdated = {}

        # Remove parts of speech
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
        return jobsNoStopWordsUpdated

    # Function to determine the tf_idf
    # Training set = job descriptions
    # Test set = resume(s) or set of resumes
    # Returns tuple consisting of x,y and vocab
    def tf_idf(self, training_set, test_set):
        tf_idf_vectorizer = TfidfVectorizer(use_idf=False, sublinear_tf=False, stop_words=stopwords.words('english'))
        corpus = []
        for i in range(len(training_set)):
            corpus.append(training_set[i]["title"] + " " + " ".join(word for word in training_set[i]["description"]))

        # create bag of words and transform corpus into a document term frequency matrix
        x = tf_idf_vectorizer.fit_transform(corpus)
        arrType = np.asarray(test_set)
        if len(arrType.shape) == 1:

            # transform the resume into a term frequency matrix
            resumeCorpus = " ".join(word for word in test_set)
            y = tf_idf_vectorizer.transform([resumeCorpus])

        # This portion of code is untested
        elif len(arrType.shape) == 2:
            resumeCorpus = [" ".join(word for word in section) for section in test_set]
            y = tf_idf_vectorizer.transform(resumeCorpus)

        else:
            return Exception("Either a 1 or 2 dimensional array must be passed")

        # get the vocabulary
        bagOfWords = tf_idf_vectorizer.vocabulary_

        # return tuple
        return (x, y, bagOfWords)

    # Function to get the cosine similarity
    # Takes in two document term frequency matrixes returned from the tf-idf function
    def get_cosine_similarity(self, x, y):
        return cosine_similarity(x, y).flatten()
