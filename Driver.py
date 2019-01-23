import re as regex

import numpy as np
from gensim.summarization import keywords

from DataProcessor import *


def main():
    containsNumsSpecialChars = r'^[!@#$%^&*(),.?":{}|<>0-9]*$'
    # BEGIN SECTION FOR PRE-PROCESSING

    # initialize the data processor object
    dp = DataProcessor()
    r = Rake()
    # Get the jobs from indeed.com
    jobs, jobs2 = dp.get_jobs("Java Developer", 11590, 10)

    # get the bigrams for the paragraph separated jobs
    jobs2_bigrams = dp.get_all_bigrams_paragraphs(jobs2, 3)

    # strip certain parts of speech
    jobs2_bigrams_processed = dp.process_jobs_paragraphs(jobs2_bigrams)

    # load the resume up as a doc or docx file
    resumeStr = dp.process_resume("TalWeissResume.docx", False)

    # This segregates the paragraphs in the resume
    resumeList = dp.process_resume("TalWeissResume.docx", True)

    #filter out empty paragraphs
    resumeList = [paragraph for paragraph in resumeList if len(paragraph) > 0]


    # pre process the resume
    resumeStrUpdated = dp.strip_resume_stopwords_punctuation_pos(resumeStr)
    resumeListUpdated = dp.strip_resume_stopwords_punctuation_pos(resumeList)

    # Retrieve the key phrases from the resume by using nltk_rake library
    r.extract_keywords_from_text(" ".join(word for word in resumeStrUpdated))
    resume_key_phrases = r.get_ranked_phrases()

    # get potential keywords filter by pos
    # resume_keywords = dp.filter_pos(resumeStrUpdated, ["NN", "NNS", "NNP", "NNPS"])
    # resume_keywords = list(set(resume_keywords))

    # #getting keywords via gensim,
    resume_keywords = keywords(" ".join(w for w in resumeStrUpdated), split="\n")
    resume_keywords = [w[0] for w in nltk.pos_tag(resume_keywords) if w[1] in ["NN", "NNS", "NNP", "NNPS"]]
    resume_keywords = [w for w in resume_keywords if bool(regex.match(containsNumsSpecialChars, w)) is False]

    # pre process the jobs wihtout the bigrams
    processed_jobs = dp.process_jobs(jobs)
    processed_jobs_noBigrams = processed_jobs

    # Method that will retrieve the bigrams for the jobs
    # Bi-grams that appear twice or more
    processed_jobs_no_bigrams = processed_jobs
    for i in range(len(processed_jobs)):
        processed_jobs[i]["description"] = dp.get_bigrams(processed_jobs[i]["description"], 2)

    processed_jobs_all_bigrams = dp.get_all_bigrams(processed_jobs_no_bigrams, 3)

    # process tf-idf for the whole resume
    # x = the jobs
    # y = the resume
    # process tf-idf for the whole resume
    x1, y1, features = dp.tf_idf(processed_jobs_all_bigrams, resumeStrUpdated)

    similarity_score_whole = dp.get_cosine_similarity(x1, y1)

    # END OF PRE-PROCESSING SECTION

    # BEGIN OUTPUTTING RESULTS

    # retrieve the top 5 job indices using argsort
    # argsort sorts by putting the highest valued indice at the last index
    top_5_indices = similarity_score_whole.argsort()[:-6:-1]

    top_5_jobs = {}
    for i in top_5_indices:
        top_5_jobs[i] = (similarity_score_whole[i], jobs[i]["title"] + " " + jobs[i]["description"])

    # Take the top 5 scores and use them to get the relevant paragraphs.
    top_5_jobs_paragraphs = {}
    for i in top_5_indices:
        top_5_jobs_paragraphs[i] = {}
        top_5_jobs_paragraphs[i]["title"] = jobs2_bigrams_processed[i]["title"]
        top_5_jobs_paragraphs[i]["description"] = jobs2_bigrams_processed[i]["description"]

    # tf idf scores for each paragraph in the resume
    tf_idf_for_top_5_jobs_paragraphs = dp.tf_idf2(top_5_jobs_paragraphs, resumeListUpdated)

    # holds the similarity score between each paragraph in the resume with each paragraph in the top 5 jobs
    similarity_scores_paragraphs = {}
    for doc_id, paragraph_scores in tf_idf_for_top_5_jobs_paragraphs.items():
        similarity_scores_paragraphs[doc_id] = {}
        for paragraph_num, matrix in paragraph_scores.items():
            similarity_scores_paragraphs[doc_id][paragraph_num] = dp.get_cosine_similarity(matrix[0], matrix[1])

    # store 3 closes paragraphs from job description for each paragraph in your resume
    top_3_paragraphs_per_job = {}
    for doc_id, paragraph_scores in similarity_scores_paragraphs.items():
        top_3_paragraphs_per_job[doc_id] = {}
        for paragraph_num, scores in paragraph_scores.items():
            top_3_paragraphs_per_job[doc_id][paragraph_num] = scores.argsort()[:-4:-1]

    # iterate through the dictionary that holds the top 3 paragraphs in jobs for each paragraph in the resume
    # output the job title
    # print the resume paragraph number
    # print the closest paragraph indices that are close to the
    # print the actual scores themselves
    for doc_id, resume_paragraphs in top_3_paragraphs_per_job.items():
        print("\njob %s" % doc_id)
        print(jobs[doc_id]["title"])
        for resume_paragraph, job_paragraphs in resume_paragraphs.items():
            print("paragraph %s in your resume is close to the following paragraph for this job \n" % resume_paragraph,
                  job_paragraphs)
            print("Actual cosine similarity scores: ",
                  similarity_scores_paragraphs[doc_id][resume_paragraph][job_paragraphs], "\n")

    # Iterate through the top 3 paragraphs per job and take out the paragraphs that have a cosine sim of 0.
    # Make a list of possible skills taken from nouns and adjectives (build a set of related words with skills.
    # Play with the gensim library to get keywords
    # Look at tf-idf by order of decreasing value for the words
    for doc_id, paragraphs in top_3_paragraphs_per_job.items():
        for resume_paragraph, job_paragraphs in paragraphs.items():
            for i, job in enumerate(job_paragraphs):
                try:
                    top_3_paragraphs_per_job[doc_id][resume_paragraph] = top_3_paragraphs_per_job[doc_id][
                        resume_paragraph].tolist()
                except:
                    print("conversion failed")

                if similarity_scores_paragraphs[doc_id][resume_paragraph][job] == 0.0:
                    print("removing indice with score %s" % similarity_scores_paragraphs[doc_id][resume_paragraph][job])
                    print(i)
                    # print(top_3_paragraphs_per_job[doc_id][resume_paragraph][i])
                    try:
                        top_3_paragraphs_per_job[doc_id][resume_paragraph].remove(job)
                    except:
                        print("could not pop indice")

    # Output the filtered indices in which the similarity score is greater than 0
    for doc_id, resume_paragraphs in top_3_paragraphs_per_job.items():
        print("job %s" % doc_id)
        for resume_paragraph, job_paragraphs in resume_paragraphs.items():
            print("\njob paragraph % s " % job_paragraphs)
            print("scores from job paragraphs")
            for i, score in enumerate(similarity_scores_paragraphs[doc_id][resume_paragraph][job_paragraphs]):
                print(score)

                if score == 0.0:
                    print("the score %s" % score, " was not removed and is associated with %s " %
                          top_3_paragraphs_per_job[doc_id][resume_paragraph][i])

    # get a list of keywords from the list of jobs using gensim.summarization
    # Consider removing this list and for-loop as it is un
    keywords_per_job = []
    for job_id, job in jobs.items():
        keywords_per_job.append(
            keywords(job["description"], pos_filter=("JJ", "JJR", "JJS", "NN", "NNS", "NNP", "NNPS"), split="\n"))

    # Get the tf-idf to determine the keywords from the job set
    # Consider setting use_idf to true
    tf_idf_vectorizer = TfidfVectorizer(use_idf=True, sublinear_tf=False, stop_words=stopwords.words('english'))

    # create the dataset from the processed jobs dictionary
    # corpus = [job["title"] + " " + " ".join(word for word in job["description"]) for i, job in
    #           processed_jobs_all_bigrams.items()]
    resumeKeywordCorpus = " ".join(word for word in resume_keywords)

    corpus_keywords = []
    for i in range(len(keywords_per_job)):
        corpus_keywords.append(jobs[i]["title"] + " " + " ".join(keyword for keyword in keywords_per_job[i]))

    # filter the corpus keywords by keeping only nouns for processing tf-idf results
    for i in range(len(corpus_keywords)):
        corpus_keywords[i] = " ".join(
            word[0] for word in nltk.pos_tag(corpus_keywords[i].split()) if word[1] in ["NN", "NNS", "NNP", "NNPS"])

    # filter the corpus by keeping only nouns for preprocessing.
    # for i in range(len(corpus)):
    #     corpus[i] = " ".join(w[0] for w in nltk.pos_tag(corpus[i].split()) if w[1] in ["NN","NNS","NNP","NNPS"])

    # tf-idf between keywords in the resume and the jobs
    # Train the data set
    x = tf_idf_vectorizer.fit_transform(corpus_keywords)
    # pass the resume as the test set
    y = tf_idf_vectorizer.transform([resumeKeywordCorpus])

    # obtain the similarity scores between the resume keywords and corpus
    similarity_score_3 = dp.get_cosine_similarity(x, y)

    # sort indices from higher to lower score
    sorted_sim_score_indices = similarity_score_3.argsort()[::-1]

    # output the cosine similarity scores scores
    print("============= COSINE SIMILARITY SCORES BETWEEN RESUME KEYWORDS AND JOBS =============")
    for indice in sorted_sim_score_indices:
        print("%s)" % indice, jobs[indice]["title"], " || ", similarity_score_3[indice])


    # Get the top n words from based on the highest tf-idf weights
    # S1) get the feature names and store it in an numpy array
    features_array = np.array(tf_idf_vectorizer.get_feature_names())

    # S2) iterate through the job corpus and run argsort over each set of tf-idf weights in the compressed sparse matrix
    # The below loop will also output the job title, list of top 10 terms, and weight that belongs to the term
    # Outputs score from highest to lowest
    print("============= TF-IDF for higher weight terms =============")
    for i in range(len(jobs)):
        tfidf_sorting = np.argsort(x.toarray()[i]).flatten()[::-1]
        print(jobs[i]["title"], "\n")
        print("%s)" % i, features_array[tfidf_sorting][:10], "\n")
        for j, score in enumerate(x.toarray()[i][tfidf_sorting][:10]):
            print("%s) " % j, score, " %s" % features_array[tfidf_sorting][j])
        print("\n")

    # Get top n words based on the lowest tf-idf weights
    print("============= TF-IDF for lower weight terms =============")
    for i in range(len(jobs)):
        copyX = np.array(x.toarray()[i].flatten())  # save a copy of the array
        tfidf_sorting = np.argsort(copyX)

        print("%s)" % i, jobs[i]["title"], "\n")
        count = 0  # counter to keep track of the top terms to print
        for j, score in enumerate(copyX[tfidf_sorting]):
            if score > 0.0:
                print("%s) " % count, score, " %s" % features_array[tfidf_sorting][j])  # print the features
                count += 1
                if count == 10:
                    break  # exit loop if we printed out the top n terms
        print("\n")

    # Section to get the matching keywords from the entire dataset sorted based on relevancy of job
    top_indices = similarity_score_whole.argsort()[::-1]

    # list that holds a tuple (job id, list of keywords) consisting of matching keywords from the job and resume
    matchingKeyWordsPerJob = []

    # holds a tuple that consists of keywords missing from the job description but found in the resume
    nonMatchingKeywordPerJobForResume = []

    #holds a tuple that consists of keywords missing from the resume but found in the job description
    nonMatchingKeyWordsPerJob = []

    # updated list of words from the resume filtered for proper nouns
    resumeStrUpdatedPosFiltered = dp.filter_pos(resumeStrUpdated, POS_to_keep=["NNP"])

    # Lowercase the terms from the resume to be used to find intersection with the propernouns
    resumeProperNouns = [t.lower() for t in resumeStrUpdatedPosFiltered]

    # tuple that consists of job id and list of all keywords from the job only
    jobKeyWordsOnly = []
    for indice in top_indices:
        jobKeywords = dp.filter_pos(processed_jobs_all_bigrams[indice]["description"], POS_to_keep=["NNP"])
        jobKeyWordsOnly.append((indice, jobKeywords))


    for indice in top_indices:
        jobKeywords, nonMatchingWordsResume, nonMatchingWordsJobs = dp.compare_words(resumeStrUpdated,
                                                                                     processed_jobs_all_bigrams[indice][
                                                                                         "description"],
                                                                                     filter_pos=["NNP"])
        matchingKeyWordsPerJob.append((indice, jobKeywords))
        nonMatchingKeywordPerJobForResume.append((indice, nonMatchingWordsResume))
        nonMatchingKeyWordsPerJob.append((indice, nonMatchingWordsJobs))


    jobKeywordsOnlyFrequency = {}

    for job_id, terms in jobKeyWordsOnly:
        for term in terms:
            if term in jobKeywordsOnlyFrequency:
                jobKeywordsOnlyFrequency[term] += 1
            else:
                jobKeywordsOnlyFrequency[term] = 1



    jobKeywordsOnlyFrequencySorted = sorted(jobKeywordsOnlyFrequency, key=lambda x: jobKeywordsOnlyFrequency[x])

    # get the sum of the total number of words in the entire job set
    totalWordsInJobSet = 0
    for term, count in jobKeywordsOnlyFrequency.items():
        totalWordsInJobSet = totalWordsInJobSet + count

    # filter the words with a frequency greater than 1.5% (L1)
    jobKeywordsOnlyFrequencySortedUpdated = [term for term in jobKeywordsOnlyFrequencySorted if
                                             (jobKeywordsOnlyFrequency[term] / totalWordsInJobSet) * (100) >= .15]

    # get the intersection of words from the resume and the proper nouns

    # initialize var called properNouns which will take the terms from the jobKeyWordsOnlyFrequencySortedUpdated and lower case the terms
    properNouns = [t.lower() for t in jobKeywordsOnlyFrequencySortedUpdated]

    # typecase properNouns to a set same with proper nouns from the resume
    properNounsSet = set(properNouns)
    resumeProperNounsSet = set(resumeProperNouns)

    # retrieve the intersection of resume terms and proper nouns below (L2
    intersectionResumeJobs = resumeProperNounsSet.intersection(properNounsSet)

    # retrieve the intersection of the words from each of the top 5 jobs with the proper nouns (L3)
    matchingProperNounsPerTop5Jobs = []
    for indice in top_5_indices:
        nnps = dp.filter_pos(processed_jobs_all_bigrams[indice]["description"], POS_to_keep=["NNP"])
        nnps = [t.lower() for t in nnps]
        matchingProperNounsPerTop5Jobs.append((indice, properNounsSet.intersection(set(nnps))))

    properNounsResumeNTop5Jobs = []
    # retrieve the intersection between the keywords in your resume and top 5 jobs
    for indice, properNouns in matchingProperNounsPerTop5Jobs:
        properNounsResumeNTop5Jobs.append((indice, properNouns.intersection(resumeProperNounsSet)))

    # retrieve the proper noun non-matches between each of the top 5 jobs and the resume
    nonMatchesPerTop5Jobs = []

    # add terms that are in the job description but not in the resume
    for jobId, terms in matchingProperNounsPerTop5Jobs:
        nonMatchesPerTop5Jobs.append((jobId, [t for t in terms if t not in intersectionResumeJobs]))



main()