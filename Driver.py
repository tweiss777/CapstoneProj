import numpy as np
from gensim.summarization import keywords

from DataProcessor import *


def main():
    # initialize the data processor object
    dp = DataProcessor()

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

    # Retrieve possible skills in resume by returning only the nouns found
    possible_skills = dp.get_skills(resumeList, ["NN", "NNS", "NNP", "NNPS", "JJ", "JJR", "JJS"])

    # Get keywords from the resume
    resume_keywords = keywords(resumeStr, pos_filter=("JJ", "JJR", "JJS", "NN", "NNS", "NNP", "NNPS"), split="\n")

    # pre process the resume
    resumeStrUpdated = dp.strip_resume_stopwords_punctuation_pos(resumeStr)
    resumeListUpdated = dp.strip_resume_stopwords_punctuation_pos(resumeList)

    # pre process the jobs wihtout the bigrams
    processed_jobs = dp.process_jobs(jobs)
    processed_jobs_noBigrams = processed_jobs

    # Method that will retrieve the bigrams for the jobs
    processed_jobs_no_bigrams = processed_jobs
    for i in range(len(processed_jobs)):
        processed_jobs[i]["description"] = dp.get_bigrams(processed_jobs[i]["description"], 2)

    processed_jobs_all_bigrams = dp.get_all_bigrams(processed_jobs_no_bigrams, 3)

    # process tf-idf for the whole resume
    # x = the jobs
    # y = the resume
    # process tf-idf for the whole resume
    x1, y1, bagOfWords = dp.tf_idf(processed_jobs, resumeStrUpdated)

    similarity_score_whole = dp.get_cosine_similarity(x1, y1)

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
    # filter keywords by
    keywords_per_job = []
    for job_id, job in jobs.items():
        keywords_per_job.append(
            keywords(job["description"], pos_filter=("JJ", "JJR", "JJS", "NN", "NNS", "NNP", "NNPS"), split="\n"))

    # Get the tf-idf to determine the keywords from the job set
    # Consider setting use_idf to true
    tf_idf_vectorizer = TfidfVectorizer(use_idf=True, sublinear_tf=False, stop_words=stopwords.words('english'))

    # create the dataset from the processed jobs dictionary
    corpus = [" ".join(word for word in job["description"]) for i, job in processed_jobs_all_bigrams.items()]
    resumeKeywordCorpus = " ".join(word for word in resume_keywords)
    # Train the data set
    x = tf_idf_vectorizer.fit_transform(corpus)
    # pass the resume as the test set
    y = tf_idf_vectorizer.transform([resumeKeywordCorpus])

    # obtain the similarity scores between the resume keywords and corpus
    similarity_score_3 = dp.get_cosine_similarity(x, y)

    # sort indices from higher to lower
    sorted_sim_score_indices = similarity_score_3.argsort()[::-1]

    # output the cosine similarity scores scores
    for indice in sorted_sim_score_indices:
        print(jobs[indice]["title"], " || ", similarity_score_3[indice])

    # retrieve & store the vocabulary
    bagOfWords2 = tf_idf_vectorizer.vocabulary_

    # Get the top n words from based on the highest tf-idf weights
    # S1) get the feature names and store it in an numpy array
    features_array = np.array(tf_idf_vectorizer.get_feature_names())

    # S2) iterate through the job corpus and run argsort over each set of tf-idf weights in the compressed sparse matrix
    # The below loop will also output the job title, list of top 10 terms, and weight that belongs to the term
    for i in range(len(jobs)):
        tfidf_sorting = np.argsort(x.toarray()[i]).flatten()[::-1]
        print(jobs[i]["title"], "\n")
        print("%s)" % i, features_array[tfidf_sorting][:10], "\n")
        for j, score in enumerate(x.toarray()[i][tfidf_sorting][:10]):
            print("%s) " % j, score, " %s" % features_array[tfidf_sorting][j])
        print("\n")

    # Get top n words based on the lowest tf-idf weights
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


main()