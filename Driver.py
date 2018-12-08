from DataProcessor import *


def main():
    dp = DataProcessor()

    # Get the jobs from indeed.com
    jobs, jobs2 = dp.get_jobs("Java Developer", 11590, 10)


    jobs2_bigrams = dp.get_all_bigrams_paragraphs(jobs2, 3)

    # strip certain parts of speech
    jobs2_bigrams_processed = dp.process_jobs_paragraphs(jobs2_bigrams)

    # load the resume up as a doc or docx file
    resumeStr = dp.process_resume("TalWeissResume.docx", False)

    # This segregates the paragraphs in the resume
    resumeList = dp.process_resume("TalWeissResume.docx", True)
    resumeList = [paragraph for paragraph in resumeList if len(paragraph) > 0]
    # pre process the resume
    resumeStrUpdated = dp.strip_resume_stopwords_punctuation_pos(resumeStr)
    resumeListUpdated = dp.strip_resume_stopwords_punctuation_pos(resumeList)  # Untested

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

    # retrieve the top 5 job indices
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
    similarity_scores = {}
    for doc_id, paragraph_scores in tf_idf_for_top_5_jobs_paragraphs.items():
        similarity_scores[doc_id] = {}
        for paragraph_num, matrix in paragraph_scores.items():
            similarity_scores[doc_id][paragraph_num] = dp.get_cosine_similarity(matrix[0], matrix[1])

    # store 3 closes paragraphs from job description for each paragraph in your resume
    top_3_paragraphs_per_job = {}
    for doc_id, paragraph_scores in similarity_scores.items():
        top_3_paragraphs_per_job[doc_id] = {}
        for paragraph_num, scores in paragraph_scores.items():
            top_3_paragraphs_per_job[doc_id][paragraph_num] = scores.argsort()[:-4:-1]


main()
