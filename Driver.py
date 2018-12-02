from DataProcessor import *


def main():
    dp = DataProcessor()

    # Get the jobs from indeed.com
    jobs = dp.get_jobs("Java Developer", 11590, 10,separate_paragraphs=False)

    # These jobs are paragraph divided
    jobs2 = dp.get_jobs("Java Developer", 11590,10,separate_paragraphs=True)
    # get the bigrams for the paragraph segregated jobs
    jobs2_bigrams = dp.get_all_bigrams_paragraphs(jobs2, 3)

    # strip certain parts of speech
    jobs2_bigrams_processed = dp.process_jobs_paragraphs(jobs2_bigrams)

    # load the resume up as a doc or docx file
    resumeStr = dp.process_resume("TalWeissResume.docx", False)

    # This segregates the paragraphs in the resume
    resumeList = dp.process_resume("TalWeissResume.docx", True)

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
    x1, y1, bagOfWords = dp.tf_idf(processed_jobs, resumeStrUpdated,paragraphs=False)

    #process tf-idf for the paragraphs in the resume
    x2,y2, bagOfWords = dp.tf_idf(processed_jobs_all_bigrams,resumeListUpdated,paragraphs=True)

    similarity_score_whole = dp.get_cosine_similarity(x1, y1)

    similarity_score_paragraphs = dp.get_cosine_similarity(x2,y2)

    # retrieve the top 5 job indices
    top_5_indices = similarity_score_whole.argsort()[:-5:-1]

    top_5_jobs = {}
    for i in top_5_indices:
        top_5_jobs[i] = (similarity_score_whole[i],jobs[i]["title"] + " " + jobs[i]["description"])




main()
