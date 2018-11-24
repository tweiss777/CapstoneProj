from DataProcessor import *


def main():
    dp = DataProcessor()

    jobs = dp.get_jobs("Java Developer", 11590, 10)

    resumeStr = dp.process_resume("TalWeissResume.docx", False)
    resumeList = dp.process_resume("TalWeissResume.docx", True)

    resumeStrUpdated = dp.strip_resume_stopwords_punctuation_pos(resumeStr)
    resumeListUpdated = dp.strip_resume_stopwords_punctuation_pos(resumeList)  # Untested

    processed_jobs = dp.process_jobs(jobs)
    processed_jobs_noBigrams = processed_jobs
    # Make a method that will retrieve the bigrams for the jobs
    processed_jobs_no_bigrams = processed_jobs
    for i in range(len(processed_jobs)):
        processed_jobs[i]["description"] = dp.get_bigrams(processed_jobs[i]["description"], 3)


    x, y, bagOfWords = dp.tf_idf(processed_jobs, resumeStrUpdated)

    similarity_score = dp.get_cosine_similarity(x, y)


main()
