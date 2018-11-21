from DataProcessor import *


def main():
    dp = DataProcessor()

    jobs = dp.get_jobs("Java Developer", 11590, 10)

    resumeStr = dp.process_resume("TalWeissResume.docx", False)
    resumeList = dp.process_resume("TalWeissResume.docx", True)

    resumeStrUpdated = dp.strip_resume_stopwords_punctuation_pos(resumeStr)
    resumeListUpdated = dp.strip_resume_stopwords_punctuation_pos(resumeList)  # Untested

    processed_jobs = dp.process_jobs(jobs)

    x, y, bagOfWords = dp.tf_idf(processed_jobs, resumeStrUpdated)

    similarity_score = dp.get_cosine_similarity(x, y)


main()
