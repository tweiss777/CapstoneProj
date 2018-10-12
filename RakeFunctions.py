from rake_nltk import Rake


class RakeFunctions:

    def process_rake(self,text):
        r = Rake()
        r.extract_keywords_from_text(text)
        return r.get_ranked_phrases()
