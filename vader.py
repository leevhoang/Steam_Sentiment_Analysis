from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd

analyzer = SentimentIntensityAnalyzer()

# 5/17/2021 - Perform vader score analysis on one review
def analyze_polarity_scores(review):
	return analyzer.polarity_scores(review)


# 5/17/2021 - Code to provide vader scores for reviews.
# Reviews is a pandas series containing text reviews.
def vader_analysis(reviews):
    # analysis_results = []
    # # counter = 0
    # for review in reviews:
    #     # if counter < 2000:
    #     analysis_results.append(analyzer.polarity_scores(review))
    #     # del analysis_results[counter]['neu']
    #     # counter += 1

    analysis_results = reviews.apply(lambda review: analyze_polarity_scores(review))

    return analysis_results
