from nltk.sentiment.vader import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()


def vader_analysis(reviews):
    analysis_results = []
    # counter = 0
    for review in reviews:
        # if counter < 2000:
        analysis_results.append(analyzer.polarity_scores(review))
        # del analysis_results[counter]['neu']
        # counter += 1
    return analysis_results
