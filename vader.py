from nltk.sentiment.vader import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()


# 5/17/2021 - Perform vader score analysis on one review
def analyze_polarity_scores(review):
    return analyzer.polarity_scores(review)

# Vader class
class Vader:
    def __init__(self):
        self.analysis_results = []

    # Perform vader analysis 
    def vader_analysis(self, reviews):
        print("Vader analysis:")
        counter = 0
        for review in reviews:
            counter += 1
            if counter % 20000 == 0:
                print(counter)
            if analyzer.polarity_scores(review)['compound'] > 0:
                self.analysis_results.append(True)
            else:
                self.analysis_results.append(False)
            # self.analysis_results.append(analyzer.polarity_scores(review))
        return self.analysis_results

    def vader_validation(self, y_train):
        print("Vader validation:")
        accuracy = 0
        counter = 0
        temp = y_train.to_numpy()
        print("Length of vader analysis result:", len(self.analysis_results))
        print("Length of testing set (y_train):", len(temp))
        for i, j in zip(self.analysis_results, temp):
            if i == j:
                accuracy += 1
            # if counter % 10000 == 0:
            #     print("Compared:", counter)
            counter += 1
        print("# of Matching result:", accuracy)
        print("Total compared reviews:", counter)
        accuracy_rate = accuracy / len(self.analysis_results)
        return accuracy_rate

