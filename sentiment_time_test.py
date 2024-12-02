import time as time

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer



analyzer = SentimentIntensityAnalyzer()

start = time.time()
print(start)
for i in range(2):
    print(analyzer.polarity_scores("This is some very bad shit I'mm seeing here")['compound'])

end = time.time()
print(end-start)
