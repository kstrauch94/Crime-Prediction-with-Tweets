from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_validate

import random
from tweets_feature_extractor import build_pipeline_steps

bigram_lexicon = "bigrams-pmilexicon.txt"
unigram_lexicon = "unigrams-pmilexicon.txt"


def read(path, tag):
	
	with open(path, "r") as f:
		tweets = f.readlines()
		
	tweet_tag = [[tweet, tag] for tweet in tweets]
	
	return tweet_tag


def test(corpus):	

	random.shuffle(corpus)

	tweets, labels = zip(*corpus)

	vectorizer = build_pipeline_steps(do_bigram_sent=True, do_unigram_sent=True, bigram_sent_file=bigram_lexicon, unigram_sent_file=unigram_lexicon)

	X = vectorizer.fit_transform(tweets)

	clf = LogisticRegression()

	scoring = ["f1_micro", "f1_macro", "precision_micro", "precision_macro", "recall_micro", "recall_macro"]
	  
	f1_scores = cross_validate(clf, X, labels, cv=10, scoring=scoring, return_train_score=False)

	for score_name, scores in f1_scores.items():
		print("average {} : {}\n".format(score_name,sum(scores)/len(scores)))
		
def train(corpus):
	random.shuffle(corpus)

	tweets, labels = zip(*corpus)

	vectorizer = build_pipeline_steps(do_bigram_sent=True, do_unigram_sent=True, bigram_sent_file="bigrams-pmilexicon.txt", unigram_sent_file="unigrams-pmilexicon.txt")

	X = vectorizer.fit_transform(tweets)

	clf = LogisticRegression()

	clf.fit(X, labels)

	return clf

if __name__ == "__main__":
	pos = "positive.txt"
	neg = "negative.txt"

	corpus = read(pos, "pos") + read(neg, "neg")
	
	test(corpus)

