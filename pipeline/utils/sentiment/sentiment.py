"""
Guidelines :

1) Call sentiment_of_document for finding the sentiment of some particular documents.
2) The following files work in support with tweet_feature_extractor.py file
   
"""



from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import cross_validate

import random,os
from tweets_feature_extractor import build_pipeline_steps

current_directory = os.path.dirname(__file__)

bigram_lexicon = current_directory + "/" + "bigrams-pmilexicon.txt"
unigram_lexicon = current_directory + "/" + "unigrams-pmilexicon.txt"


def read(path, tag):    
    with open(path, "r") as f:
        tweets = f.readlines()
    tweet_tag = [[tweet, tag] for tweet in tweets]
    return tweet_tag


def test(corpus):    
    random.seed(42)
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
    vectorizer = build_pipeline_steps(do_bigram_sent=True, do_unigram_sent=True, bigram_sent_file=bigram_lexicon, unigram_sent_file=unigram_lexicon)
    X = vectorizer.fit_transform(tweets)
    clf = LogisticRegression()
    clf.fit(X, labels)
    return TweetClf(clf, vectorizer)

class TweetClf:
    
    def __init__(self, clf, vectorizer):
        self.classifier = clf
        self.vectorizer = vectorizer
        
    @property    
    def clf(self):
        return self.classifier
        
    def vectorize(self, tweets):
        return self.vectorizer.transform(tweets)
        
    def predict(self, tweets):
        X = self.vectorize(tweets)
        return self.clf.predict(X)
        
    def predict_proba(self, tweets):
        X = self.vectorize(tweets)
        return self.clf.predict_proba(X)
        
    def score_document(self, tweets):
        probs = self.predict_proba(tweets)
        neg_score, pos_score = 0, 0
        for neg, pos in probs:
            neg_score += neg
            pos_score += pos
        return (pos_score - neg_score) / len(probs)

# For finding the sentiment of individual tweets
def find_sentiment_tweet(tweets):
    """input :
        pandas dataseries"""
    #return data.predict_proba([tweets])
    return data.score_document([tweets])

# For finding the sentiment of all the documents in all the grids
def find_sentiment_doc(list_Sentence):
    """input :
        list of sentences"""
	#return data.predict_proba(list_Sentence)
    return data.score_document(list_Sentence)

# For finding the sentiment of some particular documents
def sentiment_of_document(documents):
    """input :
        a) List of List of List
        b) tokenized words""" 
    sentiment_list,tweets_list,document_list = ([] for i in range(3))
    for document in documents:
        for tweets in document:
            tweets_list.append(" ".join(tweets))
        document_list.append(tweets_list)
        sentiment_list.append(findSentimentDoc(document_list[-1]))
    return sentiment_list

pos = current_directory + "/" + "positive.txt"
neg = current_directory + "/" + "negative.txt"
corpus = read(pos, 1) + read(neg, -1)
#testing = test(corpus)
data = train(corpus)
	

