from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.ensemble import GradientBoostingClassifier

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

    vectorizer = build_pipeline_steps(do_bigram_sent=True, do_unigram_sent=True, bigram_sent_file="bigrams-pmilexicon.txt", unigram_sent_file="unigrams-pmilexicon.txt")

    X = vectorizer.fit_transform(tweets)

    clf = LogisticRegression()
    #clf = SGDClassifier()
    #clf = svm.LinearSVM()
    
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
    
if __name__ == "__main__":
    pos = "positive.txt"
    neg = "negative.txt"

    test_doc = ["This is not a good tweet :)", "I love you !", "I want to kill you", "lets go party !"]
    test_doc2 = ["I am really drunk", "I hate you", "I want to kill you", "lets go party !"]
    
    corpus = read(pos, 1) + read(neg, -1)
    
    #test(corpus)
    t = train(corpus)
    
    print(t.predict_proba(test_doc2))
    print(t.score_document(test_doc2))

