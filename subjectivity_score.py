def tweet_sub_score(tweet, score_lookup):
    
    score = 0
    
    for word in tweet:
        if word in score_lookup:
            score += score_lookup[word]
            
    return score
    
def sub_score(pos_tweet):
    
    subj_score_file = "Data" + os.sep + "subjectivity_score.txt"

    score_lookup = process_subjectivity_file(subj_score_file)
    
    return np.array([tweet_sub_score(tw[1], score_lookup) for tw in pos_tweet]).reshape(-1, 1)
	
def process_subjectivity_file(filename):

    scores = {}
    
    with open(filename, "r") as f:
        for line in f.readlines():
            line = line.split(" ")
            word = line[2].split("=")[1]
            score = line[-1].split("=")[1].strip()
            if score == "negative":
                score = -1
            elif score == "positive":
                score = 1
            else:
                score = 0
            
            scores[word] = score
    
    return scores