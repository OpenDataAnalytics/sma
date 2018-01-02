from sklearn import svm, model_selection, tree
import json
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
import numpy as np


# all the common stopwords we may have
stopwords = ["a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost",
             "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "amoungst",
             "amount", "an", "and", "another", "any", "anyhow", "anything", "anyway", "anywhere", "are",
             "around", "as", "at", "back", "be", "became", "because", "become", "becomes", "becoming", "been",
             "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill",
             "both", "bottom", "but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry",
             "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either",
             "eleven", "else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone",
             "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first",
             "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further",
             "get", "give", "go", "had", "has", "hasnt", "have", "hence", "here", "hereafter",
             "hereby", "herein", "hereupon", "how", "however",
             "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "keep",
             "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "meanwhile",
             "might", "mill", "more", "moreover", "most", "mostly", "move", "much", "must",
             "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no",
             "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto",
             "or", "otherwise", "out", "over", "own", "part", "per",
             "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems",
             "serious", "several", "should", "show", "side", "since", "sincere", "six", "sixty", "so",
             "somehow", "sometime", "sometimes", "somewhere", "still", "such",
             "system", "take", "ten", "than", "that", "the", "then", "thence",
             "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "thickv",
             "thin", "third", "this", "though", "three", "through", "throughout", "thru", "thus", "to",
             "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up",
             "upon", "very", "via", "was", "well", "were", "what", "whatever", "when", "whence",
             "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether",
             "which", "while", "whether", "whole", "why", "will", "with",
             "within", "without", "would", "yet", "the", "it's"]


key_words = ["asthma", "inhaler", "runny", "nose", "wheezing", "sneezing", "breath", "breathing", "breath-triggered",
             "survivor", "fever", "cough", "coughing", "lung", "sore", "throat", "chest", "stress", "headache",
             "respiratory", "weaken", "tired", "weak", "anxiety", "panic", "pale", "grouchy",
             "moody", "cold", "exercise", "sweaty", "blue", "lips", "fingernails", "stress", "irritants", "parents",
             "family", "pollen", "airquality", "albuterol", "xolair", "montelukast", "nebulizer",
             "flovent", "singulair", "advair", "bronchodilator", "bronchodilators", "short of breath", "chest tight"]

personal_pronouns = ["i", "me", "we", "my", "you", "your", "he", "him", "his", "she", "her", "they", "them", "their"]

asthmas = ["asthma", "#asthma", "@asthma", "albuterol", "xolair", "montelukast","flovent", "advair", "bronchodilator", "bronchodilators"]

key_hashtags = ["asthmasurvivor", "asthma", "BeatingAsthma", "survivor", "respiratory", "airquality", "pollen", "cough",
                "wheeze", "sneeze"]

air_qualitys = ["pollen", "airquality", "co", "no2", "pm2.5"]


def prepare_data(file1, file2):
    positive_tweets = []
    negative_tweets = []


    for line in open(file1).readlines():
        tweet = json.loads(line)
        tweet_text = tweet["text"]
        text = tweet_text.lower().strip()

        entities = tweet["entities"]
        hashtag = entities["hashtags"]
        hashtags_find = 0
        if hashtag:
            for item in hashtag:
                if item['text'].lower() in key_hashtags:
                    hashtags_find = 1

        positive_tweets.append([int(1), text, int(hashtags_find)])


    for line in open(file2).readlines():
        tweet = json.loads(line)
        tweet_text = tweet["text"]
        text = tweet_text.lower().strip()

        entities = tweet["entities"]
        hashtag = entities["hashtags"]
        hashtags_find = 0
        if hashtag:
            for item in hashtag:
                if item['text'].lower() in key_hashtags:
                    hashtags_find = 1

        negative_tweets.append([int(0), text, int(hashtags_find)])

    tweets = positive_tweets + negative_tweets


    # Extract the vocabulary of keywords
    vocab = dict()
    for class_label, text1, text2 in tweets:
        for term in text1.split():
            term = term.lower()
            if len(term) > 2 and term not in stopwords:
                if term in vocab:
                    vocab[term] = vocab[term] + 1
                else:
                    vocab[term] = 1

    # Remove terms whose frequencies are less than a threshold (e.g., 15)
    vocab = {term: freq for term, freq in vocab.items() if freq > 50}

    # Generate an id (starting from 0) for each term in vocab
    vocab = {term: idx for idx, (term, freq) in enumerate(vocab.items())}
    #print(vocab)  # number of features =


    # Generate X and y
    X = []
    y = []
    for class_label, text1, text2 in tweets:
        x = [0] * (len(vocab) + 6)
        terms = [term for term in text1.split() if len(term) > 2]
        for term in terms:
            if term in vocab:
                x[vocab[term]] += 1

            if term in key_words:
                x[len(vocab)] += 1

            if term in personal_pronouns:
                x[len(vocab)+1] = 1

            if term in asthmas:
                x[len(vocab)+2] = 1

            if term in air_qualitys:
                x[len(vocab)+3] = 1

        x[len(vocab)+4] = text2
        x[len(vocab)+5] = len(terms)

        y.append(class_label)
        X.append(x)

    return X, y, vocab




# using svm and ten-fold cross-validation to train a classifier and predict the labels for text data
def svm_classification():

    X, y, vocab = old_generate_data("./data/positive_augment.txt", "./data/negative_tweets.txt")

    # 10 folder cross validation to estimate the best w and b, using SVM
    svc = svm.SVC(kernel='linear', probability=True)
    Cs = range(1, 20)
    clf = GridSearchCV(estimator=svc, param_grid=dict(C=Cs), cv=10)
    clf.fit(X, y)

    print("Model accuracy = ", clf.score(X,y))
    print("Model accuracy = ", clf.best_score_)

    # generate test data
    test_tweets = []
    for line in open("./data/sample_data.txt").readlines():
        tweet = json.loads(line)
        tweet_text = tweet["text"]
        text = tweet_text.lower().strip()
        test_tweets.append(text)

    X_test = []
    for text in test_tweets:
        x = [0] * len(vocab)
        terms = [term for term in text.split() if len(term) > 2]
        for term in terms:
            if term in vocab:
                x[vocab[term]] += 1
        X_test.append(x)

    prob = clf.predict_proba(X_test)




# find the tweets with lowest negative posibility
def find_predicted_positive():
    X, y, vocab = prepare_data("./data/positive_augment.txt", "./data/negative_tweets.txt")

    # generate test data
    test_tweets = []
    for line in open("./data/R_tweets.txt").readlines():
        tweet = json.loads(line)
        tweet_text = tweet["text"]
        text = tweet_text.lower().strip()

        entities = tweet["entities"]
        hashtag = entities["hashtags"]
        hashtags_find = 0
        if hashtag:
            for item in hashtag:
                if item['text'].lower() in key_hashtags:
                    hashtags_find = 1

        test_tweets.append([text, int(hashtags_find)])

    X_test = []
    for text1, text2 in test_tweets:
        x = [0] * (len(vocab) + 6)
        terms = [term for term in text1.split() if len(term) > 2]
        for term in terms:
            if term in vocab:
                x[vocab[term]] += 1
            if term in key_words:
                x[len(vocab)] += 1
            if term in personal_pronouns:
                x[len(vocab)+1] = 1
            if term in asthmas:
                x[len(vocab)+2] = 1
            if term in air_qualitys:
                x[len(vocab)+3] = 1

        x[len(vocab)+4] = text2
        x[len(vocab)+5] = len(terms)

        X_test.append(x)
    #print("number in test set: ", len(X_test))

    # 10 folder cross validation to estimate the best w and b, using SVM
    svc = svm.SVC(kernel='linear', probability=True)
    Cs = range(1, 20)
    clf = GridSearchCV(estimator=svc, param_grid=dict(C=Cs), cv=10)
    clf.fit(X, y)

    prob_2 = clf.predict(X_test)
    positive_list2 = []
    for i in range(len(prob_2)):
        if prob_2[i] == 1:
            positive_list2.append(i)
    print("Number predict positive : ", len(positive_list2))
    print("index of tweets predicted positive : ", positive_list2)


    prob = clf.predict_proba(X_test)[: ,1]
    prob = np.array(prob)
    sorted_list = np.argsort(-prob)
    sorted_250 = sorted_list[:250]
    print("index of tweets with least negative possibility : ", sorted_250)

    # saving first 250 tweets with least negative possibility to file positive_predicted_250.txt
    tweet_count = 0
    for line in open("./data/R_tweets.txt").readlines():
        if tweet_count in sorted_250:
            tweet = json.loads(line)
            with open("./data/positive_predicted_250.txt", 'a') as f:
                f.write(json.dumps(tweet))
                f.write('\n')

        tweet_count += 1




# compute the validation results(precision, recall, f1-score) for ten-fold cross-validation
def compute_validationMatrix():

    X, y, vocab = prepare_data("./data/positive_augment.txt", "./data/negative_tweets.txt")
    #X, y, vocab = old_generate_data("./data/positive_augment.txt", "./data/negative_tweets.txt")

    # 10 folder cross validation to estimate the best w and b, using SVM
    clf = svm.SVC(kernel='linear', C=1)
    scores = cross_val_score(clf, X, y, cv = 10, scoring = make_scorer(f1_score))
    print('f1_macro: ', "mean: ", np.mean(scores), "std: ", np.std(scores))

    scores = cross_val_score(clf, X, y, cv = 10, scoring='precision')
    print('precision: ', "mean: ", np.mean(scores), "std: ", np.std(scores))

    scores = cross_val_score(clf, X, y, cv = 10, scoring='recall')
    print('recall: ', "mean: ", np.mean(scores), "std: ", np.std(scores))





def validate_model():
    X, y, vocab = prepare_data("./data/positive_augment.txt", "./data/negative_tweets.txt")

    # 10 folder cross validation to estimate the best w and b, using SVM
    svc = svm.SVC(kernel='linear')
    Cs = range(1, 20)
    clf = GridSearchCV(estimator=svc, param_grid=dict(C=Cs), cv=10)
    clf.fit(X, y)

    y_pred = clf.predict(X)

    print("precision: ", precision_score(y, y_pred, average="macro"))
    print("Recall: ", recall_score(y, y_pred, average="macro"))
    print("f1_score: ", f1_score(y, y_pred, average="macro"))




# step 1
#svm_classification()


# step 2 find possible predicted positive tweets
find_predicted_positive()


# step 3 calculate 10-fold cross-validation validation matrix
#compute_validationMatrix()


# step 4 validate model
#validate_model()



