from sklearn.linear_model import LogisticRegression
from sklearn import svm
import pylab as pl
import numpy as np
from sklearn import cross_validation
import json
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# all the common stopwords we may have
stopwords = ["a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the", "it's", "does", "make", "making"]


tweets = []
positive_tweets = []
negative_tweets = []
# load tweets text and add the label for each tweet
def load_tweets():
    for line in open("./data/positive_tweets.txt").readlines():
        tweet = json.loads(line)
        tweet_text = tweet["text"]
        text = tweet_text.lower().strip()
        positive_tweets.append([int(1), text])


    for line in open("./data/negative_tweets.txt").readlines():
        tweet = json.loads(line)
        tweet_text = tweet["text"]
        text = tweet_text.lower().strip()
        negative_tweets.append([int(0), text])


load_tweets()
tweets = positive_tweets + negative_tweets



# Extract the vocabulary of keywords
vocab = dict()
for class_label, text in tweets:
    for term in text.split():
        term = term.lower()
        if len(term) > 2 and term not in stopwords:
            if term in vocab:
                vocab[term] = vocab[term] + 1
            else:
                vocab[term] = 1


# Remove terms whose frequencies are less than a threshold (e.g., 15)
vocab = {term: freq for term, freq in vocab.items() if freq > 15}


# Generate an id (starting from 0) for each term in vocab
vocab = {term: idx for idx, (term, freq) in enumerate(vocab.items())}
#print(vocab)  # number of features = 80



# Generate X and y
X = []
y = []
for class_label, text in tweets:
    x = [0] * len(vocab)
    terms = [term for term in text.split() if len(term) > 2]
    for term in terms:
        if term in vocab:
            x[vocab[term]] += 1
    y.append(class_label)
    X.append(x)



# 10 folder cross validation to estimate the best w and b, using SVM
svc = svm.SVC(kernel='linear')
Cs = range(1, 20)
clf = GridSearchCV(estimator=svc, param_grid=dict(C=Cs), cv = 10)
clf.fit(X, y)

#print("Model accuracy = ", clf.score(X,y))
print("Model accuracy = ", clf.best_score_)
print("Best parameters = ", clf.best_params_)



# generate test data
test_tweets = []
for line in open("./data/test_tweets.txt").readlines():
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


y_predict = clf.predict(X_test)

print("SVM predictions: ",'\n', y_predict)


# compute the testing accuracy
n_correct = 0
# y_test is the true labels
y_test = np.concatenate((np.zeros(60) + 1, np.zeros(60)), axis= 0)
for i in range(len(y_test)):
    if y_predict[i] == y_test[i]:
        n_correct += 1

print("testing accuracy = ", float(n_correct)/len(y_test)*100, "%")





"""
# using logistic regression

svc = LogisticRegression()
Cs = range(1, 20)
clf = GridSearchCV(estimator=svc, param_grid=dict(C=Cs), cv = 10)
clf.fit(X, y)
print("Model accuracy = ", clf.best_score_)

y_predict = clf.predict(X_test)
print("Logistic predictions: ",'\n', y_predict)

"""






"""
# build neural network model
X = np.array(X)
X = X.astype(float)
y = np.array(y)
def create_baseline():
	model = Sequential()
	model.add(Dense(X.shape[1], input_dim=X.shape[1], kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model


# evaluate model with standardized dataset and output the accuracy
seed = 7
np.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, X, y, cv=kfold)
print("Standardized: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


"""




"""
# Generate X and y
X = []
y = []
for class_label, text in tweets:
    x = [0] * len(vocab)
    terms = [term for term in text.split() if len(term) > 2]
    for term in terms:
        if vocab.has_key(term):
            x[vocab[term]] += 1
    y.append(class_label)
    X.append(x)
#print("X = ",'\n', X)
#print("y = ",'\n', y)


# 10 folder cross validation to estimate the best w and b
svc = svm.SVC(kernel='linear')
Cs = range(1, 20)
clf = GridSearchCV(estimator=svc, param_grid=dict(C=Cs), cv = 10)
clf.fit(X,y)
print("Model accuracy = ", clf.score(X,y))
print("Model accuracy = ", clf.best_score_)
print("Best parameters = ", clf.best_params_)



# predict the class labels of new tweets
tweets = []
for line in open('unlabeled_tweets.txt').readlines():
    tweets.append(line)


# Generate X for testing tweets
X = []
for text in tweets:
    x = [0] * len(vocab)
    terms = [term for term in text.split() if len(term) > 2]
    for term in terms:
        if vocab.has_key(term):
            x[vocab[term]] += 1
    X.append(x)
y = clf.predict(X)


# print 100 example tweets and their class labels
for idx in range(1,100):
    print('Sentiment Class (1 means positive; 0 means negative): ', y[idx])
    print('TEXT: ', idx, tweets[idx])

print(sum(y), len(y))



"""










"""
# Find the top 10 frequency words(features)
values = vocab.values()
frequency = sorted(values, reverse= True)[:10]

dic = {}
for item in vocab:
    for i in range(len(frequency)) :
        if vocab[item] == frequency[i]:
            dic[item] = frequency[i]

vocab = dic
print("Top ten features(frequency): ",'\n', vocab)



#print "Predict unlabled tweets :", '\n', y,'\n'
with open('predicted_tweets.txt','w') as f:
    idx=0
    for tweet in tweets:
        f.write(str(y[idx])+" ,"+ tweet)
        idx +=1




# compute the testing accuracy
n_correct = 0
for i in range(len(y_test)):
    if y_predict[i] == y_test[i]:
        n_correct += 1

print("testing accuracy = ", float(n_correct)/len(y_test))


"""

