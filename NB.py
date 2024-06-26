from os import listdir

new_folder_list  = []

# creates a list of all the folders in the "20_newsgroups" directory and appends them to the new_folder_list.
for folder  in listdir("20_newsgroups"):
    new_folder_list .append(folder)

all_files = []
from os.path import join

# The code below creates a list of all the files in each of the folders in the "20_newsgroups" directory and appends them to the all_files list.
for folder_name in new_folder_list :
    folder_path = join("20_newsgroups", folder_name)
    subfiles = []
    for f in listdir(folder_path):
        subfiles.append(f)
    all_files.append(subfiles)

from os.path import join

# The code below creates a list of full file paths for each file in the "20_newsgroups" directory by joining the folder name and the file name using os.path.join.
full_path_list = [join("20_newsgroups", join(new_folder_list [fo], fi)) for fo in range(len(new_folder_list )) for fi in all_files[fo]]

import itertools
label_list = list(itertools.chain.from_iterable([itertools.repeat(folder_name, len(listdir(join("20_newsgroups", folder_name)))) for folder_name in new_folder_list ]))

from sklearn.model_selection import train_test_split

# The code below splits the full_path_list into two sets, one for training and one for testing. It also creates a list of labels for the training and testing sets.
train_docs, test_docs, Y_train, Y_test = train_test_split(full_path_list, label_list, random_state=0, test_size=0.5)
Y_test = np.asarray(Y_test)
train_label_set = np.asarray(Y_train)

import string

# The clean function removes punctuation and words less than three letters long from a list of words.
def clean(words):
    table = str.maketrans('', '', '\t')
    words = [word.translate(table) for word in words]

    punctuations = (string.punctuation).replace("'", "")
    trans_table = str.maketrans('', '', punctuations)
    stripped_words = [word.translate(trans_table) for word in words]

    words_lst = []
    for word in stripped_words:
        if len(word) > 2 and word.isalpha():
            word = word.strip("'")
            words_lst.append(word.lower())

    return words_lst

from nltk.corpus import stopwords
stop_words_lst = stopwords.words('english')

# The split_words_sent function splits a sentence into words, removes stopwords and calls the clean function to clean up the words.
def split_words_sent(line):
    words = line[0:len(line)-1].strip().split(" ")
    words = clean(words)
    words = list(filter(lambda x: x not in stop_words_lst, words))  
    return words

# The remove_meta function removes any metadata at the beginning of a document.
def remove_meta(lines):
    start = 0
    for i, line in enumerate(lines):
        if line == '\n':
            start = i + 1
            break
    return list(filter(lambda x: x != '\n', lines[start:]))

# The split_words function splits a document into sentences and calls the split_words_sent function to split each sentence into words.
def split_words(path):
    with open(path, 'r') as f:
        text_lines = f.readlines()
    text_lines = remove_meta(text_lines)
    
    doc_words = []
    
    for line in text_lines:
        doc_words.append(split_words_sent(line))

    return doc_words

# Define a function to flatten a nested list
def flat_list(list):
    return [j for i in list for j in i]


list_of_words = []

# Loop through each document in train_docs, split the words and add them to list_of_words
for document in train_docs:    
    t = split_words(document)
    list_of_words.append(flat_list(t))

import numpy as np
np_words_lst = np.asarray(flat_list(list_of_words))

words, counts = np.unique(np_words_lst, return_counts=True)

# Sort the words and their frequencies in descending order
word_freq_lst, words_lst = zip(*(sorted(zip(counts, words), reverse=True)))
word_freq_lst, words_lst = list(word_freq_lst), list(words_lst)


feature_lst = words_lst[0:2500]


word_dict = {}
doc_num = 1

# Loop through each document in list_of_words, convert it to a numpy array and get unique words and their frequencies
for doc_words in list_of_words:
    np_doc_words = np.asarray(doc_words)
    w, c = np.unique(np_doc_words, return_counts=True)
    word_dict[doc_num] = {w[i]: c[i] for i in range(len(w))}
    doc_num += 1

# Create a numpy array called train_feature_set to store the frequencies of the top 2500 most frequent words in each document
train_feature_set = np.asarray([[word_dict[k].get(f, 0) for f in feature_lst] for k in word_dict.keys()])


test_words_lst = []

for document in test_docs:
        test_words_lst.append(flat_list(split_words(document)))

word_dict_test = {}
doc_num = 1

# Loop through each document in test_words_lst, convert it to a numpy array and get unique words and their frequencies
for doc_words in test_words_lst:
    np_doc_words = np.asarray(doc_words)
    w, c = np.unique(np_doc_words, return_counts=True)
    word_dict_test[doc_num] = {w[i]: c[i] for i in range(len(w))}
    doc_num += 1

X_test = np.asarray([[word_dict_test[k].get(f, 0) for f in feature_lst] for k in word_dict_test.keys()])


# Define a Naive Bayes model class
class NB_Model:
    # Initialize the NB_Model class with a hyperparameter of 1.0
    def __init__(self, hyper_parameter=1.0):
        self.hyper_parameter = hyper_parameter
        self.classes_ = None
        self.class_count_new = None
        self.feat_count = None
        self.class_prior_new = None
        self.feature_prob = None

    # Define a fit method for the NB_Model class that trains the model on the given data X and labels y
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_feature_lst = X.shape[1]
        self.class_count_new = np.zeros(n_classes, dtype=int)
        self.feat_count = np.zeros((n_classes, n_feature_lst), dtype=int)
        self.class_prior_new = np.zeros(n_classes, dtype=np.float64)
        self.feature_prob = np.zeros((n_classes, n_feature_lst), dtype=np.float64)

        for i, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.class_count_new[i] = X_c.shape[0]
            self.feat_count[i] = np.sum(X_c, axis=0)
            self.class_prior_new[i] = (self.hyper_parameter + self.class_count_new[i]) / (n_classes * self.hyper_parameter + X.shape[0])
            self.feature_prob[i] = np.log((self.feat_count[i] + self.hyper_parameter) / (np.sum(self.feat_count[i]) + self.hyper_parameter * n_feature_lst))

        return self

    # Define a predict method for the NB_Model class that predicts the class labels for the given data X
    def predict(self, X):
        return self.classes_[np.argmax(self.plp(X), axis=1)]

    # Define a predict_log_proba method for the NB_Model class that predicts the class probabilities for the given data X
    def plp(self, X):
        return np.dot(X, self.feature_prob.T) + np.log(self.class_prior_new)

    # Define a score method for the NB_Model class that calculates the accuracy of the model on the given data X and labels y
    def score(self, X, y):
        return np.mean(self.predict(X) == y)
        

model = NB_Model()
model.fit(train_feature_set, train_label_set)
model.predict(X_test)
accuracy = model.score(X_test, Y_test)*100

print("Accuracy of the Navie Bayes Model is {:.2f}".format(accuracy),"%")










