## Imports
import pandas as pd
import numpy as np
from collections import Counter
import nltk
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
import random

# Immutables
file = 'unarXive_sample/context_centered_sample/citation_context_sample.csv'
punctuations = list(string.punctuation)
keywords = ['CIT', 'MAINCIT', 'REF', 'FORMULA']
filtr = punctuations + keywords

# Functions
def readCitations(file, ret='all'):
    ''' Open csv with citations, choose what to return:
            - a dict with all citation contexts for every cited paper
            - the corresponding arxiv and mag ids
            - the csv as dataframe
            - all three of the above
        Based on MAG ID since arXiv ID seems to be less complete
    '''
    columns = ['cited_paper_mag_id',
               'adjacent_citations_mag_ids',
               'citing_paper_mag_id',
               'cited_paper_arxiv_id',
               'adjacent_citations_arxiv_ids',
               'citing_paper_arxiv_id',
               'citation_context']

    cit = pd.read_csv(file, sep='\u241E', encoding='utf-8', engine='python', names=columns)

    keys = list(set(cit['cited_paper_mag_id']))
    context_dict = {k:[] for k in keys}
    mag2arx_dict = {k:None for k in keys}

    for i in range(len(cit)):
        # Main vars -- Later add adjacent citations?
        mag = cit['cited_paper_mag_id'][i]
        arx = cit['cited_paper_arxiv_id'][i]
        citcon = cit['citation_context'][i]

        # Add context to dictionary
        temp_list = context_dict[mag]
        temp_list.append(citcon)
        context_dict[mag] = temp_list

        # Add arXiv ID if available
        mag2arx_dict[mag] = arx

    if ret == 'cit':
        return cit
    elif ret == 'con':
        return context_dict
    elif ret == 'm2a':
        return mag2arx_dict
    else:
        return cit, context_dict, mag2arx_dict

def splitTrainTest(cit, common_range=10, test_size=100):
    ''' Split the dataset in test and training data, using the most common cited
        articles to try and evaluate as testset. Number of most common and size of
        testset can be determined manually defaults 10 and 100
        returns:
        test_cit  - dataframe with citations for classification
        train_cit - dataframe with citations for training
    '''
    common_counter = Counter(cit['cited_paper_mag_id']).most_common()
    most_common = [common_counter[i][0] for i in range(common_range)]
    mc_index = []
    for mc in most_common:
        for i in cit.loc[cit['cited_paper_mag_id'] == mc].index:
            mc_index.append(i)

    # Check correct size of testset.
    if len(mc_index) < test_size:
        print('Invalid test size for entered range')
        return 0,0

    test_index = list(random.sample(mc_index, test_size))
    test_cit = cit.iloc[test_index]
    train_index = list(set(range(len(cit))) - set(test_index))
    train_cit = cit.iloc[train_index]

    return train_cit, test_cit

def vocabFilter(cit):
    ''' Filter out punctuation and special characters
    '''
    ## Filter out punctuation and weird words
    citcon_ftd = [ [word.lower() for word in nltk.word_tokenize(ct) if word not in filtr]
                  for ct in cit['citation_context']
                 ]
    ## Corpus with all sentences
    citcon_sent = [' '.join(citcon) for citcon in citcon_ftd]

    # Vocabulary of the sentences in cit
    vocab = []
    for ct in citcon_sent:
        for word in nltk.word_tokenize(ct):
            if word not in filtr and word not in vocab:
                vocab.append(word)

    return citcon_sent, vocab

def bowVector(corpus, vocab):
    ''' Create bag of words model with CountVectorizer, needs a corpus
        returns:
        X         - the fitted model
        wordvec   - the word vector
        vec_vocab - vocabulary of the model
    '''
    vectorizer = CountVectorizer(vocabulary=vocab)
    X = vectorizer.fit_transform(corpus)
    wordvec = X.toarray()
    vec_vocab = vectorizer.get_feature_names_out()

    return X, wordvec, vec_vocab

if __name__=='__main__':
    print('Starting Pipeline...')
    ## Data prep
    # Run the first part of the pipeline
    cit, context_dict, mag2arx_dict = readCitations(file)
    train, test = splitTrainTest(cit)
    train_sent, train_vocab = vocabFilter(train)
    test_sent, _ = vocabFilter(test)
    X, wordvec, vec_vocab = bowVector(train_sent, train_vocab)
    y = train['cited_paper_mag_id']
    X_test, wordvec_test, vocab_test = bowVector(test_sent, train_vocab)
    y_test = list(test['cited_paper_mag_id'])

    ## SVM classifier
    clf = svm.SVC()
    clf.fit(X,y)

    res = clf.predict(X_test)
    results = Counter([res[i] == y_test[i] for i in range(len(res))])
    print(results)
