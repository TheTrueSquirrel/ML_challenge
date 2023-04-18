
import pandas as pd


#Loading the datasets (here you can change path to point to appropriate directory)

train = pd.read_json("/Users/jurrienboogert/Documents/DATA_SCIENCE_AND_SOCIETY/Machine Learning/assignment/train.json")
test = pd.read_json("/Users/jurrienboogert/Documents/DATA_SCIENCE_AND_SOCIETY/Machine Learning/assignment/test.json")


# TRAINING SET PREPROCESS

#Safe length of title in new variable
lengths = []

for i in range(0, len(train['title'])):
    lengths.append(len(train['title'][i]))

title_lengths = pd.DataFrame (lengths, columns = ['title_len'])
train = pd.merge(train, title_lengths, left_index=True, right_index=True)

#Safe length of abstract in new variable
lengths_a = []

for i in range(0, len(train['abstract'])):
    lengths_a.append(len(train['abstract'][i]))

abstract_lengths = pd.DataFrame (lengths_a, columns = ['abstract_len'])

train = pd.merge(train, abstract_lengths, left_index=True, right_index=True)

# Adding all information results in a higher accuracy, therefore here all variables/columns are added.
train["features"] = train['abstract'] + " " + train['title'] + " " + train['year'].astype(str) + " " + train['venue'] + " " + train['authorName'] + " " + train['title_len'].astype(str) + " " + train['abstract_len'].astype(str)


# TEST SET PREPROCESS

#Safe length of title in new variable
lengths = []

for i in range(0, len(test['title'])):
    lengths.append(len(test['title'][i]))

title_lengths = pd.DataFrame (lengths, columns = ['title_len'])
test = pd.merge(test, title_lengths, left_index=True, right_index=True)

#Safe length of abstract in new variable
lengths_a = []

for i in range(0, len(test['abstract'])):
    lengths_a.append(len(test['abstract'][i]))

abstract_lengths = pd.DataFrame (lengths_a, columns = ['abstract_len'])

test = pd.merge(test, abstract_lengths, left_index=True, right_index=True)

# Adding all information results in a higher accuracy, therefore here all variables/columns are added.
test["features"] = test['abstract'] + " " + test['title'] + " " + test['year'].astype(str) + " " + test['venue'] + " " +  test['title_len'].astype(str) + " " + test['abstract_len'].astype(str)


# Remove special characters and symbols etc. NOT removing stopwords results in higher accuracy (makes sense when thinking about writing style of individual authors, it differentiates them from others).
# We've chosen to vectorise and use the tfidtransformer in the end (pipeline), because there
# parameters can be more easily tuned with gridsearch and the output from this script doesn't result in an inmense file because of the amount of vectorised columns.
from nltk.stem import WordNetLemmatizer
#from nltk.corpus import stopwords
import re

special_character_remover = re.compile('[/(){}\[\]\|*@;:<>,.”“"]')
extra_symbol_remover = re.compile('[^0-9a-z #+_]')
#STOPWORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = special_character_remover.sub('', text)
    text = extra_symbol_remover.sub(' ', text)
    #text = ' '.join(word for word in text.split() if word not in STOPWORDS)
    text = lemmatizer.lemmatize(text,pos='v')
    return text

train['features'] = train['features'].apply(clean_text)
test['features'] = test['features'].apply(clean_text)


# Just renaming
df = train
df_test = test

X = df.features
y = df.authorId


from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.pipeline import Pipeline

#building the pipeline. Here you can change the model or change parameters by hand.

pl = Pipeline([('vect', CountVectorizer(ngram_range=(1,2))),
               ('tfidf', TfidfTransformer(norm='l2',smooth_idf=False, sublinear_tf=True)),
               ('clf', SGDClassifier(learning_rate="optimal", alpha=0.0001, penalty='l2',  n_jobs=7, random_state=3)),
              ])


# First method using train-test-split. Not used in the end.
'''
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state = 42, stratify=None)

# Training fit and predicting for train test split
pl.fit(X_train, y_train)

y_pred = pl.predict(X_test)

print(f'accuracy {accuracy_score(y_pred,y_test)}')
'''


# Cross Validation without Gridsearch to test validation accuracy on chosen model manually. Not used for model selection in the end.
'''
cv_results = cross_validate(pl, X, y, cv=3)

test_scores = cv_results['test_score']
ave_score = sum(test_scores)/len(test_scores)
print('Baseline score: ' + str(round(ave_score,3)))
'''


# Cross validation with Gridsearch included. Here you can change or add whatever parameters you want to test against eachother from Pipeline. Prepare for looooooooooooong training times :/
'''
from sklearn.model_selection import GridSearchCV

param_grid = {
    #"vect__max_df": (1.0,),
    # 'vect__max_features': (None, 5000, 10000, 50000),
    #"vect__ngram_range": ((1, 1),(1,2),(1,3)),
    #'tfidf__use_idf': (True, False),
    #'tfidf__norm': ('l1', 'l2'),
    #"clf__learning_rate": ("optimal",),
    #"clf__max_iter": (20,),
    "clf__alpha": (0.001,0.0001,.00001,0.000001),
    "clf__penalty": ("l2",'elasticnet'),
    #'tfidf__sublinear_tf': (True, False),
    #'clf__max_iter': (10, 50, 80),
}

grid = GridSearchCV(pl, param_grid, cv=3, refit=True, verbose=3)
grid.fit(X,y)

print(grid.best_estimator_)
print('best score: ' + str(round(grid.best_score_,2)))
'''


# Here using the chosen model to fit on the full trainingdata to generate the predictions on the testdata. (takes about 4.20 min)
pl.fit(X,y)

y_pred = pl.predict(df_test.features)

# Produce list of dictionairies
pred = []
dict = {}
for i in range(0,len(df_test.paperId)):
    dict['paperId'] = df_test.paperId[i]
    dict['authorId'] = str(y_pred[i])
    pred.append(dict.copy())

# WRITE PREDICTION JSON from 'pred'
import json
with open('predicted.json', 'w') as outfile:
    json.dump(pred,outfile)