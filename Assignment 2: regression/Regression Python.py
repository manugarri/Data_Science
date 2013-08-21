import pandas as pd
import numpy as np
import gc
from nltk import word_tokenize  #to stem
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer


def mae(x,y):
    error=0
    n=x.shape[0]
    for x,y in zip(x,y):
        error+=fabs(x-y)
    return error/n

def remove_dummies(dummydf,reference,n_features):
	col_remove=[]
	ref=reference.value_counts().sort_index(1).index[n_features:] # we want to only get the column top 100 uniques, so we will remove the rest
	for col in np.array(dummydf.columns):
		for i in ref:
			if col==i:
				col_remove.append(col)
				
	dummydf_reduced=dummydf.drop(col_remove,axis=1)
	return dummydf_reduced


class LemmaTokenizer(object): #tokenizer for CountVectorizer for stemming using Wordnet Corpora
      def __init__(self):
            self.wnl = WordNetLemmatizer()  
      def __call__(self, doc):  
            return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]  


#data = pd.read_csv('train.csv')
data = pd.read_csv('train_200k.csv',usecols=[2,4,8,10,11]) #this way we free memory

n_sources=30
n_loc=10
n_features=500

vect=TfidfVectorizer(tokenizer=LemmaTokenizer(),min_df=3,stop_words='english', max_features=n_features)
) #play with max_features
desc_vect=vect.fit_transform(data.FullDescription)
desc_vect=pd.DataFrame(desc_vect.todense())


#sourcename dummy variables. We'll get the top ones
sourcename=pd.get_dummies(data.SourceName)
sourcename=remove_dummies(sourcename,data.SourceName,n_sources)
#category dummy variables
categories=pd.get_dummies(data.Category)
#Locatio dummies
location=pd.get_dummies(data.LocationNormalized)
location=remove_dummies(location,data.LocationNormalized,n_loc)






data=desc_vect.join(sourcename)
data=data.join(categories)
data=data.join(location)

#free memory
del desc_vect
del sourcename
del categories
del location
gc.collect() #free memory

#load the Salary Normalized
Salaries = pd.read_csv('train_200k.csv',usecols=[10])

#====Train Test Split=============================================================
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(train,data2.SalaryNormalized)
model=LinearRegression()
model.fit(x_train,y_train)
prediction=model.predict(x_test)
mae(prediction,y_test)
#===================================================================================


from sklearn.metrics import mean_absolute_error

from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import cross_val_score
cross_val_score(LinearRegression(),data,np.array(Salaries.SalaryNormalized),score_func=mean_absolute_error).mean()



n_sources=30
n_loc=10
n_features=500

#MAE  8925.96


vect=TfidfVectorizer(tokenizer=LemmaTokenizer(),min_df=3,stop_words='english', max_features=n_features)
n_sources=20
n_loc=20
n_features=500
#mae 9000

vect=TfidfVectorizer(tokenizer=LemmaTokenizer(),min_df=3,stop_words='english', max_features=n_features, ngram_range=(1,2))
n_sources=20
n_loc=10
n_features=500