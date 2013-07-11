from sklearn.cross_validation import train_test_split
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
from nltk import word_tokenize  #to stem
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import train_test_split
from sklearn.metrics import auc_score

class LemmaTokenizer(object): #tokenizer for CountVectorizer for stemming using Wordnet Corpora
      def __init__(self):
            self.wnl = WordNetLemmatizer()  
      def __call__(self, doc):  
            return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]    
      
train = read_csv('/media/Almacen/data_science/homework/Logistic Regresion Assignment/train-utf8.csv')

vect = CountVectorizer(tokenizer=LemmaTokenizer(),stop_words='english', min_df=1,ngram_range=(1,3))

train2=vect.fit_transform(train.Comment)

x_train,x_test=train_test_split(train,random_state=42) #random state to ensure same split
x_train2,x_test2=train_test_split(train2,random_state=42)

x_train2=x_train2.tocoo() # it seems that Multinomial works with COO matrix but not CSR which are the ones spit out by train_test_split
x_test2=x_test2.tocoo()

classifier = MultinomialNB(fit_prior=True).fit(x_train2,x_train[:,0])
auc_score(x_train[:,0],classifier.predict(x_train2))  #Training AUC of 0.99
auc_score(x_test[:,0],classifier.predict(x_test2)) #OOS AUC of 0.65


#find the best set of parameters for Countvectorizer


from pylab import plot,show

def iterate_vectorizer(initial_value,iterations,vect_string):
	auc_training=[]
	auc_oos=[]
	dfs=range(initial_value,iterations)
	print dfs
	for n in dfs:
		print n
		vect = eval(vect_string % n)
		train2=vect.fit_transform(train.Comment)
		x_train2,x_test2=train_test_split(train2,random_state=42)
		x_train2=x_train2.tocoo() 
		x_test2=x_test2.tocoo()
		classifier = MultinomialNB(fit_prior=True).fit(x_train2,x_train[:,0]) #the third term has to be a list, or an array)
		auc_training.append(auc_score(x_train[:,0],classifier.predict(x_train2)))
		auc_oos.append(auc_score(x_test[:,0],classifier.predict(x_test2)))
	results= zip(dfs,auc_training,auc_oos)
	print results
	return auc_plot(results)
	return results
	
def auc_plot(results):
	res_df=pd.DataFrame(results,columns=('a','b','c'))
	plot(res_df.a, res_df.b, '-b', label='Training AUC')
	plot(res_df.a, res_df.c, '-r', label='OOS AUC')
	show()






pd.DataFrame([(y,x) for x,y,z in results]).plot()
show()

#finding the optimal min_df
vect_string = "CountVectorizer(tokenizer=LemmaTokenizer(),min_df =%d,stop_words='english',strip_accents='unicode',lowercase=True)"
iterations=20
initial_value=0
results=iterate_vectorizer(initial_value,iterations,vect_string)

#min_df=5 seems to be the best min

#lets see the max_df
vect_string = "CountVectorizer(tokenizer=LemmaTokenizer(),min_df =5,max_df=%d,stop_words='english',strip_accents='unicode',lowercase=True)"
iterations=250
initial_value=5
results=iterate_vectorizer(initial_value,iterations,vect_string)



#optimal max_df = 158 so i wont include the parameter

#lets see the ngrams
vect_string = "CountVectorizer(tokenizer=LemmaTokenizer(),ngram_range=(1,%d),min_df =5,stop_words='english',strip_accents='unicode',lowercase=True)"
iterations=6
initial_value=1
results=iterate_vectorizer(initial_value,iterations,vect_string)
#best result with ngrams (1,2)


vect = CountVectorizer(tokenizer=LemmaTokenizer(),ngram_range=(1,2),min_df =5,stop_words='english',strip_accents='unicode',lowercase=True)

#lets find the optimal alpha
def iterate_Multinomial_alpha(vect):
	auc_training=[]
	auc_oos=[]
	dfs=np.arange(0,3,0.1)
	print dfs
	for n in dfs:
		print n
		train2=vect.fit_transform(train.Comment)
		x_train2,x_test2=train_test_split(train2,random_state=42)
		x_train2=x_train2.tocoo() 
		x_test2=x_test2.tocoo()
		classifier = MultinomialNB(fit_prior=True, alpha=n).fit(x_train2,x_train[:,0]) 
		auc_training.append(auc_score(x_train[:,0],classifier.predict(x_train2)))
		auc_oos.append(auc_score(x_test[:,0],classifier.predict(x_test2)))
	results= zip(dfs,auc_training,auc_oos)
	print results
	return auc_plot(results)
	
# alpha = 1 (auto) seems to be the best.

#produce the final submission

classifier = MultinomialNB(fit_prior=True) 
vect = CountVectorizer(tokenizer=LemmaTokenizer(),ngram_range=(1,2),min_df =5,stop_words='english',strip_accents='unicode',lowercase=True)
train = read_csv('/media/Almacen/data_science/homework/Logistic Regresion Assignment/train-utf8.csv')
test = read_csv('/media/Almacen/data_science/homework/Logistic Regresion Assignment/test-utf8.csv')
X_train = vect.fit_transform(train.Comment)
X_test = vect.transform(test.Comment)

model=classifier.fit(X_train, list(train.Insult))

predictions=classifier.predict.proba(x_test)[:,1]
submission = pd.DataFrame({'id': test.id, 'insult': predictions})
submission.to_csv('MNBsubmission.csv', index=False)
