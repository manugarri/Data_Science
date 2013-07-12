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
from sklearn.cross_validation import cross_val_score

class LemmaTokenizer(object): #tokenizer for CountVectorizer for stemming using Wordnet Corpora
      def __init__(self):
            self.wnl = WordNetLemmatizer()  
      def __call__(self, doc):  
            return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]    
      
train = read_csv('/media/Almacen/data_science/homework/Logistic Regresion Assignment/train-utf8.csv')


vect = CountVectorizer(tokenizer=LemmaTokenizer(),stop_words='english', min_df=1,ngram_range=(1,3))

model= MultinomialNB()

def iterate_vectorizer(initial_value,final_value,step,vect,model):
	cross_score=[]
	dfs=np.arange(initial_value,final_value,step)
	print dfs
	for n in dfs:
		print n
		vect = eval(vect % n)
		train2=vect.fit_transform(train.Comment)
		cross_score.append(cross_val_score(model,train2.toarray(),train.Insult).mean())		
	results= cross_score
	print results
	return auc_plot(results)
	

def auc_plot(results):
	res_df=pd.DataFrame(results,columns=('a','b','c'))
	plot(res_df.a, res_df.b, '-b', label='Training AUC')
	plot(res_df.a, res_df.c, '-r', label='OOS AUC')
	show()

vect = "CountVectorizer(tokenizer=LemmaTokenizer(),stop_words='english', min_df=%d)"
initial_value=1
final_value=7
step=1
model= MultinomialNB()
iterate_vectorizer(initial_value,final_value,step,vect,model)



model= MultinomialNB()
vect = CountVectorizer(stop_words='english', min_df=1,ngram_range=(1,3))
train2=vect.fit_transform(train.Comment)
cross_val_score(model,train2.toarray(),train.Insult,score_func=auc_score).mean()
#0.6669

model= MultinomialNB(fit_prior=True)
vect = CountVectorizer(stop_words='english', min_df=1,ngram_range=(1,3))
train2=vect.fit_transform(train.Comment)
cross_val_score(model,train2.toarray(),train.Insult,score_func=auc_score).mean()
#0.6669

model= MultinomialNB(fit_prior=True)
vect = CountVectorizer(tokenizer=LemmaTokenizer(),stop_words='english', min_df=1,ngram_range=(1,3))
train2=vect.fit_transform(train.Comment)
cross_val_score(model,train2.toarray(),train.Insult,score_func=auc_score).mean()
#0.6849

model= MultinomialNB(fit_prior=True)
vect = CountVectorizer(tokenizer=LemmaTokenizer(), min_df=1)
train2=vect.fit_transform(train.Comment)
cross_val_score(model,train2.toarray(),train.Insult,score_func=auc_score).mean()
#0.7003

model= MultinomialNB(fit_prior=True)
vect = CountVectorizer(tokenizer=LemmaTokenizer(), min_df=2)
train2=vect.fit_transform(train.Comment)
cross_val_score(model,train2.toarray(),train.Insult,score_func=auc_score).mean()
#0.7783

model= MultinomialNB(fit_prior=True)
vect = CountVectorizer(tokenizer=LemmaTokenizer(), min_df=3)
train2=vect.fit_transform(train.Comment)
cross_val_score(model,train2.toarray(),train.Insult,score_func=auc_score).mean()
#0.7829

model= MultinomialNB(fit_prior=True)
vect = CountVectorizer(tokenizer=LemmaTokenizer(), min_df=2, ngram_range=(1,2))
train2=vect.fit_transform(train.Comment)
cross_val_score(model,train2.toarray(),train.Insult,score_func=auc_score).mean()
#0.7792

model= MultinomialNB(fit_prior=True)
vect = CountVectorizer(tokenizer=LemmaTokenizer(), min_df=3, ngram_range=(1,2))
train2=vect.fit_transform(train.Comment)
cross_val_score(model,train2.toarray(),train.Insult,score_func=auc_score).mean()
#0.7863

model= MultinomialNB(fit_prior=True)
vect = CountVectorizer(tokenizer=LemmaTokenizer(), min_df=3, ngram_range=(1,2),strip_accents='unicode')
train2=vect.fit_transform(train.Comment)
cross_val_score(model,train2.toarray(),train.Insult,score_func=auc_score).mean()
#0.7868


model= MultinomialNB(fit_prior=True)
vect = CountVectorizer(tokenizer=LemmaTokenizer(), min_df=3, ngram_range=(1,2),strip_accents='unicode')
train2=vect.fit_transform(train.Comment)
cross_val_score(model,train2.toarray(),train.Insult,score_func=auc_score).mean()
#0.7868


#trying TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
model= MultinomialNB(fit_prior=True)
vect = TfidfVectorizer(tokenizer=LemmaTokenizer(), min_df=3, ngram_range=(1,2),strip_accents='unicode')
train2=vect.fit_transform(train.Comment)
cross_val_score(model,train2.toarray(),train.Insult,score_func=auc_score).mean()
#0.5614


model= MultinomialNB(fit_prior=True)
vect = TfidfVectorizer(tokenizer=LemmaTokenizer(), min_df=3, ngram_range=(1,2),strip_accents='unicode', binary= True)
train2=vect.fit_transform(train.Comment)
cross_val_score(model,train2.toarray(),train.Insult,score_func=auc_score).mean()
#0.5614


#trying HashingVectorizer

from sklearn.feature_extraction.text import HashingVectorizer
model= MultinomialNB(fit_prior=True)
vect = HashingVectorizer(tokenizer=LemmaTokenizer(), ngram_range=(1,2),strip_accents='unicode')
train2=vect.fit_transform(train.Comment)
cross_val_score(model,train2.toarray(),train.Insult,score_func=auc_score).mean()

#produce the final submission

classifier = MultinomialNB(fit_prior=True) 
vect = CountVectorizer(tokenizer=LemmaTokenizer(), min_df=3, ngram_range=(1,2),strip_accents='unicode')
train = read_csv('/media/Almacen/data_science/homework/Logistic Regresion Assignment/train-utf8.csv')
test = read_csv('/media/Almacen/data_science/homework/Logistic Regresion Assignment/test-utf8.csv')
X_train = vect.fit_transform(train.Comment)
X_test = vect.transform(test.Comment)

model=classifier.fit(X_train, list(train.Insult))

predictions=model.predict_proba(X_test)[:,1]
submission = pd.DataFrame({'id': test.id, 'insult': predictions})
submission.to_csv('MNBsubmission 2.csv', index=False)
