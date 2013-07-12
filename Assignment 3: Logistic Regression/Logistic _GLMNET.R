#get the 100 most common words in the insult comments
setwd('/media/Almacen/data_science/homework//Logistic Regresion Assignment')
train<-read.csv('train-utf8.csv',header=T,as.is=T)
library('tm')

#histograms of insults/noninsults. Maybe there is a pattern in the hours/days/months?
train$Date<-gsub('Z','',train$Date)
train$Date<-strptime(train$Date, '%y%m%d%H%M%S')
hist(as.numeric(strftime(train$Date,'%H')))
hist(as.numeric(strftime(train$Date[train$Insult==1],'%H')))
hist(as.numeric(strftime(train$Date[train$Insult==0],'%H')))
library('ggplot2')
ggplot(train,aes(x=as.numeric(strftime(train$Date,'%H')),fill=Insult))+geom_histogram()+facet_wrap(~Insult) #not an obvious difference in terms of hours might be because the data is only from a month.
ggplot(train,aes(x=strftime(train$Date,'%u'),fill=Insult))+geom_histogram()+facet_wrap(~Insult)


#we'll try to get a model using a document matrix of common words in insults that arent common words in noninsults
corpus_insults<-Corpus(VectorSource(train$Comment[train$Insult==1]))
corpus_insults <- tm_map(corpus_insults, tolower)
corpus_insults <- tm_map(corpus_insults, removePunctuation)
corpus_insults <- tm_map(corpus_insults, removeNumbers)
corpus_insults <- tm_map(corpus_insults, removeWords, stopwords('english'))
myDtm_insults <- TermDocumentMatrix(corpus_insults, control = list(minWordLength = 3,maxWordLength=10))
#words_insults<-sort(rowSums(as.matrix(myDtm_insults)), decreasing=TRUE)[1:1000]
#we do the same with not insults
corpus_notinsults<-Corpus(VectorSource(train$Comment[train$Insult==0]))
corpus_notinsults <- tm_map(corpus_notinsults, tolower)
corpus_notinsults <- tm_map(corpus_notinsults, removePunctuation)
corpus_notinsults <- tm_map(corpus_notinsults, removeNumbers)
corpus_notinsults <- tm_map(corpus_notinsults, removeWords, stopwords('english'))
myDtm_notinsults <- TermDocumentMatrix(corpus_notinsults, control = list(minWordLength = 3,maxWordLength=10))
#words_notinsults<-sort(rowSums(as.matrix(myDtm_notinsults)), decreasing=TRUE)[1:1000]
#get the unique insult words dictionary
y<-findFreqTerms(myDtm_insults, lowfreq=5)
n<-findFreqTerms(myDtm_notinsults, lowfreq=5)
words_unique<-y[!(y %in% n)]   #list of common words in insults not in notinsults
words_unique<-Dictionary(words_unique) #dictionary of common words in insults not in notinsults
rm(list=c('corpus_notinsults','corpus_insults','myDtm_insults','myDtm_notinsults','y','n'))



#add an extra variable, length of comments
i=1
wordcount<-function(x){
  str2 <- gsub(' {2,}',' ',x)
  train$length[i]<<-length(strsplit(str2,' ')[[1]])
  i<<-i+1
}
sapply(train$Comment,wordcount)

#add another field, number of punctuation signs
library('stringr')
i=1
punctuationcount<-function(x){
  str<-str_replace_all(x, "[^[:punct:]]", " ")
  str2 <- gsub(' {2,}',' ',str)
  train$punctuation[i]<<-length(strsplit(str2,' ')[[1]])-1
  i<<-i+1
}
sapply(train$Comment,punctuationcount)


#Lets add another field with a count of the most common insults: 
insults<-c('ass','asshole','bitch','bitches','bullshit','coward','crap','cunt','damn','dead','dick','dirty','disgusting','dumb','faggot','fake','fat','fool','fuck','gay','hate','idiot','stupid','fucking','dumb','shit','moron','dick','loser','pathetic','gay','racist','lol','troll','nigga','fool','suck','hate','sick','fuckin','hell','retard','retarded','pussy','kill','die','god','idiots','asshole','faggot','ill','fake','ugly','crap','coward','idiotic','trash','dead','bullshit','fucked','morons','cock','filthy','fucker','scum','sucking','trolling','balls','bastard','moronic','sucker','dumbass','maggot','nazi','bigot','dickhead','fools','miserable','niggas','slut','stfu','commie','fucks','jerk','losers','motherfucker','prick','rape','rapist','redneck','retards','whiny','despicable','fck','fcking','freak','freaks','fuckers','hoe','imbecile','jackass','negro','puke','sicko')
i=1
icount<-vector()
insultcounter<-function(x){
  str2 <- gsub(' {2,}',' ',x)
  words<-(strsplit(str2,' ')[[1]])
  insultcount<-0
  for(word in words){
    for( insult in insults){
      if (word==insult){
        insultcount<-insultcount+1
      }
    }
  }
  icount[i]<<-insultcount
  i<<-i+1
}
sapply(train$Comment,insultcounter)
train$insult_count<-icount
rm(list=c('i','icount'))
==================================================================================================================****************
#save the auxiliary train file and reopen to clean memory
write.csv(train,file='train_aux.csv',row.names=F)
train<-read.csv('train_aux.csv',header=T,as.is=T)


#Lets do a cross validation with a train/test split

n<-nrow(train)
train.idx<-sample(1:n,0.7*n)
train.split<-train[train.idx,]
test.split<-train[-train.idx,]
corpus<-Corpus(VectorSource(train.split$Comment))
myDtm <- TermDocumentMatrix(corpus, control = list(dictionary=words_unique,tolower=T,stopwords('english'),removePunctuation=T,removeNumbers=T))
matrix.train<-(t(as.data.frame(inspect((myDtm)))))
matrix.train<-(as.data.frame(matrix.train))
matrix.train$length<-train.split$length
matrix.train$punctuation<-train.split$punctuation
matrix.train$insult_count<-train.split$insult_count
matrix.train$Insult<-train.split$Insult
var<-names(matrix.train[1:ncol(matrix.train)])
f<-as.formula(paste(var[ncol(matrix.train)], "~", paste(var[1:ncol(matrix.train)-1], collapse="+")))
#glmnet
model.reg <- glmnet( as.matrix(matrix[,c(1:59)]), as.matrix(matrix[60]) )
model.reg <- cv.glmnet( as.matrix(matrix[,c(1:59)]), as.matrix(matrix[60]) )
train.predict.regcv <- predict(model.reg, as.matrix(matrix.train[,c(1:59)]), s=model.regcv$lambda.min,type='response')

#glm
model<-glm(f,data=matrix.train,family='binomial')
prediction<-(predict(model,matrix.train,type='response'))
library('ROCR')
performance(prediction(train.predict.regcv,matrix.train$Insult),measure='auc') #glmnet training AUC of 0.757
performance(prediction(prediction,matrix.train$Insult),measure='auc') #glm training AUC of 0.763


#OOS AUC
corpus<-Corpus(VectorSource(test.split$Comment))
myDtm <- TermDocumentMatrix(corpus, control = list(dictionary=words_unique,tolower=T,stopwords('english'),removePunctuation=T,removeNumbers=T))
matrix.test<-(t(as.data.frame(inspect((myDtm)))))
matrix.test<-(as.data.frame(matrix.test))
matrix.test$length<-test.split$length
matrix.test$punctuation<-test.split$punctuation
matrix.test$insult_count<-test.split$insult_count
matrix.test$Insult<-test.split$Insult

#glmnet
test.predict.regcv <- predict(model.reg, as.matrix(matrix.test[,c(1:59)]), s=model.regcv$lambda.min,type='response')
#glm
prediction<-(predict(model,matrix.test,type='response'))

performance(prediction(test.predict.regcv,matrix.test$Insult),measure='auc') #glmnet OOS of 0.732
performance(prediction(prediction,matrix.test$Insult),measure='auc') #glm OOS AUC of 0.735




#Create final Logistic Model submission
test<-read.csv('test-utf8.csv',header=T,as.is=T)
wordcount_test<-function(x){
  str2 <- gsub(' {2,}',' ',x)
  test$length[i]<<-length(strsplit(str2,' ')[[1]])
  i<<-i+1
}
punctuationcount_test<-function(x){
  str<-str_replace_all(x, "[^[:punct:]]", " ")
  str2 <- gsub(' {2,}',' ',str)
  test$punctuation[i]<<-length(strsplit(str2,' ')[[1]])-1
  i<<-i+1
}
i=1
sapply(test$Comment,wordcount_test)
i=1
sapply(test$Comment,punctuationcount_test)
i=1
icount<-vector()
sapply(test$Comment,insultcounter)
test$insult_count<-icount

corpus<-Corpus(VectorSource(train$Comment))
myDtm <- TermDocumentMatrix(corpus, control = list(dictionary=words_unique,tolower=T,stopwords('english'),removePunctuation=T,removeNumbers=T))
matrix.train<-(t(as.data.frame(inspect((myDtm)))))
matrix.train<-(as.data.frame(matrix.train))
matrix.train$length<-train$length
matrix.train$punctuation<-train$punctuation
matrix.train$insult_count<-train$insult_count
matrix.train$Insult<-train$Insult
corpus<-Corpus(VectorSource(test$Comment))
myDtm <- TermDocumentMatrix(corpus, control = list(dictionary=words_unique,tolower=T,stopwords('english'),removePunctuation=T,removeNumbers=T))
matrix.test<-(t(as.data.frame(inspect((myDtm)))))
matrix.test<-(as.data.frame(matrix.test))
matrix.test$length<-test$length
matrix.test$punctuation<-test$punctuation
matrix.test$insult_count<-test$insult_count

model<-glm(f,data=matrix.train,family='binomial')
prediction<-(predict(model,matrix.test,type='response'))
submission_logistic<-data.frame(test$id,prediction)
write.csv(submission_logistic,file='submission_logistic.csv',row.names=F)


#attempt 2 : glmnet
library('glmnet')

model.reg <- cv.glmnet( as.matrix(matrix.train[,c(1:59)]), as.matrix(matrix.train[60]) , family='binomial' )
train.predict.regcv <- predict(model.reg, as.matrix(matrix.test[,c(1:59)]), s=model.reg$lambda.min,type='response')
submission_logistic<-data.frame(test$id,train.predict.regcv)
write.csv(submission_logistic,file='submission_logistic_glmnet.csv',row.names=F)
