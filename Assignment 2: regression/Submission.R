#create minimum absolute error
mae<-function(x,y){
  mean(abs(x-y))
}
#create minimum squared error
mse<-function(x,y){
  mean((x-y)^2)
}
setwd('/media/Almacen/data_science/homework/regression_assignment')
data<-read.csv('train_200k.csv',header=T,as.is=T)


#create different binomial labels for Location and Titles
a<-table(data$LocationNormalized)
a<-data.frame(rownames(a),a)
a<-a[,c('Var1','Freq')]
a<-a[order(a$Freq,decreasing=T),] # this is how we get the list of the Top Locations


data$London<-data$LocationNormalized=='London'
data$South_East_London<-data$LocationNormalized=='South East London'
data$The_City<-data$LocationNormalized=='The City'
data$Central_London<-data$LocationNormalized=='Central London'
data$Manchester<-data$LocationNormalized=='Manchester'
data$Leeds<-data$LocationNormalized=='Leeds'
data$Belfast<-data$LocationNormalized=='Belfast'
data$Birmingham<-data$LocationNormalized=='Birmingham'

#the position labels where chosing exploring manually the data. I couldnt make tm work. My idea was to 'zoom out' the locations, by changing the LocationRaw/Normalized for the 3rd tier of the Location Tree
data$Senior=grepl('Senior',data$Title,ignore.case=T,perl=T)
data$Manager=grepl('Manager',data$Title,ignore.case=T,perl=T)
data$Intern=grepl('Intern',data$Title,ignore.case=T,perl=T)
data$Director=grepl('Director',data$Title,ignore.case=T,perl=T)
data$Analyst=grepl('Analyst',data$Title,ignore.case=T,perl=T)
data$Entry=grepl('Entry',data$Title,ignore.case=T,perl=T)
data$SeniorDirector=grepl('Senior Director',data$Title,ignore.case=T,perl=T)
data$Engineer=grepl('Engineer',data$Title,ignore.case=T,perl=T)



#preparing test and training
n<-nrow(data)
training.idx<-sample(1:n,0.7*n) #training set will be 70% of the sample
training<-data[training.idx,]
test<-data[1-training.idx,]

model<-lm(log(SalaryNormalized)~South_East_London+The_City+Central_London+London+Manchester+Leeds+Belfast+Birmingham+Entry+Analyst+Director+Manager+Senior+SeniorDirector+Engineer+SourceName+ContractType:ContractTime+Category,data=training)
prediction<-exp(predict(model, test))
mae( test$SalaryNormalized , prediction)
mae(exp(fitted(model)),training$SalaryNormalized)


#predicting the test values
test<-read.csv('test.csv',header=T,as.is=T)
test$London<-test$LocationNormalized=='London'
test$South_East_London<-test$LocationNormalized=='South East London'
test$The_City<-test$LocationNormalized=='The City'
test$Central_London<-test$LocationNormalized=='Central London'
test$Manchester<-test$LocationNormalized=='Manchester'
test$Leeds<-test$LocationNormalized=='Leeds'
test$Belfast<-test$LocationNormalized=='Belfast'
test$Birmingham<-test$LocationNormalized=='Birmingham'
test$Senior=grepl('Senior',test$Title,ignore.case=T,perl=T)
test$Manager=grepl('Manager',test$Title,ignore.case=T,perl=T)
test$Intern=grepl('Intern',test$Title,ignore.case=T,perl=T)
test$Director=grepl('Director',test$Title,ignore.case=T,perl=T)
test$Analyst=grepl('Analyst',test$Title,ignore.case=T,perl=T)
test$Entry=grepl('Entry',test$Title,ignore.case=T,perl=T)
test$SeniorDirector=grepl('Senior Director',test$Title,ignore.case=T,perl=T)
test$Engineer=grepl('Engineer',test$Title,ignore.case=T,perl=T)
prediction<-exp(predict(model, test))
submission<-data.frame(test$Id,prediction)
write.table(submission,file="submission.csv",sep=",",row.names=F)
