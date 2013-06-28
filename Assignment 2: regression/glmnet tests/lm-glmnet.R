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
data$London<-data$LocationNormalized=='London'
data$South_East_London<-data$LocationNormalized=='South East London'
data$The_City<-data$LocationNormalized=='The City'
data$Central_London<-data$LocationNormalized=='Central London'
data$Manchester<-data$LocationNormalized=='Manchester'
data$Leeds<-data$LocationNormalized=='Leeds'
data$Belfast<-data$LocationNormalized=='Belfast'
data$Birmingham<-data$LocationNormalized=='Birmingham'
data$Senior=grepl('Senior',data$Title,ignore.case=T,perl=T)
data$Manager=grepl('Manager',data$Title,ignore.case=T,perl=T)
data$Intern=grepl('Intern',data$Title,ignore.case=T,perl=T)
data$Director=grepl('Director',data$Title,ignore.case=T,perl=T)
data$Analyst=grepl('Analyst',data$Title,ignore.case=T,perl=T)
data$Entry=grepl('Entry',data$Title,ignore.case=T,perl=T)
data$SeniorDirector=grepl('Senior Director',data$Title,ignore.case=T,perl=T)
data$Engineer=grepl('Engineer',data$Title,ignore.case=T,perl=T)

#loading the test file

test<-read.csv('test.csv',header=T,as.is=T)
#add extra columns
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

#we use lm to predict the SalaryNormalized values for the test file
model<-lm(log(SalaryNormalized)~South_East_London+The_City+Central_London+London+Manchester+Leeds+Belfast+Birmingham+Entry+Analyst+Director+Manager+Senior+SeniorDirector+Engineer+SourceName+ContractType:ContractTime+Category,data=data)

#add the prediction as the SalaryNormlized column for the test dataframe
test$SalaryNormalized<-exp(predict(model, test))


library('glmnet')
#2.once we have used lm to provide a lmprediction for test, we will use that to predict the test again based on glmnet.
======================================================================================================
#now training is the whole 200k set. We reopen it
training<-read.csv('train_200k.csv',header=T,as.is=T)


training$London<-training$LocationNormalized=='London'
training$South_East_London<-training$LocationNormalized=='South East London'
training$The_City<-training$LocationNormalized=='The City'
training$Central_London<-training$LocationNormalized=='Central London'
training$Manchester<-training$LocationNormalized=='Manchester'
training$Leeds<-training$LocationNormalized=='Leeds'
training$Belfast<-training$LocationNormalized=='Belfast'
training$Birmingham<-training$LocationNormalized=='Birmingham'
training$Senior=grepl('Senior',training$Title,ignore.case=T,perl=T)
training$Manager=grepl('Manager',training$Title,ignore.case=T,perl=T)
training$Intern=grepl('Intern',training$Title,ignore.case=T,perl=T)
training$Director=grepl('Director',training$Title,ignore.case=T,perl=T)
training$Analyst=grepl('Analyst',training$Title,ignore.case=T,perl=T)
training$Entry=grepl('Entry',training$Title,ignore.case=T,perl=T)
training$SeniorDirector=grepl('Senior Director',training$Title,ignore.case=T,perl=T)
training$Engineer=grepl('Engineer',training$Title,ignore.case=T,perl=T)


#remove training variables not present in test
training<-training[(training$SourceName %in% test$SourceName),] 
training<-training[(training$Category %in% test$Category),] 


#create the training.matrix  and test.matrix
training.matrix<-model.matrix(~training$South_East_London+training$The_City+training$Central_London+training$London+training$Manchester+training$Leeds+training$Belfast+training$Birmingham+training$Entry+training$Analyst+training$Director+training$Manager+training$Senior+training$SeniorDirector+training$Engineer+training$SourceName+training$ContractType:training$ContractTime+training$Category+training$SalaryNormalized)

#now we have a test$SalaryNormalized from the lm
test.matrix<-model.matrix(~test$South_East_London+test$The_City+test$Central_London+test$London+test$Manchester+test$Leeds+test$Belfast+test$Birmingham+test$Entry+test$Analyst+test$Director+test$Manager+test$Senior+test$SeniorDirector+test$Engineer+test$SourceName+test$ContractType:test$ContractTime+test$Category+test$SalaryNormalized)

#create final model
model<-cv.glmnet(training.matrix,matrix(log(training$SalaryNormalized))) 
#final prediction
prediction<-exp(as.vector(predict(model,test.matrix,s="lambda.min")))

submission<-data.frame(test$Id,prediction)
write.table(submission,file="submissionTest.csv",sep=",",row.names=F)
                 