#I downloaded all my Gmail personal emails from 2003 into my computer as individual .txt files using Thunderbird. 
#Then, with cat and the command line, I created two txt files including the timestamps of sent and received emails.

#load the received_timestamps and sent_timestamps (i did some cleaning in excel that i didnt know how to do i command line)
sent <- read.csv('sent_timestamps.csv',as.is=T,header=F)
received <- read.csv('received_timestamps.csv',as.is=T,header=F)

#change column names to make them match
colnames(sent)<-c('date','time')
colnames(received)<-c('date','time')
#add a column 'type' to both received (value = 'Received') and sent (value = 'Sent)
received$type<-'Received'
sent$type<-"Sent"

#join both 'received' and 'sent' data frames into one:
times<-rbind(received,sent)

#convert the 'date' column into a Date object column (so R recognizes it as a date)
times$date<-as.Date(times$date,'%m/%d/%Y')

#adding a month/day columns based of the date column
times$month <- months(as.Date(times$date))
times$day <- weekdays(as.Date(times$date))

#Sorting the Month column as ordered factors. If not, ggplot just sort them alphabetically
times$month <- factor(times$month,levels=c("January","February","March", "April","May","June","July","August","September", "October","November","December"),ordered=TRUE)

#plottiong the histogram by month of year
ggplot(times,aes(x=month,fill=type))+geom_histogram()+facet_wrap(~type)

#assigning the Days of week as factors on the desired order:
times$day<-factor(times$day,levels=c('Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'))
#plotting the histogram by day of week
ggplot(times,aes(x=day,fill='type'))+geom_histogram()+facet_wrap(~type)

#=========================================================================
#added in Excel the Hour from the Timestamps, so I could get a histogram by hour.
ggplot(times,aes(x=hour,fill=type))+geom_histogram()+facet_wrap(~type)+theme_bw()+scale_x_discrete()

#===========================================================================
#plot line of count of emails sent and received by week
#add a column to the timestamp with the week number from 0 to the final week (total of 516 weeks)

received_time$week=(floor(difftime(strptime(received_time$date, format = "%m/%d/%Y"),strptime(received_time[nrow(received_time),1], format = "%m/%d/%Y"),units="weeks"))+516)

sent_time$week=(floor(difftime(strptime(sent_time$date, format = "%m/%d/%Y"),strptime(sent_time[nrow(sent_time),1], format = "%m/%d/%Y"),units="weeks"))+516)

times=rbind(received_time,sent_time)
times$week=as.numeric(times$week)	#this way we remove the 'weeks' string
#get summary of Received/Sent and week
counts <- ddply(times, .(times$type, times$week), nrow)
colnames(counts)=c('type','week','freq')



