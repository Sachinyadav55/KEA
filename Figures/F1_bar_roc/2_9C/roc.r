rm(list = setdiff(ls(), lsf.str()))
library(ROCR)
s_g=read.table('LC_ME_score.csv')
l_g=read.table('LC_ME_label.csv')
s_p=read.table('18_all_score.csv')
l_p=read.table('18_all_label.csv')

pdf("2_9c_roc.pdf" ) 

pred <- prediction(s_p,l_p)
perf <- performance(pred,"tpr","fpr")

predg <- prediction(s_g,l_g)
perfg <- performance(predg,"tpr","fpr")

plot( perf, col='blue', main="Statistical comparison", percent=TRUE, lwd=4)
plot( perfg, add=TRUE,col='red', lwd=4)
legend("bottomright", legend=c("9 type cancer model", "2 type cancer model"), col= c('blue','brown'), lwd=2)

dev.off() 
