rm(list = setdiff(ls(), lsf.str()))
library(ROCR)
s_g=read.table('D_gene_reduce_score.csv')
l_g=read.table('D_gene_reduce_label.csv')
s_p=read.table('Doxorubicin_score_probelevel.csv')
l_p=read.table('Doxorubicin_label_probelevel.csv')
pdf("all_cancer_related_roc.pdf" ) 
pred <- prediction(s_p,l_p)
perf <- performance(pred,"tpr","fpr")

predg <- prediction(s_g,l_g)
perfg <- performance(predg,"tpr","fpr")

plot( perf, col='blue', main="Statistical comparison", percent=TRUE, lwd=4)
plot( perfg, add=TRUE,col='red', lwd=4)
legend("bottomright", legend=c("all gene model", "cancer related gene model"), col= c('blue','brown'), lwd=2)
dev.off() 

