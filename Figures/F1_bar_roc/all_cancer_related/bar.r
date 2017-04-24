
rm(list = setdiff(ls(), lsf.str()))
library(ggplot2)
library(plyr)
library(grid)

pdf("all_cancer_related_bar.pdf" ) 
D=read.csv("stats_Doxorubicin.csv",header=TRUE)

p=ggplot(data=D, aes(x=stats, y=number, fill=label)) +
    geom_bar(stat="identity", position=position_dodge())
p+scale_fill_manual(values=c("red","blue"))
dev.off() 

