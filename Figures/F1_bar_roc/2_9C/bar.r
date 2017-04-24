
rm(list = setdiff(ls(), lsf.str()))
library(ggplot2)
library(plyr)
library(grid)

D=read.csv("stats.csv",header=TRUE)

pdf("2_9c_bar.pdf" ) 
p=ggplot(data=D, aes(x=stats, y=number, fill=label)) +
    geom_bar(stat="identity", position=position_dodge())
p+scale_fill_manual(values=c("red","blue"))
dev.off() 
