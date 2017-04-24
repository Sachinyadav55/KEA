

rm(list = setdiff(ls(), lsf.str()))
library(ggplot2)
library(plyr)
library(grid)

# Multiple plot function
#
# ggplot objects can be passed in ..., or to plotlist (as a list of ggplot objects)
# - cols:   Number of columns in layout
# - layout: A matrix specifying the layout. If present, 'cols' is ignored.
#
# If the layout is something like matrix(c(1,2,3,3), nrow=2, byrow=TRUE),
# then plot 1 will go in the upper left, 2 will go in the upper right, and
# 3 will go all the way across the bottom.
#
multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
  require(grid)

  # Make a list from the ... arguments and plotlist
  plots <- c(list(...), plotlist)

  numPlots = length(plots)

  # If layout is NULL, then use 'cols' to determine layout
  if (is.null(layout)) {
    # Make the panel
    # ncol: Number of columns of plots
    # nrow: Number of rows needed, calculated from # of cols
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                    ncol = cols, nrow = ceiling(numPlots/cols))
  }

 if (numPlots==1) {
    print(plots[[1]])

  } else {
    # Set up the page
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))

    # Make each plot, in the correct location
    for (i in 1:numPlots) {
      # Get the i,j matrix positions of the regions that contain this subplot
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))

      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col))
    }
  }
}


x1=read.csv("GSE20565_score.csv",header=TRUE)
x2=read.csv("GSE18521_score.csv",header=TRUE)
x3=read.csv("GSE30161_score.csv",header=TRUE)
l1=length(x1[,1])
v1=matrix("GSE9899",l1,1)
l2=length(x2[,1])
v2=matrix("GSE18521",l2,1)
l3=length(x3[,1])
v3=matrix("GSE30161",l3,1)
x1=cbind(x1,v1)
x2=cbind(x2,v2)
x3=cbind(x3,v3)
colnames(x1)[10]="label"
colnames(x2)[10]="label"
colnames(x3)[10]="label"

x=rbind(x1,x2,x3)
#plots <- list()  # new empty list

pdf("3sets_drugs_score.pdf" ) 
grid.newpage()
pushViewport(viewport(layout = grid.layout(8, 1)))
vplayout <- function(x, y) viewport(layout.pos.row = x, layout.pos.col = y)

k=1
for (i in 8:1) {
    p=ggplot(x, aes(x[,(i+1)], fill=factor(label))) + geom_density(alpha=.3) + ggtitle(colnames(x)[i+1]) + xlab("score") + ylab("density") + geom_vline(aes(xintercept=0),color="blue", linetype="dashed", size=0.2) + xlim(-10, 10) + ylim(0,0.6)
    #print(plots[[i]])
    print(p, vp = vplayout(k, 1))
    k=k+1
} 

#layout <- matrix(c(1, 1, 2, 3, 4, 5), nrow = 2, byrow = TRUE)
#multiplot(plotlist = plots, cols = 1)
dev.off() 