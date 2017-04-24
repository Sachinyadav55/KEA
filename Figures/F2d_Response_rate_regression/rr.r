rm(list = setdiff(ls(), lsf.str()))
library(ggplot2)
library(plyr)
library(grid)


x=read.csv("Response_rate.csv",header=TRUE,row.names=1)

fit <- lm(predicted_response_rate ~ observed_literature_response_rate, data = x)
summary(fit)
pdf("Response_rate.pdf" ) 
p = ggplot(fit$model, aes(x=observed_literature_response_rate, y=predicted_response_rate, label = rownames(x) ) ) + 
  geom_point(aes(size = 3)) +
  stat_smooth(method = "lm", col = "red") +
  labs(title = paste("R2 = ",round(signif(summary(fit)$adj.r.squared, 5), digits=4),
                     " Slope =",round(signif(fit$coef[[2]], 5), digits=4),
                     " P =",round(signif(summary(fit)$coef[2,4], 5), digits=4),
                     "cor_coefficient =",round(cor(x$observed_literature_response_rate, x$predicted_response_rate), digits=4)))#+ xlim(0, 100) + ylim(0,100)
p + geom_text(aes(vjust=2, hjust=0.6), size=6)

dev.off() 


