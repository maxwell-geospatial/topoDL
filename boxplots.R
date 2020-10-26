#Read in packages
library(ggplot2)
library(cowplot)
library(dplyr)

#Read in data
topo_metrics <- read.csv("D:/Dropbox/ABGSL/azure_project/paper/results/metrics3.csv")

#Alter site field to ordered factor
topo_metrics$set <- factor(topo_metrics$set, ordered = TRUE, 
                                levels = c("test", "val", "oh", "va"))

#F1/Dice Plot
f1_g <- ggplot(topo_metrics, aes(x=set, y=f1, fill=set))+
  geom_boxplot()+
  theme_classic()+
  scale_y_continuous(breaks=c(0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1), 
                     limits=c(0,1), 
                     labels=c("0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"))+
  scale_x_discrete(labels = c("Test", "KY", "OH", "VA"))+
  labs(x="Dataset", y="F1 Score", fill="Dataset")+
  scale_fill_manual(values = c("#3FC199", "#289BDE", "#DE6428", "#7147A6"), labels=c("Test", "Validation", "OH Validation", "VA Validation"))+
  theme(axis.text.y = element_text(size=12))+
  theme(axis.text.x = element_text(size=12))+
  theme(plot.title = element_text(size=18))+
  theme(axis.title = element_text(size=14))+
  theme(strip.text = element_text(size = 14))+
  theme(legend.title = element_text(size=14))+
  theme(panel.grid.major.y= element_line(colour = "gray",size=0.4, linetype=2))+
  theme(legend.position="none")

#Precision Plot
prec_g <- ggplot(topo_metrics, aes(x=set, y=precision, fill=set))+
  geom_boxplot()+
  theme_classic()+
  scale_y_continuous(breaks=c(0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1), 
                     limits=c(0,1), 
                     labels=c("0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"))+
  scale_x_discrete(labels = c("Test", "KY", "OH", "VA"))+
  labs(x="Dataset", y="Precision", fill="Dataset")+
  scale_fill_manual(values = c("#3FC199", "#289BDE", "#DE6428", "#7147A6"), labels=c("Test", "Validation", "OH Validation", "VA Validation"))+
  theme(axis.text.y = element_text(size=12))+
  theme(axis.text.x = element_text(size=12))+
  theme(plot.title = element_text(size=18))+
  theme(axis.title = element_text(size=14))+
  theme(strip.text = element_text(size = 14))+
  theme(legend.title = element_text(size=14))+
  theme(panel.grid.major.y= element_line(colour = "gray",size=0.4, linetype=2))+
  theme(legend.position="none")

#Recall Plot
recall_g <- ggplot(topo_metrics, aes(x=set, y=recall, fill=set))+
  geom_boxplot()+
  theme_classic()+
  scale_y_continuous(breaks=c(0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1), 
                     limits=c(0,1), 
                     labels=c("0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"))+
  scale_x_discrete(labels = c("Test", "KY", "OH", "VA"))+
  labs(x="Dataset", y="Recall", fill="Dataset")+
  scale_fill_manual(values = c("#3FC199", "#289BDE", "#DE6428", "#7147A6"), labels=c("Test", "Validation", "OH Validation", "VA Validation"))+
  theme(axis.text.y = element_text(size=12))+
  theme(axis.text.x = element_text(size=12))+
  theme(plot.title = element_text(size=18))+
  theme(axis.title = element_text(size=14))+
  theme(strip.text = element_text(size = 14))+
  theme(legend.title = element_text(size=14))+
  theme(panel.grid.major.y= element_line(colour = "gray",size=0.4, linetype=2))+
  theme(legend.position="none")

#Specificity Plot
spec_g <- ggplot(topo_metrics, aes(x=set, y=specificity, fill=set))+
  geom_boxplot()+
  theme_bw()+
  scale_y_continuous(breaks=c(0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1), 
                     limits=c(0,1), 
                     labels=c("0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"))+
  scale_x_discrete(labels = c("Test", "KY", "OH", "VA"))+
  labs(x="Dataset", y="Specificity", fill="Dataset")+
  scale_fill_manual(values = c("#3FC199", "#289BDE", "#DE6428", "#7147A6"), labels=c("Test", "Validation", "OH Validation", "VA Validation"))+
  theme(axis.text.y = element_text(size=12, color="gray40"))+
  theme(axis.text.x = element_text(size=12, color="gray40"))+
  theme(plot.title = element_text(size=18))+
  theme(axis.title = element_text(size=14))+
  theme(strip.text = element_text(size = 14))+
  theme(legend.title = element_text(size=14))+
  theme(legend.position="none")

#Accuracy Plot
acc_g <- ggplot(topo_metrics, aes(x=set, y=acc, fill=set))+
  geom_boxplot()+
  theme_bw()+
  scale_y_continuous(breaks=c(0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1), 
                     limits=c(0,1), 
                     labels=c("0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"))+
  scale_x_discrete(labels = c("Test", "KY", "OH", "VA"))+
  labs(x="Dataset", y="Overall Accuracy", fill="Dataset")+
  scale_fill_manual(values = c("#3FC199", "#289BDE", "#DE6428", "#7147A6"), labels=c("Test", "Validation", "OH Validation", "VA Validation"))+
  theme(axis.text.y = element_text(size=12, color="gray40"))+
  theme(axis.text.x = element_text(size=12, color="gray40"))+
  theme(plot.title = element_text(size=18))+
  theme(axis.title = element_text(size=14))+
  theme(strip.text = element_text(size = 14))+
  theme(legend.title = element_text(size=14))+
  theme(panel.grid.major.y= element_line(colour = "black",size=0.75))+
  theme(legend.position="none")

#Combine Plots
plot_grid(f1_g, prec_g, recall_g, nrow=3)


#Obtain Summary Metrics by Set
topo_metrics %>% group_by(set) %>% summarise(acc=mean(acc), f1=mean(f1), precision=mean(precision), recall=mean(recall), specificity=mean(specificity))
