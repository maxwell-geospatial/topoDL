setwd("C:/Users/amaxwel6/Desktop")
t2a <- read.csv("train2a.csv")
t2a$size <- 2
t2a$set <- "a"
t2b <- read.csv("train2b.csv")
t2b$size <- 2
t2b$set <- "b"
t2c <- read.csv("train2c.csv")
t2c$size <- 2
t2c$set <- "c"
t2d <- read.csv("train2d.csv")
t2d$size <- 2
t2d$set <- "d"
t2e <- read.csv("train2e.csv")
t2e$size <- 2
t2e$set <- "e"

t5a <- read.csv("train5a.csv")
t5a$size <- 5
t5a$set <- "a"
t5b <- read.csv("train5b.csv")
t5b$size <- 5
t5b$set <- "b"
t5c <- read.csv("train5c.csv")
t5c$size <- 5
t5c$set <- "c"
t5d <- read.csv("train5d.csv")
t5d$size <- 5
t5d$set <- "d"
t5e <- read.csv("train5e.csv")
t5e$size <- 5
t5e$set <- "e"

t10a <- read.csv("train10a.csv")
t10a$size <- 10
t10a$set <- "a"
t10b <- read.csv("train10b.csv")
t10b$size <- 10
t10b$set <- "b"
t10c <- read.csv("train10c.csv")
t10c$size <- 10
t10c$set <- "c"
t10d <- read.csv("train10d.csv")
t10d$size <- 10
t10d$set <- "d"
t10e <- read.csv("train10e.csv")
t10e$size <- 10
t10e$set <- "e"

t15a <- read.csv("train15a.csv")
t15a$size <- 15
t15a$set <- "a"
t15b <- read.csv("train15b.csv")
t15b$size <- 15
t15b$set <- "b"
t15c <- read.csv("train15c.csv")
t15c$size <- 15
t15c$set <- "c"
t15d <- read.csv("train15d.csv")
t15d$size <- 15
t15d$set <- "d"
t15e <- read.csv("train15e.csv")
t15e$size <- 15
t15e$set <- "e"


all <- rbind(t2a, t2b, t2c, t2d, t2e, 
             t5a, t5b, t5c, t5d, t5e,
             t10a, t10b, t10c, t10d, t10e,
             t15a, t15b, t15c, t15d, t15e)

library(dplyr)
all$grpvar <- paste0(as.character(all$size), "_", as.character(all$epoch))
all_grp <- all %>% group_by(c(grpvar)) %>% summarize(epoch = mean(epoch), size=mean(size), t_dice_m =mean(dsc_m), t_prec_m=mean(prec_m), t_recall_m=mean(recall_m), 
                                                                  v_dice_m =mean(val_dsc_m), v_prec_m=mean(val_prec_m), v_recall_m=mean(val_recall_m),
                                                                  t_dice_mn =min(dsc_m), t_prec_mn=min(prec_m), t_recall_mn=min(recall_m), 
                                                                  v_dice_mn =min(val_dsc_m), v_prec_mn=min(val_prec_m), v_recall_mn=min(val_recall_m),
                                                                  t_dice_mx =max(dsc_m), t_prec_mx=max(prec_m), t_recall_mx=max(recall_m), 
                                                                  v_dice_mx =max(val_dsc_m), v_prec_mx=max(val_prec_m), v_recall_mx=max(val_recall_m),
                                                                  t_acc_m =mean(acc_m), t_acc_mn=min(acc_m), t_acc_mx=max(acc_m), v_acc_m=mean(val_acc_m), v_acc_mn=min(val_acc_m),  v_acc_mx=max(val_acc_m))

loss_m <- read.csv("D:/Dropbox/ABGSL/azure_project/paper/results/10_14_2020.csv")


library(ggplot2)
gdice <- ggplot(all_grp)+
  geom_ribbon(aes(x= epoch+1, ymin=v_dice_mn, ymax=v_dice_mx), fill="#cb4335", alpha=.5)+
  geom_ribbon(aes(x= epoch+1, ymin=t_dice_mn, ymax=t_dice_mx), fill="#3498db", alpha=.5)+
  geom_line(aes(x=epoch+1, y=v_dice_m), color = "#cb4335", lwd=1)+
  geom_line(aes(x=epoch+1, y=t_dice_m), color = "#3498db", lwd=1)+
  facet_grid(size~.)+
  geom_line(data=loss_m, aes(x=epoch+1, y=dsc_m), lwd=1, color="#A652F0")+
  geom_line(data=loss_m, aes(x=epoch+1, y=val_dsc_m), lwd=1, color="#56A422")+
  theme_classic()+
  labs(x="Epoch", y="Dice Coefficient")+
  scale_y_continuous(expand = c(.005, .005), 
                     breaks = c(0, .2, .4, .6, .8, 1), 
                     limits= c(0, 1), 
                     labels= c("0.0", "0.2", "0.4", "0.6", "0.8", "1.0"))+
  scale_x_continuous(expand = c(.005, .005), 
                     breaks = c(1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100), 
                     limits= c(1,100), 
                     labels= c("1", "10", "20", "30", "40", "50", "60", "70", "80", "90", "100"))+
  theme(axis.text.y = element_text(size=12, color="gray40"))+
  theme(axis.text.x = element_text(size=12, color="gray40"))+
  theme(axis.title = element_text(size=16, color="gray40"))+
  theme(legend.position = "None")+
  theme(panel.grid.major.y = element_line(colour = "gray40", linetype="dashed"))+
  theme(panel.spacing.y = unit(1, "lines"))

library(ggplot2)
gprec <- ggplot(all_grp)+
  geom_ribbon(aes(x= epoch+1, ymin=v_prec_mn, ymax=v_prec_mx), fill="#cb4335", alpha=.5)+
  geom_ribbon(aes(x= epoch+1, ymin=t_prec_mn, ymax=t_prec_mx), fill="#3498db", alpha=.5)+
  geom_line(aes(x=epoch+1, y=v_prec_m), color = "#cb4335", lwd=1)+
  geom_line(aes(x=epoch+1, y=t_prec_m), color = "#3498db", lwd=1)+
  facet_grid(size~.)+
  geom_line(data=loss_m, aes(x=epoch+1, y=prec_m), lwd=1, color="#A652F0")+
  geom_line(data=loss_m, aes(x=epoch+1, y=val_prec_m), lwd=1, color="#56A422")+
  theme_classic()+
  labs(x="Epoch", y="Precision")+
  scale_y_continuous(expand = c(.005, .005), 
                     breaks = c(0, .2, .4, .6, .8, 1), 
                     limits= c(0, 1), 
                     labels= c("0.0", "0.2", "0.4", "0.6", "0.8", "1.0"))+
  scale_x_continuous(expand = c(.005, .005), 
                     breaks = c(1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100), 
                     limits= c(1,100), 
                     labels= c("1", "10", "20", "30", "40", "50", "60", "70", "80", "90", "100"))+
  theme(axis.text.y = element_text(size=12, color="gray40"))+
  theme(axis.text.x = element_text(size=12, color="gray40"))+
  theme(axis.title = element_text(size=16, color="gray40"))+
  theme(legend.position = "None")+
  theme(panel.grid.major.y = element_line(colour = "gray40", linetype="dashed"))+
  theme(panel.spacing.y = unit(1, "lines"))
  

library(ggplot2)
grecall <- ggplot(all_grp)+
  geom_ribbon(aes(x= epoch+1, ymin=v_recall_mn, ymax=v_recall_mx), fill="#cb4335", alpha=.5)+
  geom_ribbon(aes(x= epoch+1, ymin=t_recall_mn, ymax=t_recall_mx), fill="#3498db", alpha=.5)+
  geom_line(aes(x=epoch+1, y=v_recall_m), color = "#cb4335", lwd=1)+
  geom_line(aes(x=epoch+1, y=t_recall_m), color = "#3498db", lwd=1)+
  facet_grid(size~.)+
  geom_line(data=loss_m, aes(x=epoch+1, y=recall_m), lwd=1, color="#A652F0")+
  geom_line(data=loss_m, aes(x=epoch+1, y=val_recall_m), lwd=1, color="#56A422")+
  theme_classic()+
  labs(x="Epoch", y="Recall")+
  scale_y_continuous(expand = c(.005, .005), 
                     breaks = c(0, .2, .4, .6, .8, 1), 
                     limits= c(0, 1), 
                     labels= c("0.0", "0.2", "0.4", "0.6", "0.8", "1.0"))+
  scale_x_continuous(expand = c(.005, .005),
                     breaks = c(1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100), 
                     limits= c(1,100), 
                     labels= c("1", "10", "20", "30", "40", "50", "60", "70", "80", "90", "100"))+
  theme(axis.text.y = element_text(size=12, color="gray40"))+
  theme(axis.text.x = element_text(size=12, color="gray40"))+
  theme(axis.title = element_text(size=16, color="gray40"))+
  theme(legend.position = "None")+
  theme(panel.grid.major.y = element_line(colour = "gray40", linetype="dashed"))+
  theme(panel.spacing.y = unit(1, "lines"))


library(ggplot2)
gacc <- ggplot(all_grp)+
  geom_ribbon(aes(x= epoch+1, ymin=v_acc_mn, ymax=v_acc_mx), fill="#cb4335", alpha=.5)+
  geom_ribbon(aes(x= epoch+1, ymin=t_acc_mn, ymax=t_acc_mx), fill="#3498db", alpha=.5)+
  geom_line(aes(x=epoch+1, y=v_acc_m), color = "#cb4335", lwd=1)+
  geom_line(aes(x=epoch+1, y=t_acc_m), color = "#3498db", lwd=1)+
  facet_grid(size~.)+
  geom_line(data=loss_m, aes(x=epoch+1, y=acc_m), lwd=1, color="#A652F0")+
  geom_line(data=loss_m, aes(x=epoch+1, y=val_acc_m), lwd=1, color="#56A422")+
  theme_classic()+
  labs(x="Epoch", y="Accuracy")+
  scale_y_continuous(expand = c(.005, .005), 
                     breaks = c(0, .2, .4, .6, .8, 1), 
                     limits= c(0, 1), 
                     labels= c("0.0", "0.2", "0.4", "0.6", "0.8", "1.0"))+
  scale_x_continuous(expand = c(.005, .005),
                     breaks = c(1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100), 
                     limits= c(1,100), 
                     labels= c("1", "10", "20", "30", "40", "50", "60", "70", "80", "90", "100"))+
  theme(axis.text.y = element_text(size=12, color="gray40"))+
  theme(axis.text.x = element_text(size=12, color="gray40"))+
  theme(axis.title = element_text(size=16, color="gray40"))+
  theme(legend.position = "None")+
  theme(panel.grid.major.y = element_line(colour = "gray40", linetype="dashed"))+
  theme(panel.spacing.y = unit(1, "lines"))

library(cowplot)
plot_grid(gdice, gacc, gprec, grecall, nrow=2, align="v")


all_grp2 <- all %>% group_by(size, set) %>% summarize(size=mean(size), t_dice_m =mean(val_dsc_m), t_prec_m=mean(val_prec_m), t_recall_m=mean(val_recall_m))

ggplot(all_grp2, aes(x=as.factor(size), y=t_dice_m))+
  geom_point(shape=15, color="#CE542D", size=2)+
  geom_point(aes(y=t_prec_m), position = position_nudge(x = 0.2), shape=16, color="#285EC6", size=2)+
  geom_point(aes(y=t_recall_m), position = position_nudge(x = -0.2), shape=17, color="#56A422", size=2)+
  geom_hline(yintercept=.96, color="#CE542D", linetype="dashed", lwd=1)+
  geom_hline(yintercept=.97, color="#285EC6", linetype="dashed", lwd=1)+
  geom_hline(yintercept=.95, color="#56A422", linetype="dashed", lwd=1)+
  theme_classic()+
  labs(x="Number of Topographic Maps", y="Metrics")+
  scale_y_continuous(expand = c(.005, .005), 
                     breaks = c(.3, .4, .5, .6, .7, .8, .9, 1), 
                     limits= c(.3, 1), 
                     labels= c("0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"))+
  theme(axis.text.y = element_text(size=12))+
  theme(axis.text.x = element_text(size=12))+
  theme(axis.title = element_text(size=16))+
  theme(legend.position = "None")+
  theme(panel.grid.major.y = element_line(colour = "gray40", linetype="dashed"))

metrics <- read.csv("D:/Dropbox/ABGSL/azure_project/paper/results/sample_size/sample_size_metrics.csv")
metrics$size <- rep(c(10, 15, 2, 5), each=15)
metrics$size <- factor(metrics$size, ordered=TRUE, levels = c("2", "5", "10", "15"))

library(dplyr)
m_ky <- metrics %>% filter(set=="KY")
m_oh <- metrics %>% filter(set=="OH")
m_va <- metrics %>% filter(set=="VA")

dsc_s <- ggplot(data=m_ky, aes(x=as.factor(size), y=dsc_m))+
  geom_point(shape=15, color="#CE542D", size=2, position = position_nudge(x = -0.2))+
  geom_point(data=m_oh, aes(x=as.factor(size), y=dsc_m), shape=16, color="#285EC6", size=2)+
  geom_point(data=m_va, aes(x=as.factor(size), y=dsc_m), shape=17, color="#56A422", position = position_nudge(x = 0.2), size=2)+
  geom_hline(yintercept=.949, color="#CE542D", linetype="solid", lwd=1)+
  geom_hline(yintercept=.894, color="#285EC6", linetype="solid", lwd=1)+
  geom_hline(yintercept=.837, color="#56A422", linetype="solid", lwd=1)+
  theme_classic()+
  labs(x="Number of Topographic Maps", y="Dice Coefficient")+
  scale_y_continuous(expand = c(.005, .005), 
                     breaks = c(0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1), 
                     limits= c(0, 1), 
                     labels= c("0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"))+
  theme(axis.text.y = element_text(size=12))+
  theme(axis.text.x = element_text(size=12))+
  theme(axis.title = element_text(size=16))+
  theme(legend.position = "None")+
  theme(panel.grid.major.y = element_line(colour = "gray40", linetype="dashed"))

prec_s <- ggplot(data=m_ky, aes(x=as.factor(size), y=prec_m))+
  geom_point(shape=15, color="#CE542D", size=2, position = position_nudge(x = -0.2))+
  geom_point(data=m_oh, aes(x=as.factor(size), y=prec_m), shape=16, color="#285EC6", size=2)+
  geom_point(data=m_va, aes(x=as.factor(size), y=prec_m), shape=17, color="#56A422", position = position_nudge(x = 0.2), size=2)+
  geom_hline(yintercept=.954, color="#CE542D", linetype="solid", lwd=1)+
  geom_hline(yintercept=.966, color="#285EC6", linetype="solid", lwd=1)+
  geom_hline(yintercept=.971, color="#56A422", linetype="solid", lwd=1)+
  theme_classic()+
  labs(x="Number of Topographic Maps", y="Precision")+
  scale_y_continuous(expand = c(.005, .005), 
                     breaks = c(0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1), 
                     limits= c(0, 1), 
                     labels= c("0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"))+
  theme(axis.text.y = element_text(size=12))+
  theme(axis.text.x = element_text(size=12))+
  theme(axis.title = element_text(size=16))+
  theme(legend.position = "None")+
  theme(panel.grid.major.y = element_line(colour = "gray40", linetype="dashed"))

rec_s <- ggplot(data=m_ky, aes(x=as.factor(size), y=recall_m))+
  geom_point(shape=15, color="#CE542D", size=2, position = position_nudge(x = -0.2))+
  geom_point(data=m_oh, aes(x=as.factor(size), y=recall_m), shape=16, color="#285EC6", size=2)+
  geom_point(data=m_va, aes(x=as.factor(size), y=recall_m), shape=17, color="#56A422", position = position_nudge(x = 0.2), size=2)+
  geom_hline(yintercept=.944, color="#CE542D", linetype="solid", lwd=1)+
  geom_hline(yintercept=.835, color="#285EC6", linetype="solid", lwd=1)+
  geom_hline(yintercept=.741, color="#56A422", linetype="solid", lwd=1)+
  theme_classic()+
  labs(x="Number of Topographic Maps", y="Recall")+
  scale_y_continuous(expand = c(.005, .005), 
                     breaks = c(0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1), 
                     limits= c(0, 1), 
                     labels= c("0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"))+
  theme(axis.text.y = element_text(size=12))+
  theme(axis.text.x = element_text(size=12))+
  theme(axis.title = element_text(size=16))+
  theme(legend.position = "None")+
  theme(panel.grid.major.y = element_line(colour = "gray40", linetype="dashed"))

acc_s <- ggplot(data=m_ky, aes(x=as.factor(size), y=acc_m))+
  geom_point(shape=15, color="#CE542D", size=2, position = position_nudge(x = -0.2))+
  geom_point(data=m_oh, aes(x=as.factor(size), y=acc_m), shape=16, color="#285EC6", size=2)+
  geom_point(data=m_va, aes(x=as.factor(size), y=acc_m), shape=17, color="#56A422", position = position_nudge(x = 0.2), size=2)+
  geom_hline(yintercept=.989, color="#CE542D", linetype="solid", lwd=1)+
  geom_hline(yintercept=.963, color="#285EC6", linetype="solid", lwd=1)+
  geom_hline(yintercept=.942, color="#56A422", linetype="solid", lwd=1)+
  theme_classic()+
  labs(x="Number of Topographic Maps", y="Accuracy")+
  scale_y_continuous(expand = c(.005, .005), 
                     breaks = c(0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1), 
                     limits= c(0, 1), 
                     labels= c("0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"))+
  theme(axis.text.y = element_text(size=12))+
  theme(axis.text.x = element_text(size=12))+
  theme(axis.title = element_text(size=16))+
  theme(legend.position = "None")+
  theme(panel.grid.major.y = element_line(colour = "gray40", linetype="dashed"))

library(cowplot)
plot_grid(dsc_s, acc_s, prec_s, rec_s, nrow=2, align="v")