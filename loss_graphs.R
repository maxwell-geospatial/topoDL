library(ggplot2)
library(cowplot)

loss_m <- read.csv("D:/Dropbox/ABGSL/azure_project/paper/results/10_14_2020.csv")

dsc_p <- ggplot(loss_m, aes(x=epoch+1, y=dsc_m))+
  geom_line(color="#3498db", lwd=1.2)+
  geom_line(aes(x=epoch+1, y=val_dsc_m), color="#cb4335", lwd=1.2)+
  theme_classic()+
  labs(x="Epoch", y="Dice")+
  scale_y_continuous(expand = c(.005, .005), 
                     breaks = c(.70, .75, .80, .85, .90, .95, 1), 
                     limits= c(.70, 1), 
                     labels= c("0.70", "0.75", "0.80", "0.85", "0.90", "0.95", "1.00"))+
  scale_x_continuous(expand = c(.05, .05), 
                     breaks = c(1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100), 
                     limits= c(1,100), 
                     labels= c("1", "10", "20", "30", "40", "50", "60", "70", "80", "90", "100"))+
  theme(axis.text.y = element_text(size=12, color="gray40"))+
  theme(axis.text.x = element_text(size=12, color="gray40"))+
  theme(axis.title = element_text(size=16, color="gray40"))+
  theme(legend.position = "None")+
  theme(panel.grid.major.y = element_line(colour = "gray40", linetype="dashed"))

acc_p <- ggplot(loss_m, aes(x=epoch+1, y=acc_m))+
  geom_line(color="#3498db", lwd=1.2)+
  geom_line(aes(x=epoch+1, y=val_acc_m), color="#cb4335", lwd=1.2)+
  theme_classic()+
  labs(x="Epoch", y="Binary Accuracy")+
  scale_y_continuous(expand = c(.005, .005), 
                     breaks = c(.70, .75, .80, .85, .90, .95, 1), 
                     limits= c(.70, 1), 
                     labels= c("0.70", "0.75", "0.80", "0.85", "0.90", "0.95", "1.00"))+
  scale_x_continuous(expand = c(.05, .05), 
                     breaks = c(1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100), 
                     limits= c(1,100), 
                     labels= c("1", "10", "20", "30", "40", "50", "60", "70", "80", "90", "100"))+
  theme(axis.text.y = element_text(size=12, color="gray40"))+
  theme(axis.text.x = element_text(size=12, color="gray40"))+
  theme(axis.title = element_text(size=16, color="gray40"))+
  theme(legend.position = "None")+
  theme(panel.grid.major.y = element_line(colour = "gray40", linetype="dashed"))

recall_p <- ggplot(loss_m, aes(x=epoch+1, y=recall_m))+
  geom_line(color="#3498db", lwd=1.2)+
  geom_line(aes(x=epoch+1, y=val_recall_m), color="#cb4335", lwd=1.2)+
  theme_classic()+
  labs(x="Epoch", y="Recall")+
  scale_y_continuous(expand = c(.005, .005), 
                     breaks = c(.70, .75, .80, .85, .90, .95, 1), 
                     limits= c(.70, 1), 
                     labels= c("0.70", "0.75", "0.80", "0.85", "0.90", "0.95", "1.00"))+
  scale_x_continuous(expand = c(.05, .05), 
                     breaks = c(1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100), 
                     limits= c(1,100), 
                     labels= c("1", "10", "20", "30", "40", "50", "60", "70", "80", "90", "100"))+
  theme(axis.text.y = element_text(size=12, color="gray40"))+
  theme(axis.text.x = element_text(size=12, color="gray40"))+
  theme(axis.title = element_text(size=16, color="gray40"))+
  theme(legend.position = "None")+
  theme(panel.grid.major.y = element_line(colour = "gray40", linetype="dashed"))

prec_p <- ggplot(loss_m, aes(x=epoch+1, y=prec_m))+
  geom_line(color="#3498db", lwd=1.2)+
  geom_line(aes(x=epoch+1, y=val_prec_m), color="#cb4335", lwd=1.2)+
  theme_classic()+
  labs(x="Epoch", y="Precision")+
  scale_y_continuous(expand = c(.005, .005), 
                     breaks = c(.70, .75, .80, .85, .90, .95, 1), 
                     limits= c(.70, 1), 
                     labels= c("0.70", "0.75", "0.80", "0.85", "0.90", "0.95", "1.00"))+
  scale_x_continuous(expand = c(.05, .05), 
                     breaks = c(1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100), 
                     limits= c(1,100), 
                     labels= c("1", "10", "20", "30", "40", "50", "60", "70", "80", "90", "100"))+
  theme(axis.text.y = element_text(size=12, color="gray40"))+
  theme(axis.text.x = element_text(size=12, color="gray40"))+
  theme(axis.title = element_text(size=16, color="gray40"))+
  theme(legend.position = "None")+
  theme(panel.grid.major.y = element_line(colour = "gray40", linetype="dashed"))

plot_grid(dsc_p, acc_p, prec_p, recall_p, nrow=2, align="v")
