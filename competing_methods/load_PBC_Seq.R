# Title     : TODO
# Objective : TODO
# Created by: user
# Created on: 09/12/2021

defaultW <- getOption("warn")
options(warn = -1)
library("JMbayes")

load <- function() {

    data(pbc2, package = "JMbayes")
    colnames <- names(pbc2)
    data <- pbc2
    data$T_long <- data$year
    data$T_survival <- data$years
    data$delta <- data$status2
    data$SGOT <- log(data$SGOT)
    data$serBilir <- log(data$serBilir)
    data$albumin <- log(data$albumin)

    return(data)
}