# Title     : TODO
# Objective : TODO
# Created by: user
# Created on: 09/12/2021

library(lcmm)
library(glmnet)

# function to get features
lfeatures <- function(model, formFixed, formRandom, derivForm=NULL, areaForm=NULL, data.id, col_name = "") {
    data.id$T_long <- data.id$T_survival

    # subject-specific random effects
    b.id <- as.matrix(model$predRE, ncol = 1)
    RE <- b.id[, -1]
    colnames(b.id) <- paste("RE.", colnames(b.id), sep = "")

    # value
    mfX.id <- model.frame(formFixed, data = data.id)
    mfU.id <- model.frame(formRandom, data = data.id)
    Xtime <- model.matrix(formFixed, mfX.id)
    Utime <- model.matrix(formRandom, mfU.id)
    value <- Xtime %*% model$best[1:ncol(Xtime)] + diag(Utime %*% t(RE))

    # derivation
    mfX.deriv.id <- model.frame(derivForm$fixed, data = data.id)
    mfU.deriv.id <- model.frame(derivForm$random, data = data.id)
    Xtime.deriv <- model.matrix(derivForm$fixed, mfX.deriv.id)
    Utime.deriv <- model.matrix(derivForm$random, mfU.deriv.id)
    if (!is.null(derivForm$indFixed)) {
        slope <- Xtime.deriv %*% model$best[derivForm$indFixed] +
                        diag(Utime.deriv %*% t(RE[, derivForm$indRandom]))
    } else {
        slope <- Xtime.deriv %*% model$best[1:ncol(Xtime)] +
                        diag(Utime.deriv %*% t(RE))
    }

    # area
    mfX.area.id <- model.frame(areaForm$fixed, data = data.id)
    mfU.area.id <- model.frame(areaForm$random, data = data.id)
    Xtime.area <- model.matrix(areaForm$fixed, mfX.area.id)
    Utime.area <- model.matrix(areaForm$random, mfU.area.id)
    if (!is.null(areaForm$indFixed)) {
        area <- Xtime.area %*% model$best[areaForm$indFixed] +
                            diag(Utime.area %*% t(RE[, areaForm$indRandom]))
    } else {
        area <- Xtime.area %*% model$best[1:ncol(Xtime)] +
                            diag(Utime.area %*% t(RE))
    }

    # output
    b.id <- data.frame(b.id)
    out <- cbind(b.id, value, slope, area)
    colnames(out) <- paste(col_name, ".", colnames(out), sep = "")
    colnames(out)[1] <- "id"
    return(out)
}

Cox_get_long_feat <- function(data, time_dep_feat, time_indep_feat) {
    # all features
    X <- c()
    data.id <- data[!duplicated(as.integer(data[all.vars(~id)][, 1])), ]
    long_model <- list()
    long_feat <- list()
    for(i in 1:length(time_dep_feat)) {
        formFixed = as.formula(paste(time_dep_feat[[i]], " ~ T_long + I(T_long^2)"))
        formRandom = ~T_long
        derivForm = list(fixed = ~I(2 * T_long), random = ~1, indFixed = c(2, 3), indRandom = c(2))
        areaForm = list(fixed = ~-1 + T_long + I(T_long^2/2) + I(T_long^3/3),
        random = ~-1 + T_long + I(T_long^2/2), indFixed = NULL, indRandom = NULL)
        long_model[[i]] <- hlme(fixed = formFixed, random = formRandom,
        subject = "id", ng = 1, data = data)
        long_feat[[i]] <- lfeatures(model = long_model[[i]],
                               formFixed = formFixed, formRandom = formRandom,
                               derivForm = derivForm, areaForm = areaForm,
                               col_name = time_dep_feat[i], data.id = data.id)
        # combine longitudinal features
        if(i==1)
            X <- long_feat[[i]]
        else
            X <- cbind(X, long_feat[[i]][, -1])
    }
    return(as.matrix(X[, -1]))
}

Cox_cross_val <- function(X, T, delta) {
    cv.coxnet.fit <- cv.glmnet(X, Surv(T, delta), family = "cox", alpha = 1, nfolds = 10)
    # use the best penality parameter
    best.lambda <- cv.coxnet.fit$lambda.min

    return(best.lambda)
}

Cox_fit <- function(X, T, delta, lambda) {
    coxnet.fit <- glmnet(X, Surv(T, delta),
                        family = "cox", lambda = lambda, alpha = 1)
    return(coxnet.fit)
}

Cox_score <- function(trained_model, X) {
    # predictive marker
    p_coxph_en <- predict(trained_model, newx = X, type = "response")

    return(p_coxph_en)
}