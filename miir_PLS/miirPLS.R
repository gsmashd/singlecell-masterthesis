miirPLS = function (X, Y, n = NULL, ModifiedTtest = 0.5, sets = NULL, aMax = 2, 
    method = "kernel", ridge = 0.1, scrambleSets = F) 
{
    library(MASS)
    Y <- as.matrix(Y)
    X <- scale(X, scale = FALSE)
    Y <- scale(Y, scale = FALSE)
    dSets <- length(names(sets))
    if (method == "bridge") {
        mainMod <- calibrate.bridgePLS(X, Y, aMax = aMax, ridge = ridge)
    }
    if (method == "kernel") {
        mainMod <- calibrate.kernelPLS(X, Y, aMax = aMax)
    }
    W <- mainMod$sigLoadings
    #print("calibrated")
    t2 <- crossvalidatePLS(W, X, Y, ModifiedTtest, penalty = NULL, 
        aMax = aMax, method = method, ridge = ridge)
    Wjk <- t2$Wjk
    Yval <- t2$Yval
    #print("validated")
    if (!is.null(sets)) {
        for (set in names(sets)) {
            sets[[set]] <- intersect(sets[[set]], colnames(X))
        }
        sets <- sets[unlist(lapply(sets, length) > 3)]
        setStats <- setScores(W, sets, t2$t2, T2null = NULL)
        TsetReal <- setStats$setT
        print("sets computed")
    }
    penalty <- t2$penalty
    t2 <- t2$t2
    names(t2) <- colnames(X)
    if (!is.null(n)){
        print("Starting resampling")
        qv <- resamplePLS(W, t2, X, Y, TsetReal, sets, n, ModifiedTtest, 
            penalty = penalty, method = method, ridge = ridge, aMax = aMax, 
            scrambleSets = scrambleSets)
        print("Resampling finished")
        Tpert = qv$Tpert
        qSet = qv$qSet
        qv = qv$qv
    }
    else {
        Tpert = NULL
        qSet = NULL
        qv = NULL
    }
    if (is.null(sets)) {
        Wset <- NULL
        setT <- NULL
    }
    else {
        Wset <- setStats$setW
        setT <- setStats$setT
    }
    if (method == "bridge") {
        expVarX <- diag(t(mainMod$scores) %*% mainMod$scores)/sum(diag(X %*% 
            t(X)))
    }
    if (method == "kernel") {
        expVarX <- (colSums(mainMod$loadings * mainMod$loadings) * 
            diag(crossprod(mainMod$scores)))/(sum(X^2))
    }
    dY <- dim(Y)
    q2 <- vector("numeric", length = (dY[2] + 2))
    for (k in 1:dY[2]) {
        q2[k] <- 1 - sum((Y[, k] - Yval[, k, aMax])^2)/sum((Y[, 
            k]^2))
    }
    q2[dY[2] + 1] <- 1 - sum((Y - Yval[, , aMax])^2)/sum((Y)^2)
    q2[dY[2] + 2] <- 1 - sum((mainMod$f)^2)/sum((Y)^2)
    if (!is.null(qv)){
        names(qv) <- colnames(X)
    }
    return(list(T2 = t2, model = mainMod, q = qv, Wjk = Wjk, 
        Tpert = Tpert, sets = sets, qSet = qSet, Wset = Wset, 
        setT = setT, Yval = Yval, expVarX = expVarX, q2 = q2, 
        X = scale(X, scale = F, center = -1 * attr(X, "scaled:center")), 
        method = method, Y = Y))
}

calibrate.bridgePLS = function (X, Y, aMax = 3, ridge = 0.1) 
{
    a <- aMax
    dY <- dim(Y)
    dX <- dim(X)
    b <- array(NA, dim = c(dX[2], dY[2], aMax))
    W <- matrix(0, dX[2], a)
    alpha <- 1
    G0 <- svd(Y)
    if (dY[2] == 1) {
        G0 <- G0$u %*% (G0$d) %*% t(G0$u)
    }
    else {
        G0 <- G0$u %*% diag(G0$d) %*% t(G0$u)
    }
    G <- (1 - ridge) * G0 + (ridge) * diag(dY[1])
    H <- t(X) %*% G
    usv <- svd(H)
    Va <- as.matrix(usv$u[, 1:a])
    Da <- X %*% Va
    qa <- t(Y) %*% Da %*% ginv(t(Da) %*% Da)
    f <- Y - Da %*% t(qa)
    Ua <- Y %*% qa
    Q0 <- usv$v
    sf <- alpha * t(Da) %*% Da + (1 - alpha) * t(Ua) %*% Ua
    for (i in 1:a) {
        W[, i] <- sqrt(diag(sf)[i]) * Va[, i]
        if (dY[2] == 1) {
            b[, , i] = Va[, 1:i] %*% (as.matrix(qa[, 1:i]))
        }
        else {
            b[, , i] = Va[, 1:i] %*% t(as.matrix(qa[, 1:i]))
        }
    }
    rownames(W) <- colnames(X)
    fullModel <- list(sigLoadings = W, scores = Da, loadings = Va, 
        qa = qa, Ua = Ua, b = b, Q0 = Q0, f = f)
    return(fullModel)
}

calibrate.kernelPLS = function (X, Y, aMax = 3) 
{
    Y <- as.matrix(Y)
    a <- aMax
    dY <- dim(Y)
    dX <- dim(X)
    R <- matrix(0, ncol = a, nrow = dX[2])
    W <- matrix(0, ncol = a, nrow = dX[2])
    P <- matrix(0, ncol = a, nrow = dX[2])
    Q <- matrix(0, ncol = a, nrow = dY[2])
    B <- array(0, c(dX[2], dY[2], a))
    Yhat <- array(0, c(dX[1], dY[2], a))
    U <- matrix(0, ncol = a, nrow = dX[1])
    T <- matrix(0, ncol = a, nrow = dX[1])
    XtY <- crossprod(X, Y)
    for (a in 1:aMax) {
        if (dY[2] == 1) {
            w <- XtY/sqrt(c(crossprod(XtY)))
        }
        else {
            if (dY[2] < dX[2]) {
                q = eigen(crossprod(XtY), symmetric = TRUE)$vectors[, 
                  1]
                w <- XtY %*% q
                w <- w/sqrt(c(crossprod(w)))
            }
            else {
                w <- eigen(tcrossprod(XtY), symmetric = TRUE)$vectors[, 
                  1]
            }
        }
        r <- w
        if (a > 1) {
            for (j in 1:(a - 1)) {
                r <- r - (P[, j] %*% w) * R[, j]
            }
        }
        t <- X %*% r
        t2 <- c(crossprod(t))
        p <- crossprod(X, t)/t2
        q <- crossprod(XtY, r)/t2
        u <- Y %*% q
        XtY <- XtY - (t2 * p) %*% t(q)
        T[, a] <- t
        R[, a] <- r
        P[, a] <- p
        Q[, a] <- t(q)
        B[, , a] <- R[, 1:a, drop = FALSE] %*% t(Q[, 1:a, drop = FALSE])
        U[, a] <- u
        W[, a] <- w
        Yhat[, , a] <- T[, 1:a] %*% t(Q[, 1:a, drop = FALSE])
    }
    f <- Y - Yhat[, , aMax]
    sc <- sqrt(diag(crossprod(T, T)))
    Wa <- W
    for (a in 1:aMax) {
        Wa[, a] <- W[, a] * sc[a]
        T[, a] <- T[, a]
    }
    rownames(T) <- rownames(X)
    rownames(Wa) <- colnames(X)
    W <- list(sigLoadings = Wa, loadingWeights = W, scores = T, 
        loadings = P, qa = Q, Ua = U, b = B, Q0 = NULL, f = f)
    return(W)
}

crossvalidatePLS = function (W, X, Y, ModifiedTtest = 0.5, aMax = 3, method = "kernel", 
    ridge = 0.1, penalty = NULL) 
{
    library(Hotelling)
    dY <- dim(Y)
    dX <- dim(X)
    t2 <- vector("numeric", length = dX[2])
    Wjk <- array(data = NA, dim = c(dX[2], aMax, dX[1]))
    Yval <- array(data = 0, dim = c(dY[1], dY[2], aMax))
    usv <- svd(X)
    Xc <- usv$u %*% diag(usv$d)
    for (ii in 1:dX[1]) {
        if (method == "bridge") {
            subMod <- calibrate.bridgePLS(scale(Xc[-ii, ], scale = F), 
                scale(Y[-ii, ], scale = F), ridge = ridge, aMax = aMax)
        }
        if (method == "kernel") {
            subMod <- calibrate.kernelPLS(scale(Xc[-ii, ], scale = F), 
                scale(Y[-ii, ], scale = F), aMax = aMax)
        }
        for (a in 1:aMax) {
            Yval[ii, , a] <- Xc[ii, ] %*% as.matrix(subMod$b[, 
                , a])
        }
        Wi <- t(t(subMod$sigLoadings) %*% t(usv$v))
        sol <- svd(crossprod(W, Wi))
        rot <- sol$v %*% t(sol$u)
        Wjk[, , ii] <- Wi %*% rot
    }
    #print(Wjk)
    if (ModifiedTtest == 1) {
        t2 <- sqrt(apply(W^2, 1, sum))
    }
    else {
        cvmx <- array(data = NA, dim = c(dX[2], aMax, aMax))
        for (i in 1:dX[2]) {
            cvmx[i, , ] <- cov(scale(t(as.matrix(Wjk[i, , ])), 
                center = TRUE, scale = FALSE))
        }
        if (is.null(penalty)) {
            penalty <- apply(cvmx, c(2, 3), median)
        }
        for (i in 1:dX[2]) {
            cvmx[i, , ] <- (1 - ModifiedTtest) * (cvmx[i, , ]) + 
                ModifiedTtest * penalty
        }
        #print(t2)
        #options(warn=-1)
        for (i in 1:dX[2]) {
            #print("cvmx")
            #print(cvmx[i, , ])
            #print("W")
            #print(W[i, ])
            t2[i] <- hotell(cvmx[i, , ], W[i, ])
            
            
            #t2[i] <- (hotelling.test(cvmx[i, 1, ], W[i, ]) + hotelling.test(cvmx[i, 2, ], W[i, ]))/2
            #t2[i] <- hotelling.stat(cvmx[i, , ], t(W[i, ]))
            #t2[i] <- hotelling.test(cvmx[i, , ], t(W[i, ]))$
        }
        #options(warn=0)
    }
    t2 <- as.matrix(t2)
    names(t2) <- colnames(X)
    t2 <- list(t2 = t2, penalty = penalty, Yval = Yval, Wjk = Wjk)
    return(t2)
}

setScores = function (W, sets, T2, T2null = NULL, computeWset = TRUE) 
{
    library(vegan)
    dSets <- length(names(sets))
    dW <- dim(W)
    setW <- matrix(0, dSets, dW[2])
    setT <- vector("numeric", length = dSets)
    H <- sqrt(rowSums(W^2))
    for (set in 1:dSets) {
        setCov <- cor(rbind(0, t(W[sets[[set]], ])))
        setCov[setCov < 0] <- 0
        setD <- scale(setCov, center = F, scale = 1/sqrt(T2)[sets[[set]]])
        setD <- scale(t(setD), center = F, scale = 1/sqrt(T2)[sets[[set]]])
        diag(setD) <- 0
        setT[set] <- sum(setD)/(length(sets[[set]])^2 - length(sets[[set]]))
        if (computeWset) {
            setWtemp <- W[sets[[set]], ]
            for (a in 1:dW[2]) {
                setW[set, a] <- weighted.mean(setWtemp[, a], 
                  w = colMeans(setD))
            }
            rownames(setW) <- names(sets)
        }
        else {
            setW <- NULL
        }
    }
    names(setT) <- names(sets)
    return(list(setT = setT, setW = setW))
}

resamplePLS = function (W, Treal, X, Y, TsetReal, sets, n, ModifiedTtest, penalty, 
    method = "kernel", ridge = 0.01, aMax = 3, scrambleSets = F) 
{
    dY <- dim(Y)
    dX <- dim(X)
    dSet <- length(names(sets))
    dW <- dim(W)
    aMax <- dW[2]
    Tpert <- array(data = NA, dim = c(dX[2], n))
    TsetPert <- array(data = NA, dim = c(dSet, n))
    for (i in 1:n) {
        if (i %in% c(2, 3, 5, 7, 9, 11, 17, 33, 65, 129, 257, 
            513, 1025, 2049, 4097)) {
            print(paste("completed", as.character(i - 1), "resamplings"))
        }
        Yind <- sample(1:dY[1])
        Y <- Y[Yind, ]
        Y <- as.matrix(Y)
        if (method == "kernel") {
            pertModel <- calibrate.kernelPLS(X, Y, aMax = aMax)
        }
        if (method == "bridge") {
            pertModel <- calibrate.bridgePLS(X, Y, aMax = aMax, 
                ridge = ridge)
        }
        Wpert <- pertModel$sigLoadings
        ValRes <- crossvalidatePLS(Wpert, X, Y, ModifiedTtest, 
            penalty = penalty, aMax = aMax, method = method, 
            ridge = ridge)
        if (!is.null(sets)) {
            if (scrambleSets) {
                for (set in names(sets)) {
                  sets[[set]] <- sample(colnames(X), size = length(sets[[set]]))
                }
                setStatsPert <- setScores(W, sets, Treal, T2null = NULL)
            }
            else {
                setStatsPert <- setScores(Wpert, sets, ValRes$t2, 
                  T2null = NULL, computeWset = F)
            }
            TsetPert[, i] <- setStatsPert$setT
        }
        Tpert[, i] <- ValRes$t2
    }
    qv <- vector("numeric", dX[2])
    print("Computing q values")
    for (i in 1:dX[2]) {
        n_better <- colSums(Tpert >= Treal[i])
        qv[i] <- median(n_better)/(1 + sum(Treal > Treal[i]))
    }
    if (!is.null(sets)) {
        print("computing set q values")
        qSet <- vector("numeric", dSet)
        for (i in 1:dSet) {
            n_better <- colSums(TsetPert >= TsetReal[i])
            qSet[i] <- median(n_better)/(1 + sum(TsetReal > TsetReal[i]))
        }
        names(qSet) <- names(sets)
    }
    else {
        qSet <- NULL
    }
    return(list(qv = qv, Tpert = Tpert, qSet = qSet))
}

