library("e1071")
library(neuralnet)

#' Training a neural network
#' 
#' Trains a neural network using K-fold cross validation to get a validation accuracy, which is the average of the K
#' validation accuracies.
#' Returns an object containing two data frames with result data and the neural network itself.
#' @param trainData a data frame which will be used for training. Must consists of columns for every
#' feature, followed by one column containing the correct labels.
#' @param type decides whether classification or regression will be used. By default will be set to "classification" when the label column is factor and to
#' "regression" otherwise.
#' @param allConverge a logical value indicating whether training epochs should be restarted when the network did not converge in a certain step. When set to TRUE,
#' the size of the \code{cross} parameter indicates the amount of results in the \code{resultTable}.
#' @param h a numeric value specifying the amount of nodes in the hidden layer. For more hidden layers, use a vector of numeric values.
#' @param rep a number specifying amount of repetitions done per cross validation step. For example, rep = 5 and cross = 10
#' will lead to 50 reps in total.
#' @param threshold a numeric value between 0 and 1 specifying the threshold for the partial derivatives of the error function as stopping criteria.
#' When the error does not decrease more than this percentage in a step, training terminates.
#' @param stepmax the maximum steps for the training of the neural network. Reaching this maximum will terminate the training process.
#' @param lifesign an integer specifying how often (the step size) the learning algorithm should print the minimal threshold.
#' @param cross an integer specifying the K for K-fold cross validation. A value of 1 will result in no cross validation and testing
#' on the training set itself; a value equal to the sample size of the train data will result in leave-one-out cross validation.
#' 
#' @import "neuralnet"
#' 
#' @export
#' 
#' @return An object of class \code{"nnsvm"}, containing the following components:
#' \item{results}{a data frame consisting of one row with information about the training process, like the sample size in
#' the train data, used parameters, average training steps and accuracies. The values are an average over the resultTable (all CV loops).  }
#' \item{resultTable}{a data frame consisting of information about every cross validation step of the training process. }
#' \item{nn}{an object of class \code{"nn"} from the \code{\link[neuralnet]{neuralnet}} package, containing the learned network.}
#' @examples
#' nnObject <- nnTrain(trainData, h=c(2,3), threshold=0.01, cross=5) # train a network with two hidden layers and 5-fold cross validation
#' nnObject$results # to display the training information
#' compute(nnObject$nn, trainData) # use learned network to predict the whole train set
nnTrain <- function(trainData, type="unspecified", allConverge = TRUE,  h,
                    rep=1, threshold, stepmax=100000, lifesign=1000, minStep = 0, cross = 1, errorFunction=NULL, scaleFunction = NULL,outputs=1) {
  
  features <- ncol(trainData) -1
  
  trainData <- de.factor(trainData)
  
  if (type == "unspecified") {
    if(class(trainData[,ncol(trainData)]) == "factor") {
      type <- "classification"
    } else {
      type <- "regression"
    }
  } else if (type!="regression" & type != "classification"){
    stop("Illegal expression for type: use 'classification' or 'regression'. Default is 'unspecified'.")
  }
  
  if(outputs < 1) {stop("Outputs must be an integer >= 1!")}
  if(outputs > (ncol(trainData)-1)) {stop("Outputs must not exceed the number of columns -1!")}
  if((outputs > 1) & (type=="regression")) {stop("Multiple outputs is only supported for classification. For multiclass regression, use the neuralnet function manually")}
  
  if (type == "classification") {
    resultTable <- data.frame(0,0,0,0,0,0,0,0,0,0,0,0,0,0)
    names(resultTable) <- c("samples", "features", 
                            "h", "rep", "threshold", "steps", "time", "foldSize","trainAcc","trainSens","trainSpec","valAcc","valSens","valSpec")
  } else {
    resultTable <- data.frame(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
    names(resultTable)<- c("samples", "features", "h",
                           "rep", "threshold", "steps", "time", "foldSize",
                           "trainRMSE", "trainMAE", "trainCor", "trainR2",
                           "valRMSE", "valMAE", "valCor", "valR2")
  }
  
  cats("Type: ", type)
  cats("Layers: ", length(h))
  cats("Outsputs: ",outputs)
  cats("Nodes: ", h,"\n\n")
  
  data <- trainData
  
  form <-  paste(paste(names(trainData)[1:outputs],collapse="+"),' ~ ', paste(colnames(trainData)[(outputs+1):ncol(trainData)], collapse ='+'), collapse='')
  cats(form,"\n")
  
  if (cross < 2) {
    folds <- rep(1,nrow(data))
  } else {
    folds <- cut(seq(1,nrow(data)),breaks=cross,labels=FALSE)
  }
  
  ensembleList <- list()
  valScore  <- 0
  bestScore <- 0
  bestNN <- NULL
  
  # K-fold cross validation, training network
  # Returns best model
  i <- 1
  while(i <= cross) {
    cats("\nEpoch",i)
    #Segment your data by fold using the which() function 
    testIndexes <- which(folds==i,arr.ind=TRUE)
    test  <- data[testIndexes, ]
    train <- data[-testIndexes, ]
    if (cross == 1) {train <- test}
    
    result <- nnTrainData(trainData  = train,
                          testData   = test,
                          effectSize = NA,
                          effectPerc = NA,
                          h          = h,
                          threshold  = threshold,
                          stepmax    = stepmax,
                          rep        = rep,
                          lifesign   = lifesign,
                          minStep    = minStep,
                          type       = type,
                          outputs    = outputs,
                          errorFunction = errorFunction,
                          scaleFunction = scaleFunction
                          
    )
    nn <- result$nn
    

    if (is.null(nn)) {
      if(allConverge) {
        print("Restarting epoch")
      } else {
        i <- i+1
      }
      next;
    } # did not converge

    ensembleList[[i]] <- nn
    
    resultRow <- result$result
    if(type == "classification"){resultRow <- resultRow[,c(-3,-4)]}
    resultRow$samples <- nrow(trainData)
    print(resultRow)
    cats("\n")
    
    
    
    if (type == "classification") {
      if (resultRow$valAcc > bestScore) {
        bestScore <- resultRow$valAcc
        valScore <- valScore + resultRow$valAcc
        bestNN <- nn
      }
    } else {
      if (resultRow$valRMSE < bestScore) {
        bestScore <- resultRow$valRMSE
        valScore <- valScore + resultRow$valRMSE
        bestNN <- nn
      }
    }
    

    resultTable <- addToTable(resultRow, resultTable)
    i <- i + 1
  } # end CV
  
  valScore <- valScore / cross
  
  resultTable <- resultTable[-1,]
  rownames(resultTable) <- seq(length=nrow(resultTable))
  # take average
  resultRow <- as.data.frame(t(colMeans(resultTable)))
  
  print(resultTable)
  
  nn <- ensembleList
  
  object <- list(results = resultRow,
                 resultTable = resultTable,
                 nn = nn
  )
  
  valPred <- calcValPred(object,trainData,type,outputs)
  if(type == "regression") {
    resultRow$valCor <- cor(valPred, data[,1])
    resultRow$valR2 <- r2(valPred, data[,1])
  }
  
  object$valPred <- valPred
  object$results <- resultRow
  
  
  
  return(object)
  
  
}






# Helper function used by nnTrain en nnSimulatedTrain. Use is not recommended.

# Train a neural network on a given data set and test set, with the given parameters as arguments.
# A train set should start with features as columns, followed by a "lab" column with labels.
# No cross validation used.

#' Train a neural network, without CV
#'
#' Helper function used by nnTrain en nnSimulatedTrain. Use is not recommended.
#' Train a neural network on a given data set and test set, with the given parameters as arguments.
#' A train set should start with features as columns, followed by a "lab" column with labels.
#' No cross validation used.
#' 
#' @keywords internal
nnTrainData <- function(trainData, testData, effectSize=NA, effectPerc=NA,
                        h, rep=1, threshold, stepmax=100000, lifesign=1000,type, errorFunction = NULL, outputs=1, minStep = 0,scaleFunction=NULL) {
  
  trainData[,1] <- as.numeric(as.character(trainData[,1]))
  testData[,1]   <- as.numeric(as.character(testData[,1]))
  
  if(outputs < 1) {stop("Outputs must be an integer >= 1!")}
  if(outputs > (ncol(trainData)-1)) {stop("Outputs must not exceed the number of columns -1!")}
  if((outputs > 1) & (type=="regression")) {stop("Multiple outputs is only supported for classification. For multiclass regression, use the neuralnet function manually")}
  
  
  features <- ncol(trainData) - 1
  if(is.null(errorFunction)) {errorFunction = "sse"}
  
  form <-  paste(paste(names(trainData)[1:outputs],collapse="+"),' ~ ', paste(colnames(trainData)[(outputs+1):ncol(trainData)], collapse ='+'), collapse='')
  f <- as.formula(form)

  start.time <- Sys.time()
  nn <- tryCatch({neuralnet(f, data=trainData, hidden = h,
                            act.fct = function(x) {1/(1+exp(-x))},
                            lifesign.step = lifesign, lifesign = "full",
                            rep = rep,
                            algorithm = "rprop+",
                            err.fct = errorFunction,
                            stepmax = stepmax, threshold = threshold)},
                 warning=function(w) {NULL}, error = function(w) {NULL})

  end.time <- Sys.time()
  if(is.null(nn)){
    print("Warning, no nn-object saved.")
    return(NULL)}
  if(nn$result.matrix[3,] <= minStep){
    print("Warning, steps did not exceed minStep, no nn-object saved.")
    return(NULL)}
  
  
  

  time <- as.numeric(end.time - start.time)
  time <- time/rep
  
  samples <- nrow(trainData)
  
  error <- mean(nn$result.matrix[1,])
  steps <- mean(nn$result.matrix[3,])
  
  # Try model on train set
  train <- trainData[,-1:-outputs]
  test  <- testData[,-1:-outputs]
  predTrain <- compute(nn, train)$net.result
  predTest  <- compute(nn, test)$net.result
  
  if (!is.null(scaleFunction)) {
    predTrain <- scaleFunction(predTrain)
    predTest <- scaleFunction(predTest)
    trainData[,1] <- scaleFunction(trainData[,1])
    testData[,1] <- scaleFunction(testData[,1])
  }

  # MULTICLASS
  if(outputs > 1) {
    dlist <- list()
    for(i in 1:outputs) {dlist[[i]] <- c(-1,1)}
    d <- expand.grid(dlist)
    names(d) <- names(trainData[,1:outputs])
    d$lab <- multiToOne(d)
  
    labelsTrain <- multiToOne(trainData[,1:outputs],levs=levs)
    labelsTest <- multiToOne(testData[,1:outputs],levs=levs)
    
    predTrain <- apply(predTrain,2,sign)
    predTest <- apply(predTest,2,sign)
    
    predTrain <- multiToOne(predTrain,levs=levs)
    predTest <-  multiToOne(predTest,levs=levs)
    
    prTrain <<-predTrain
    nn.tab <- predTable(predTrain, labelsTrain,newLevels=FALSE)
    trainAcc <- classAgreement(nn.tab)$diag
    trainSens <- NA
    trainSpec <- NA
    
    prTest <<- predTest
    labTest <<- labelsTest
    nn.tab <- predTable(predTest,labelsTest,newLevels=FALSE)
    cats("Validation predictions:")
    print(nn.tab)
    accTest <- classAgreement(nn.tab)$diag
    valSens <- NA
    valSpec <- NA
    
    newRow <- data.frame(samples, features, effectSize, effectPerc, h,
                         rep, threshold, steps, time,nrow(train), trainAcc, trainSens, trainSpec, accTest, valSens, valSpec)
    names(newRow)<- c("samples", "features", "effectSize", "effectPerc", "h",
                      "rep", "threshold", "steps", "time", "foldSize","trainAcc","trainSens","trainSpec","valAcc","valSens","valSpec")
    
    
  }

  # ERROR CLASSIFICATION
  else if(type == "classification") {
    labelsTrain <- as.factor(trainData[,1])
    labelsTest <- factor(testData[,1], levels=levels(labelsTrain))
    
    predTrain <- factor(sign(predTrain), levels=levels(labelsTrain))
    nn.tab <- predTable(predTrain, labelsTrain)
    trainAcc <- classAgreement(nn.tab)$diag
    trainSens <- nn.tab[1,1] / (nn.tab[1,1] + nn.tab[2,1])
    trainSpec <- nn.tab[2,2] / (nn.tab[2,2] + nn.tab[1,2])
    
    predTest <- factor(sign(predTest), levels=levels(labelsTrain))
    nn.tab <- predTable(predTest,labelsTest)
    cats("Validation predictions:")
    print(nn.tab)
    accTest <- classAgreement(nn.tab)$diag
    valSens <- nn.tab[1,1] / (nn.tab[1,1] + nn.tab[2,1])
    valSpec <- nn.tab[2,2] / (nn.tab[2,2] + nn.tab[1,2])
    if(is.na(valSens)) {valSens <- 1}
    if(is.na(valSpec)) {valSpec <- 1}
    if(is.na(trainSens)) {trainSens <- 1}
    if(is.na(trainSpec)) {trainSpec <- 1}

    
    newRow <- data.frame(samples, features, effectSize, effectPerc, h,
                         rep, threshold, steps, time,nrow(train), trainAcc, trainSens, trainSpec, accTest, valSens, valSpec)
    names(newRow)<- c("samples", "features", "effectSize", "effectPerc", "h",
                      "rep", "threshold", "steps", "time", "foldSize","trainAcc","trainSens","trainSpec","valAcc","valSens","valSpec")
    
    
  } else {
    
    # ERROR REGRESSION

    trainData$diff <- trainData[,1] - predTrain
    trainRMSE <- sqrt(mean(trainData$diff^2))
    trainMAE  <- mean(abs(trainData$diff))
    trainCor  <- cor(trainData[,1], predTrain)
    trainR2  <- summary(lm(trainData[,1] ~ predTrain))$r.squared
    
    testData$diff <- testData[,1] -predTest
    testRMSE <- sqrt(mean(testData$diff^2))
    testMAE  <- mean(abs(testData$diff))
    testCor  <- cor(testData[,1], predTest)
    testR2  <- summary(lm(testData[,1] ~ predTest))$r.squared
    
    newRow <- data.frame(samples, features, h,
                         rep, threshold, steps, time, nrow(train),
                         trainRMSE, trainMAE, trainCor, trainR2,
                         testRMSE,  testMAE,  testCor,  testR2)
    names(newRow)<- c("samples", "features", "h",
                      "rep", "threshold", "steps", "time", "foldSize",
                      "trainRMSE", "trainMAE", "trainCor", "trainR2",
                      "valRMSE", "valMAE", "valCor", "valR2")
    
    
  }
  if(outputs>1){d <<- d}
  object <- list(result = newRow,
                 nn = nn
  )
  return(object)
}










#' Training a neural network on simulated data
#' 
#' Trains a neural network using K-fold cross validation to get a validation accuracy, which is the average of the K
#' validation accuracies. Should be used on data generated with using the "generateData" function only. If exactly one feature
#' is relevant, a hypothetical accuracy will be added to the results which shows the accuracy belonging to the optimal decision boundary for
#' that feature.
#' Returns an object containing two data frames with result data and the neural network itself.
#' @param trainData a data frame which will be used for training. Must consists of columns for every
#' feature, followed by one column containing the correct labels.
#' @param type decides whether classification or regression will be used. By default will be set to "classification" when the label column is factor and to
#' "regression" otherwise.
#' @param allConverge a logical value indicating whether training epochs should be restarted when the network did not converge in a certain step. When set to TRUE,
#' the size of the \code{cross} parameter indicates the amount of results in the \code{resultTable}.
#' @param meansDiff a numeric value that should specify the difference in means for relevant features, like the value used in the \code{\link{generateData}} function.
#' @param effectPerc a numeric value between 0 and 1 that should specify the percentage of features that have difference in means, like the value used in the \code{\link{generateData}} function.
#' @param h a numeric value specifying the amount of nodes in the hidden layer. For more hidden layers, use a vector of numeric values.
#' @param rep a number specifying amount of repetitions done per cross validation step. For example, rep = 5 and cross = 10
#' will lead to 50 reps in total.
#' @param threshold a numeric value between 0 and 1 specifying the threshold for the partial derivatives of the error function as stopping criteria.
#' When the error does not decrease more than this percentage in a step, training terminates.
#' @param stepmax the maximum steps for the training of the neural network. Reaching this maximum will terminate the training process.
#' @param lifesign an integer specifying how often (the step size) the learning algorithm should print the minimal threshold.
#' @param cross an integer specifying the K for K-fold cross validation. A value of 1 will result in no cross validation and testing
#' on the training set itself; a value equal to the sample size of the train data will result in leave-one-out cross validation.
#' 
#' @seealso \code{\link{nnTrain}} is a wrapper for this function and is recommended for use on actual data sets
#' 
#' @import neuralnet
#' 
#' @export
#' 
#' @return An object of class \code{"nnsvm"}, containing the following components:
#' \item{results}{a data frame consisting of one row with information about the training process, like the sample size in
#' the train data, used parameters, average training steps and accuracies. The values are an average over the resultTable.  }
#' \item{resultTable}{a data frame consisting of information about every cross validation step of the training process. }
#' \item{nn}{an object of class \code{"nn"} from the \code{\link[neuralnet]{neuralnet}} package, containing the learned network.}
#' @examples
#' # train a network with two hidden layers and 5-fold cross validation
#' nnObject <- nnTrain(trainData, h=c(2,3), threshold=0.01, cross=5)
#' # to display the training information
#' nnObject$results 
#' # use learned network to predict the whole train set
#' compute(nnObject$nn, trainData) 
nnSimulatedTrain <- function(trainData, type="unspecified", allConverge = FALSE, meansDiff=NA, effectPerc=NA, h,
                             rep=1, threshold, stepmax=100000, lifesign=1000, cross = 1, scaleFunction = NULL) {
  
  features <- ncol(trainData) -1
  
  if (type == "unspecified") {
    if(class(trainData[,ncol(trainData)]) == "factor") {
      type <- "classification"
    } else {
      type <- "regression"
    }
  } else if (type!="regression" & type != "classification"){
    stop("Illegal expression for type: use 'classification' or 'regression'. Default is 'unspecified'.")
  }
  
  
  if (type == "classification") {
    resultTable <- data.frame(0,0,0,0,0,0,0,0,0,0,0)
    names(resultTable) <- c("samples", "features", "effectSize", "effectPerc",
                            "h", "rep", "threshold", "steps", "time", "accTrain","valAcc")
  } else {
    resultTable <- data.frame(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
    names(resultTable)<- c("samples", "features", "h",
                           "rep", "threshold", "steps", "time",
                           "trainRMSE", "trainMAE", "trainCor", "trainR2",
                           "valRMSE", "valMAE", "valCor", "valR2")
  }
  
  data <- trainData
  
  if (cross < 2) {
    folds <- rep(1,nrow(data))
  } else {
    folds <- cut(seq(1,nrow(data)),breaks=cross,labels=FALSE)
  }
  
  valScore  <- 0
  bestScore <- 0
  bestNN <- NULL
  
  # K-fold cross validation, training network
  # Returns best model
  i <- 1
  while(i <= cross) {
    cat("\nEpoch",i)
    #Segment your data by fold using the which() function 
    testIndexes <- which(folds==i,arr.ind=TRUE)
    test  <- data[testIndexes, ]
    train <- data[-testIndexes, ]
    if (cross == 1) {train <- test}
    result <- nnTrainData(trainData  = train,
                          testData   = test,
                          effectSize = effectSize,
                          effectPerc = effectPerc,
                          h          = h,
                          threshold  = threshold,
                          stepmax    = stepmax,
                          rep        = rep,
                          lifesign   = lifesign,
                          type       = type
                          
    )
    resultRow <- result$result
    resultRow$samples <- nrow(trainData)
    print(resultRow)
    resultTable <- resultTable
    nn <- result$nn
    
    if (is.null(nn)) {
      print("Restarting epoch")
      if(allConverge) {i <- i+1}
      next;
    } # did not converge
    
    
    if (type == "classification" && resultRow$valAcc > bestScore) {
      if (resultRow$valAcc > bestScore) {
        bestScore <- resultRow$valAcc
        valScore <- valScore + resultRow$valAcc
        bestNN <- nn
      }
    } else {
      if (resultRow$valRMSE < bestScore) {
        bestScore <- resultRow$valRMSE
        valScore <- valScore + resultRow$valRMSE
        bestNN <- nn
      }
    }
    
    resultTable <- addToTable(resultRow, resultTable)
    i <- i + 1
  } # end CV
  
  valScore <- valScore / cross
  
  resultTable <- resultTable[-1,]
  rownames(resultTable) <- seq(length=nrow(resultTable))
  # take average
  resultRow <- as.data.frame(t(colMeans(resultTable)))
  
  print(resultTable)
  
  object <- list(result = resultRow,
                 resultTable = resultTable,
                 nn = bestNN
  )
  return(object)
  
  
}




# Recalculate pred function
calcValPred <- function(res, data, type="regression",outputs=1,decisionValues=FALSE) {
  isSVM <- !is.null(res$svm)
  if(isSVM) {models <- res$svm
  } else {models <- res$nn
  data <- as.data.frame(lapply(data,de.factor))}
  
  if(outputs==1){valPred <- c()
  }else{
    valPred <- t(as.matrix(c(0,0)))
  }

  if(!(type=="classification" || type == "C-classification" || type =="regression")) {stop("Wrong type")}
  cross <- length(models)

  if (cross == 1) {folds <- rep(1,nrow(data))
  }else{folds <- cut(seq(1,nrow(data)),breaks=cross,labels=FALSE)}
  
  for(i in 1:cross) {
    #Segment your data by fold using the which() function 
    testIndexes <- which(folds==i,arr.ind=TRUE)
    test <- data[testIndexes, ]
    levs <- as.numeric(levels(factor(data[,1])))
    if(ncol(data) <= 2) {
      test <- as.data.frame(test[,-c(1:outputs)])
      names(test) <- names(data)[2]
    } else {
      test <- test[,-c(1:outputs)]
    }
    
    model <- models[[i]]
    pred <- predictModel(model, test,decisionValues=decisionValues)

    if((type=="classification" || type == "C-classification") & outputs == 1) {
      pred <- de.factor(pred)}
    if(outputs==1){valPred <- c(valPred, pred)
    }else         {valPred <- rbind(valPred, pred)
    }
  }
  if((type=="classification" || type == "C-classification") & !is.null(res$svm) & outputs == 1 & !decisionValues) {valPred <- factor(valPred,levels=levs)}
  if(outputs>1){valPred <- valPred[-1,]}
  return(valPred)
}

calcScore <-function(data, means) {
  score <- rep(0,nrow(data))
  for(i in 1:ncol(data)){
    col <- (data[,i] - means[i,1]) / means[i,2] 
    if (i <= 2) {col <- -col}
    score <- score + col
  }
  score
}





#' Training an SVM with double nest cross validation
#' 
#' The inner loop is for tuning the parameters, the outer loop calculates the validation score. To use tuning in the inner loop, 
#' use vector for the parameters with different values to try in grid search.
#' 
#' @param trainData a data frame which will be used for training. Must consists of columns for every
#' features, ended by a column with the labels.
#' @param kernel the kernel used in training and predicting. Options are "linear", "radial", "polynomial" and "sigmoid".
#' @param type a value specifying whether the SVM should be used as a classification or a regression machine. Use "C-classification" or
#' "nu-regression". By default, a decision will be made depending on whether the label column is of class factor.
#' @param scale a logical vector indicating the variables to be scaled. If scale is of length 1, the value is recycled as many times as needed.
#' @param cost a numeric value or vector for the cost parameter of the SVM. Supply a range (vector of different numeric values) of values to try during the cross
#' validation grid search if this is applicable. Needed for all kernels.
#' @param gamma a numeric value or vector for the gamma parameter of the SVM. Supply a range (vector of different numeric values) of values to try during the cross
#' validation grid search if this is applicable. Needed for all kernels except for \code{linear}.
#' @param degree a numeric value or vector for the degree parameter of the SVM. Supply a range (vector of different numeric values) of values to try during the cross
#' validation grid search if this is applicable. Only need for the \code{polynomial} kernel.
#' @param coef a numeric value for the coef parameter of the SVM. Only needed for \code{poylnomial} and \code{sigmoid} kernels. Cannot tuned in grid search.
#' @param outerCross an integer specifying the amount of folds in the outer loop. Choose 1 for no cross validation (train and validation results will be the same).
#' @param innerCross an integer specifying the amount of oflds in the inner tuning loop. For no tuning, supply single values for all needed parameters.
#' 
#' @keywords svm cross validation
#' 
#' @import "e1071"
#' 
#' @export
#' 
#' @return an object (list), containing the following components:
#' \item{results}{a data frame consisting of one row with information about the training process, like the sample size in
#' the train data, used parameters, and accuracies. The values are an average over the resultTable (all CV loops).}
#' \item{resultTable}{a table containing information about every outer loop, including the best parameters found in the inner loop and the performance on the validation fold.}
#' \item{svm}{the best found model SVM, which is an object of class \code{"svm"} from the \code{\link{e1071}} package.}
#' 
#' @examples
#' # train an RBF regression SVM and tune gamma. 
#' svm.result <- svmTrain(data, kernel = "radial", type="regression", gamma=c(0.01, 0.1, 1, 2, 4), cost = 1)
#' # get the table of results
#' svm.tab <- svm.results$resultTable
#' # get the resulting SVM model
#' svm.model <- svm.result$svm
#' # predict on data
#' predict(svm.model, data)
svmTrain <- function(trainData, kernel = "linear", type="unspecified", scale = FALSE,
                     gamma  = 2^(-2:2),
                     cost   = 2^(-2:2),
                     degree = 1:5,
                     coef   = 0,
                     nu     = 0.5,
                     outerCross = 10, innerCross = 5,
                     reverseSign = TRUE,
                     scaleFunction = NULL,
                     classWeights = NULL) {

  data  <- trainData

  outer <- outerCross
  inner <- innerCross

  if (outer > nrow(trainData)) {
    stop("outerCross must not exceed train data size!")
  }
  
  if (outer < 2) {
    folds <- rep(1,nrow(data))
  } else {
    folds <- cut(seq(1,nrow(data)),breaks=outer,labels=FALSE)
  }
  ensembleList <- list()
  
  if (type == "unspecified") {
    if(class(trainData[,ncol(trainData)]) == "factor") {
      type <- "C-classification"
    } else {
      type <- "nu-regression"
    }
  } else if (type == "classification") {type <- "C-classification"
  } else if (type == "regression") {type <- "nu-regression"
  } else {stop("Illegal expression for type. Use 'classification' or 'regression' (default is 'unspecified).'")}
  
  if (type == "C-classification") {
    trainData[,1] <- as.factor(trainData[,1])
    resultTable <- data.frame(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
    names(resultTable) <- c("samples", "features", "kernel", "cost", "gamma", "degree", "outer","inner", "foldSize","trainAcc","trainSens","trainSpec","valAcc","valSens","valSpec")
  } else {
    resultTable <- data.frame(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
    names(resultTable) <- c("samples", "features", "kernel", "cost", "gamma", "degree", "outer","inner","foldSize",
                            "trainRMSE", "trainMAE", "trainCor", "trainR2",
                            "valRMSE", "valMAE", "valCor", "valR2")
  }
  
  if(type== "C-classification") {
    data[,1] <- as.factor(data[,1]) 
    data[,-1] <- as.data.frame(lapply(data[,-1],function(x){as.numeric(as.character(x))}))
  } else if (type != "nu-regression") {stop("Type error")}

  
  features <- ncol(trainData) - 1
  
  bestScore <- 0

  skip <- FALSE
  if(kernel == "polynomial") {
    cats("Cost: ", cost)
    cats("Gamma: ", gamma)
    cats("Degree: ", degree)
    cats("Coef0: ", coef)
  }
  if(kernel == "radial") {
    degree <- 0
    coef<-0
    cats("Cost: ", cost)
    cats("Gamma: ", gamma)
  }
  if(kernel == "linear") {
    degree <- 0
    gamma <- 0
    coef  <- 0
    cats("Cost: ", cost)
  }
  if(kernel == "sigmoid") {
    degree <- 0
    cats("Cost: ", cost)
    cats("Gamma: ", gamma)
    cats("Coef0: ", coef)
  }
  cats("")


  formStr <- paste(names(trainData)[1],' ~ ', paste(colnames(trainData)[2:ncol(trainData)], collapse ='+'), collapse='')
  cats(formStr,"\n")
  f <- as.formula(formStr)
  if(!is.null(classWeights)){cats("Class weights will be aplied. Weights: ", classWeights)}
  if ((length(cost) == 1) && (length(gamma) == 1) && (length(coef) == 1) && (length(coef) == 1)) {skip <- TRUE}
  if(!skip && (inner > 1)) {tune <- TRUE
    cats("Parameters will be tuned.\n")
  } else {tune <- FALSE
    cats("No inner cross validation loop (parameter tuning) is performed.\n")}
  
  if (!is.null(scaleFunction)) {
    cats("Use scaling.\n")
  }
  
  #Perform CV
  for(i in 1:outer){
    
    #Segment your data by fold using the which() function 
    testIndexes <- which(folds==i,arr.ind=TRUE)

    test  <- data[testIndexes, ]
    train <- data[-testIndexes, ]

    if (outer < 2) {train <- test}
    
    if(type=="C-classification"){labels <- factor(train[,1])
    } else {labels <- de.factor(train[,1])}
    if (tune) {
    svm.tuned <- tune.svm(x=train[,-1],y=labels,
                          kernel =  kernel, gamma = gamma, nu = nu, cost = cost, degree = degree, class.weights = classWeights, coef0=coef, scale=scale,
                          tunecontrol = tune.control(cross = inner))
  
    
    params <- svm.tuned$best.parameters
    print(params)
    cats("Tune error: ",svm.tuned$best.performance)
    
    bestDegree<-params[,1]
    bestGamma <-params[,2]
    bestCoef  <-params[,3]
    bestCost  <-params[,4]
    } else {
      bestDegree<-degree
      bestGamma <-gamma
      bestCoef  <-coef
      bestCost  <-cost
    }
    
    bestNu <- nu

    svm.model <- svm(f, data=train, kernel=kernel, type=type, class.weights=classWeights, degree=bestDegree,gamma=bestGamma,coef=bestCoef,cost=bestCost, nu = bestNu, scale=scale)

    if (scale == FALSE) {
      svm.model <- svmWeights(svm.model)
      if ((reverseSign & type=="C-classification") && (length(levels(train[,1])) == 2 & de.factor(train[1,1]) == de.factor(min(levels(train[,1]))))) {
        svm.model$weights <- -svm.model$weights
        svm.model$b <- -svm.model$b
        svm.model$rev <- TRUE
      } else {
        svm.model$rev <- FALSE
      }
    }
    if(is.null(svm.model$rev)) {svm.model$rev <- FALSE}
    svm.model$id <- i
    svm.model$class.weights <- classWeights
    
      ensembleList[[i]] <- svm.model
      

      lev <- levels(as.factor(data[,1]))
      
      if(ncol(train) <= 2) {
        trainT <- as.data.frame(train[,-1])
        names(trainT) <- names(train)[2]
        testT <- as.data.frame(test[,-1])
        names(testT) <- names(test)[2]
      } else {
        trainT <- train[,-1]
        testT <- test[,-1]
      }
      
      svm.trainPred <- predict(svm.model, trainT)
      svm.testPred  <-predict(svm.model, testT)

      svm.testRawPred <- round(predictModel(svm.model, testT,decisionValues=TRUE),2)
      
      if (!is.null(scaleFunction)) {
        svm.trainPred <- scaleFunction(svm.trainPred)
        svm.testPred <- scaleFunction(svm.testPred)
        svm.testRawPred <- scaleFunction(svm.testRawPred)
        train[,1] <- scaleFunction(train[,1])
        test[,1] <- scaleFunction(test[,1])
      }


    if (type == "C-classification") {
      
      svm.trainPred <- factor(svm.trainPred, levels=lev)
      svm.testPred <- factor(svm.testPred, levels=lev)

    	labelsTrain <- factor(train[,1],levels=lev)
    	labelsTest <- factor(test[,1], levels=lev)
	
      svm.tab1 <- predTable(svm.trainPred, labelsTrain)

      trainAcc <- classAgreement(svm.tab1)$diag
      trainSens <- svm.tab1[1,1] / (svm.tab1[1,1] + svm.tab1[2,1])
      trainSpec <- svm.tab1[2,2] / (svm.tab1[2,2] + svm.tab1[1,2])
      
      svm.tab2 <- predTable(svm.testPred, labelsTest)
      testAcc <- classAgreement(svm.tab2)$diag
      cats("Validation accuracy:", testAcc)
      cats("Validation predictions:")
      print(svm.tab2)
      cats(paste(c("on test fold ", testIndexes[1], "-", testIndexes[length(testIndexes)], "."),collapse=""))
      score <- testAcc
      valSens <- svm.tab2[1,1] / (svm.tab2[1,1] + svm.tab2[2,1])
      valSpec <- svm.tab2[2,2] / (svm.tab2[2,2] + svm.tab2[1,2])

      if(is.na(valSens)) {valSens <- 1}
      if(is.na(valSpec)) {valSpec <- 1}
      if(is.na(trainSens)) {trainSens <- 1}
      if(is.na(trainSpec)) {trainSpec <- 1}
      #if(tune){cats(st(svm.tuned$best.parameters))}
      
      cats("Raw pred: ",svm.testRawPred)
      cats("Val pred: ",de.factor(svm.testPred))
      cats("True lab: ",de.factor(labelsTest))
      resultRow <- data.frame(nrow(trainData), features, kernel, bestCost, bestGamma, bestDegree, outer, inner,nrow(train), trainAcc, trainSens, trainSpec, testAcc, valSens, valSpec)
      names(resultRow) <- names(resultTable)
      cats("\n")
      if (score > bestScore) {
        bestScore <- score}
    } else {

      trainDiff <- svm.trainPred - train[,1]
      trainRMSE <- sqrt(mean(trainDiff^2))
      trainMAE  <- mean(abs(trainDiff))
      trainCor  <- cor(svm.trainPred, train[,1])
      trainR2   <- summary(lm(svm.trainPred ~ train[,1]))$r.squared
      
      testDiff <- svm.testPred - test[,1]
      testRMSE <- sqrt(mean(testDiff^2))
      testMAE  <- mean(abs(testDiff))
      testCor  <- cor(svm.testPred, test[,1])
      testR2   <- summary(lm(svm.testPred ~ test[,1]))$r.squared
      score <- testRMSE
      cats("Validation RMSE:", testRMSE,"\n")
      
      resultRow <- data.frame(nrow(trainData), features, kernel, bestCost, bestGamma, bestDegree, outer, inner,nrow(train),
                              trainRMSE, trainMAE, trainCor, trainR2,
                              testRMSE,  testMAE,  testCor,  testR2)
      names(resultRow) <- names(resultTable)
      if (score < bestScore) {
        bestScore <- score}
      
    }

    
    resultTable <- addToTable(resultRow,resultTable)
    
  }
  resultTable <- resultTable[-1,]
  rownames(resultTable) <- seq(length=nrow(resultTable))
  print(resultTable)
  resultRow <- as.data.frame(t(colMeans(resultTable[,c(-3)])))
  resultRow$kernel <- resultTable$kernel[1]
  resultRow <- resultRow[,c(1,2,ncol(resultRow),3:(ncol(resultRow)-1))]
  
  model <- ensembleList

  object <- list(results = resultRow,
                 resultTable = resultTable,
                 svm = model)
  if(type == "nu-regression") {type <- "regression"}
  valPred <- calcValPred(object, data, type)
  if(type == "regression") {
    resultRow$valCor <- cor(valPred, data[,1])
    resultRow$valR2 <- r2(valPred, data[,1])
  }
  object$valPred <- valPred
  object$results <- resultRow
  
  return(object)
}

#' Add a row to an existing data frame
#'
#' Used as a helper function for other functions. The row and data frame should have the same amounnt of columns and the same column names.
#' 
#' @param newRow row to be added to existing data frame
#' @param nnTable existing data frame
#' 
#' @export
#' 
#' @return A new data frame, with \code{newRow} added at the bottom.
addToTable <- function(newRow, nnTable) {
  nnTable <- rbind (nnTable, newRow)
  rm(newRow)
  rownames(nnTable) <- seq(length=nrow(nnTable))
  return(nnTable)
}





#' Uses the given ensemble model and data to predict.
#' 
#' Makes prediction with an SVM or an NN model ensemble. An ensemble is a list of models. 
#' 
#' @param ensemble An SVM or NN ensemble (a list of objects of type \code{svm} or \code{nn}).
#' @param test A dataframe to test on, which should contain the exact same features as the set the models were trained on.
#' Make sure the correct label vector (first column)is removed.
#' @param fun string determining the function to calculate the prediction value. By default the "mean" is taken, but the "median" is possible as well.
#' 
#' @export
#' 
#' @return A numeric vector (NN) or factor (SVM) containing the predictions for each subject.
#' 
#' @examples
#' predictModel(model$nn[[1]], data[,-1]) # test the first model from your ensemble list
#' predictModel(svm, data[,-1]) # predict on model (called svm)
nodeAnalysis <- function(data, nn) {
  sigm <- function(x) {1/(1+exp(-x))}
  
  hTable <- data.frame(x = rep(0,nrow(data)))
  
  layer1 <- as.matrix(nn$weights[[1]][[1]][-1,])
  
  len <- length(nn$weights[[1]][[1]][1,])
  len2<- ncol(data)
  for(i in 1:len) {
    hTable[,paste(c("h",i), collapse="")] <- rep(0, nrow(data))
  }
  
  wLayer2 <- nn$weights[[1]][[2]][-1,]
  
  for(i in 1:nrow(data)) {
    
    p1 <- as.matrix(as.vector(data[i,]))
    
    inputs  <- (p1 %*% layer1) + nn$weights[[1]][[1]][1,]
    
    
    for(j in 1:len) {
      bias <- nn$weights[[1]][[2]][1,1]/len
      outputs <- sigm(inputs) *  wLayer2 + bias
      hTable[,paste(c("h",j), collapse="")][i] <- outputs[j]
    }
    
  }
  hTable <- hTable[,-1]
  return(hTable)
}



weightAnalysis <- function(nn, scale=TRUE) {
  w <- rowSums(abs(nn$weights[[1]][[1]]))[-1]
  w <- data.frame(t(w))
  names(w) <- nn$model.list$variables
  if(scale){w <- w/rowSums(w) * 100}
  w
}


svmWeights <- function(svm) {
  svm$weights <- t(svm$coefs) %*% svm$SV;  
  svm$b <- -1 * svm$rho;
  svm
}

svmEnsemble <- function(svm, test) {
  N <- length(svm[[1]]$weights)
  w <- rep(0,N)
  b <- svm[[1]]$b
  for(i in 1:length(svm)) {
    w <- w + as.vector(svm[[i]]$weights)
    b <- b + svm[[i]]$b
  }
  w <- w / length(svm)
  b <- b / length(svm)
  
  pred <- (as.matrix(test) %*% w) + b
  as.vector(pred)
}


rmse <- function(pred, corr) {
  sqrt(mean((pred - corr)^2))
}

mae <- function(pred, corr) {
  mean(abs(pred-corr))
}

r2 <- function(pred, corr) {
  summary(lm(pred ~ corr))$r.squared
}

#' Calculates modus (most occuring value) of a vector.
#' 
#' Ties are broken by choosing the first occurence.
#' 
#' @param x A numeric vector or factor.
#' 
#' @return A numeric value, the modus.
#' 
#' @export
#' 
#' @examples
#' modus(c(2,4,5,4,5)) # returns 4
modus <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

#' Uses the given model and data to predict.
#' 
#' Makes prediction with an SVM or an NN model.
#' 
#' @param model An SVM or NN model of type \code{svm} or \code{nn}.
#' @param test A dataframe to test on, which should contain the exact same features as the set the model was trained on.
#' Make sure the true label vector (first column) s removed.
#' 
#' @export
#' 
#' @return A numeric vector (NN) or factor (SVM) containing the predictions for each subject.
#' 
#' @examples
#' predictModel(model$nn[[1]], data[,-1]) # test the first model from your ensemble list
#' predictModel(svm, data[,-1]) # predict on model (called svm)
predictModel <- function(model, test,outputs=1,decisionValues=FALSE) {
  cl <- class(model)
  cl <- cl[[length(cl)]]
  if(cl=="nn" & decisionValues) {stop("decisionValues=TRUE only meaningful for SVM.")}
  if (cl=="nn"){return(compute(model,test)$net.result)}
  if (cl=="svm"){
    if (decisionValues) {
      pred <- as.numeric(attributes(predict(model,test,decision.values=decisionValues))$decision.values)
      if(model$rev){return(-pred)
      } else       {return(pred)}
    } else {return(predict(model,test))}
  }
  if (cl=="list"){stop("For training an ensemble, using predictEnsemble. Argument must be of type nn or svm.")
  } else {stop("Argument must be of type nn or svm")}
}

#' Uses the given ensemble model and data to predict.
#' 
#' Makes prediction with an SVM or an NN model ensemble. An ensemble is a list of models. 
#' 
#' @param ensemble An SVM or NN ensemble (a list of objects of type \code{svm} or \code{nn}).
#' @param test A dataframe to test on, which should contain the exact same features as the set the models were trained on.
#' Make sure the correct label vector (first column)is removed.
#' @param fun string determining the function to calculate the prediction value. By default the "mean" is taken, but the "median" is possible as well.
#' 
#' @export
#' 
#' @return A numeric vector (NN) or factor (SVM) containing the predictions for each subject.
#' 
#' @examples
#' predictModel(modelEnsemble, data[,-1]) # test the first model from your ensemble list
predictEnsemble <- function(ensemble, test, fun="mean") {
  
  if(class(ensemble) != "list") {
    warning("Single model instead of list was supplied; result was predicted with single model.")
    if (class(ensemble)=="nn"){return(compute(ensemble,test)$net.result)}
    if (class(ensemble)=="svm"){return(predict(ensemble,test))
    } else {return(NULL)}
  }


  cl <- class(ensemble[[1]])
  cl <- cl[[length(cl)]]
  if ((cl != "nn") && (cl != "svm")) {
    stop("Wrong type of models")
  }
    
  df <- data.frame(x = rep(0,nrow(test)))
  
  for(i in 1:length(ensemble)) {
    if(cl == "nn")   {pr <- compute(ensemble[[i]],test)$net.result
    } else           {pr <- predictModel(ensemble[[i]],test,decisionValues=TRUE)}
    df[,paste(c("model", i), collapse = "")] <- pr
  }
  df <- as.matrix(df[,-1])

  pred <- rep(0,nrow(test))
  for(i in 1:nrow(test)) {
    if (fun == "mean") {

      pred[i] <- mean(df[i,])
    }else if (fun == "median") {pred[i] <- median(df[i,])
    }else {stop("Unknown function.")}
  }

  return(pred)
}

#' Convert a numeric vector or dataframe to a version without any factors.
#' 
#' @param x A numeric vector or dataframe with numeric vectors.
#' 
#' @return A numeric vector or dataframe with only numerics.
de.factor <- function(x) {
  if ((class(x) == "data.frame")[1]) {
    x <- as.data.frame(lapply(x,de.factor))
  } else {
    x<- as.numeric(as.character(x))
  }
  x
}






#' Generate simple simulated train data
#' 
#' Generates a data set with normal distributed features, where part of the features is relevant (an effect size > 0) while the
#' others are noise. The standard deviation is set to 1 by default to make all effect sizes equal to the difference in means. If this
#' value is changed, the meaning of the the effect size column in the results of training will lose their meaning, 
#' so this is not recommended.
#' 
#' @param samples an integer specifying the amount of train samples the train data set should have. The parameter \code{ratio} will determine
#' the size of the test set.
#' @param features an integer specifying the amount of features in the data. A column will be created for each feature.
#' @param meansDiff a numeric value specifying the differences in means between healthy and controls for the relevant features. When sd=1, we can
#' call this the "effect size". 
#' @param effectPerc a numeric value between 0 and 1 specifying the percentage of features that should be relevant, meaning their effect size
#' will be determined by the \code{effectSize} parameter. All other features will be noise and of no discriminative worth.
#' @param sd the standard deviation for all features. Defaults to 1, which is the recommended value. When not equal to 1, the results when other
#' functions may refer to effect size while this will be in fact just the difference in means.
#' @param ratio a nuermic value between 0 and 1 which decides the size of the test set. The train data will always have the same amount of samples as \code{samples}, and ratio determines
#' the percentage of the whole set that makes up the train set.
#' 
#' @export
#' 
#' @return An object (list), containing the following components:
#' \item{trainData}{a data frame with the train data}
#' \item{testData}{a data frame with the test data}
#' \item{meansTable}{a data frame which shows the means every feature got for each group}
generateData <- function(tSamples, features, meansDiff=1, effectPerc=0.2, sd=1,  ratio=1) {
  samples   <- round(tSamples/ratio)
  N         <- round(samples/2)
  samples   <- N * 2
  effectSize <- meansDiff
  
  if(sd != 1){warning("Standard deviation is not equal to zero")}
  if(effectSize == 0) {warning("Effect size is zero, meaning effectPerc has no effect and all features are noise")}
  
  # columnnames creation
  num   <- as.character(1:features)
  name  <- rep("f",features)
  names <- paste(name,num, sep='')
  
  meansH  <- c()
  meansSz <- c()
  dataSet <- data.frame(names = 1:(N*2))
  for(i in 1:features) {
    mean  <- sample(1:5,1)
    mean2 <- mean
    if (i <= effectPerc * features) {mean2 <- mean + effectSize}
    
    dataSet[names[i]] <- c(rnorm(N,mean, sd), rnorm(N, mean2, sd))
    meansH  <- c(meansH, mean)
    meansSz <- c(meansSz, mean2)
  }
  # add labels, save mean
  dataSet <- dataSet[,2:(features+1)]
  means <- data.frame(h = meansH, sz = meansSz)
  labels <- factor(c(rep(-1, samples/2), rep(1,samples/2)))
  dataSet$lab <- labels
  
  # shuffle data, split in train and test
  shuffleData <- dataSet[sample(nrow(dataSet)),] 
  trainData <- shuffleData[1:tSamples,]
  testData <- shuffleData[(tSamples+1):(N*2),]
  rm(shuffleData)
  # reset rownumbers
  rownames(trainData) <- seq(length=nrow(trainData))
  rownames(testData)  <- seq(length=nrow(testData))
  
  if (ratio==1) {
    object <- list(trainData = trainData,
                   meansTable = means
    )
    
  } else {
    object <- list(trainData = trainData,
                   testData  = testData,
                   meansTable = means
    )
  }
  return(object)
}

#' Shortcut function for concatenating strings and numerics.
#' 
#' Concatenates easier than \link{paste}, always without spaces.
#' 
#' @param ... One or more strings concatenate.
#' 
#' @return One new string.
#' 
#' @examples
#' st("one","two", 3) # to get "onetwo3"
st <- function(x,...) {
  paste(c(x,...),collapse="")
}

#' Function for printing numerics without any row numbers.
#' 
#' Should only be used for printing purposes. Does not work for tables.
#' 
#' @param ... One or more string to print.
cats <- function(x,...) {
  cat(x,...,"\n")
}

#' Get a confusion matrix for classification results.
#' 
#' The vectors are converted to factors, making sure the high labels (1) occur first in the table, before (-1). Use the function
#' \link(classAgreement()) to get the accuracy.
#' 
#' @param pred A (numeric or factor) vector with the predicted values.
#' @param true A (numeric or factor) vector with the true labels. 
#' 
#' @export
#' 
#' @return A contingency table, an object of class "table" which can easiliy be viewed.
#' 
#' @examples
#' tab <- predTable(c(-1,1,-1,1), c(-1,-1,-1,1)) # make table
#' tab
#' classAgreement(tab)$diag # get accuracy
predTable <- function(pred, true, newLevels=TRUE) {
  if(!is.factor(pred)) {pred<-factor(pred)}
  if(!is.factor(true)) {true<-factor(true)}
  if(newLevels) {
    pred<-factor(pred, levels = rev(sort(as.numeric(levels(pred)))))
    true<-factor(true, levels = rev(sort(as.numeric(levels(true)))))
  } else {true <- factor(true, levels = levels(true)[-4])}
  table(pred,true)
}

#' Print a vector as a column on the screen.
#' 
#' @param vector Vector to be printed.
#' @export
#' 
#' @examples
#' column (1:3)
#' 1
#' 2
#' 3
column <- function(vec){
  cat(de.factor(vec), sep="\n")
}

#' Write a vector to a .txt file.
#' @param vec A vector to be saved to a .txt file.
#' @param file Path to the file where it should be saved.
#' 
#' @export
#' 
#' @examples
#' write.vector(c(1,1,-1,1), "C\\Example\\vector.txt")
write.vector <- function(vec, file) {
  write.table(vec, file=file, row.names=FALSE, col.names=FALSE, quote=FALSE)
}

read.vector <- function(file) {
  read.table(file, header=FALSE)[,1]
}


#' Calculate the accuracy of a confusion matrix (diagonals divided by total).
#' 
#' @param tab A confusion matrix in the form of a (pred) table
#' 
#' @export
#' 
#' @return A numeric value between 0 and 1
#' 
#' @examples
#' acc(predTable(c(1,0,0,1,0), c(1,1,0,0,0)))
acc <- function(tab) {
  classAgreement(tab)$diag
}

crossAcc <- function(pred, corr) {
  1-classAgreement(predTable(pred,corr))$diag
}

#' Calculate the balanced accuracy of a confusion matrix. Each class influences the percentage equally.
#' 
#' @param tab A confusion matrix in the form of a (pred) table
#' 
#' @export
#' 
#' @return A numeric value between 0 and 1
#' 
#' @examples
#' bAcc(predTable(c(1,0,0,1,0), c(1,1,0,0,0)))
bAcc <- function(tab) {
  wDiag(tab)
}

wDiag <- function(tab) {
  n <- ncol(tab)
  s <- 0
  for (i in 1:n) {
    s <- s + (sum(tab) / sum(tab[,i])) * tab[i,i]
  }
  s / (sum(tab) * ncol(tab))
}

which.median <- function(x) which.min(abs(x - median(x)))


multiToOne <- function(dat,levs=NULL,outputs=2) {
  vec <- rep(NA,nrow(dat))
  dat <- dat == 1
  for(i in 1:nrow(dat)) {
    if(dat[i,1] & dat[i,2]) {vec[i]<-"BD"} else
      if(dat[i,1] & !dat[i,2]) {vec[i]<-"MD"} else 
        if(!dat[i,1] & dat[i,2]) {vec[i]<-"--"} else
        {vec[i]<-"HE"}
  }
  return(factor(vec,levels=c("HE","BD","MD","--")))
}

multiToOne2 <- function(dat,levs=NULL,outputs=2) {
  vec <- rep(NA,nrow(dat))
  dat <- dat == 1
  for(i in 1:nrow(dat)) {
    if(dat[i,1] & dat[i,2]) {vec[i]<-"--"} else
    if(dat[i,1] & !dat[i,2]) {vec[i]<-"BD"} else 
    if(!dat[i,1] & dat[i,2]) {vec[i]<-"MD"} else
    {vec[i]<-"HE"}
  }
  return(factor(vec,levels=c("HE","BD","MD","--")))
}

bestThres2 <- function(corr,rawPred) {
  roc <- roc(corr, rawPred)
  rocDF <<- data.frame(sens=roc$sensitivities,spec=roc$specificities)
  
  dists <- c()
  for(i in 1:nrow(rocDF)){
    dists <- c(dists,dist(as.matrix(rbind(c(1,1),rocDF[i,1:2]))))
  }
  bestThres <- roc$thresholds[which.min(dists)]
  
  bestThres
}

colSD <- function(table){
  apply(table,2,sd)
}
