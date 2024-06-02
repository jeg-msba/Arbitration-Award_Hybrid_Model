###############
library(plotly)
library(dbscan)
library(factoextra)
library(NbClust)
library(gridExtra)
library(cluster)
library(Rtsne)
library(ggplot2)
library(dplyr)
library(fossil)
library(readxl)
library(caret)
library(mosaic)
################

# In this program, MLB player statistics and salary will be used to perform in-sample salary predictions
# for arbitration eligible players. An initial regression is run to provide a baseline. Next, a cluster 
# analysis is performed to group players with similar statistics and salary. Four different clustering
# methods are applied. A similarity matrix is created and a heat map is plotted. The cluster that seems to 
# differentiate the players the best is chosen. Regression models are applied to each cluster and the model 
# with the lowest RMSE is chosen for each cluster. The last step is to predict the salary for each of the 
# arbitration players and compare to their actual salary.

# Read in MLB file. Removed players without salaries #
MLB <- read.csv("~/Documents/Bentley Courses/ST 635/MLB-Position-Players-Stats-and-Salaries-2023.csv")
colnames(MLB)[colnames(MLB) == 'Arbitration.Eligible'] = 'Arbitration'

# Create scatter plot to see correlation between variables #
pairs(MLB[4:(length(MLB)-1)])

# Drop players with less than 60 games. #
MLB = subset(MLB,MLB$AB>60, drop=TRUE)

# Divide salary by 1000, as its value is so much higher than the other variables #
MLB[,4] = (MLB[,4])/1000
row.names(MLB) = NULL

# Create response and predictor variables #
predictors = MLB[,5:(length(MLB)-1)]
salaries = data.frame(MLB[,4])
colnames(salaries) = "Salary"

#############################################
## Unsupervised learning, cluster analysis ## 
#############################################
# For clustering we will include predictors and salaries. 
# Some clustering algorithms do better with scaled data
mlb.data = predictors
mlb.data["salaries"]=salaries$Salary
scaled.data = scale(mlb.data)

##################################
## Start with DBSCAN clustering ##
##################################
#  Work out a good value for epsilon, m #
m=6
epsilon=850
par(mfrow = c(1, 1))
kNNdistplot(mlb.data, minPts = m)
abline(h=epsilon)

# Do the actual clustering #
clust= dbscan(mlb.data, eps = epsilon, minPts = m)
clust

# who belongs where #
dbscanclusters=clust$cluster 
hullplot(scaled.data, clust) # plot the clusters #

##########################################
## Now lets try hierarchical clustering ##
##########################################
DMatrix=dist(scaled.data)
hc.complete=hclust(DMatrix,method = "complete")
plot(hc.complete); rect.hclust(hc.complete,k=2,border ="green")
plot(hc.complete); rect.hclust(hc.complete,k=3,border ="green")
plot(hc.complete); rect.hclust(hc.complete,k=4,border ="green")
plot(hc.complete); rect.hclust(hc.complete,k=5,border ="green")
plot(hc.complete); rect.hclust(hc.complete,k=6,border ="green")

# Use elbow and silhouette to pick the "right" number of clusters #
fviz_nbclust(scaled.data, kmeans, diss = DMatrix, method = "wss",k.max = 10)+
  geom_vline(xintercept = 4, linetype = 2)+
  labs(subtitle = "Elbow method, Unconditional")
fviz_nbclust(scaled.data, kmeans,diss = DMatrix, method = "silhouette",k.max = 15)+labs(subtitle = "Silhouette method")
fviz_nbclust(scaled.data, kmeans, method = "gap_stat",k.max = 10)+geom_vline(xintercept = 9, linetype = 2)+labs(subtitle = "Gap method")

# Final decision: k = 4
h.k = 4
h.eucl.clust=cutree(hc.complete,k=h.k)
table(h.eucl.clust)

############################
## Now K-Means clustering ##
############################
set.seed(2)
km2.output=kmeans(scaled.data,2,nstart = 20)
km3.output=kmeans(scaled.data,3,nstart = 20)
km4.output=kmeans(scaled.data,4,nstart = 20)
km5.output=kmeans(scaled.data,5,nstart = 20)

# Plot elbow curve to help decide number of clusters #
fviz_nbclust(scaled.data, kmeans, method = "wss")

# Plot clusters, using 2 through 5 #
p2 <- fviz_cluster(km2.output, geom = "point", data = scaled.data) + ggtitle("k = 2")
p3 <- fviz_cluster(km3.output, geom = "point",  data = scaled.data) + ggtitle("k = 3")
p4 <- fviz_cluster(km4.output, geom = "point",  data = scaled.data) + ggtitle("k = 4")
p5 <- fviz_cluster(km5.output, geom = "point",  data = scaled.data) + ggtitle("k = 5")
grid.arrange(p2, p3, p4, p5, nrow = 2)

# make a decision on number of clusters #
km.k=5
km.output=kmeans(scaled.data,km.k,nstart = 20)
table(km.output$cluster)
km.clusters=km.output$cluster
print(p5)

##################################
# Clustering around PAM          #
##################################

# There are no qualitative factors, so we use euclidean distance, not Gower #
dist.data=daisy(mlb.data,metric = "euclidean")
summary(dist.data)

# Silhouette curve to measure width between clusters #
silhouette <- c()
silhouette = c(silhouette, NA)
for(i in 2:10){
  pam_clusters = pam(as.matrix(dist.data),
                     diss = TRUE,
                     k = i)
  silhouette = c(silhouette ,pam_clusters$silinfo$avg.width)
}
plot(1:10, silhouette,
     xlab = "Clusters",
     ylab = "Silhouette Width")
lines(1:10, silhouette)

# Choose number of clusters. Investigate the medoids (centers) of each cluster #
pam.k = 5
pam_gower_data=pam(dist.data, diss = TRUE, k = pam.k)
pam_gower_data$clusinfo
pam_gower.clust=pam_gower_data$clustering
mlb.data[pam_gower_data$medoids, ]
table(pam_gower.clust)

# Use TSNE to see the clustering in 2-D #
tsne_object <- Rtsne(dist.data, is_distance = TRUE, perplexity=10)
tsne_df <- tsne_object$Y %>% data.frame() %>% setNames(c("X", "Y")) %>% mutate(cluster = factor(pam_gower_data$clustering))
ggplot(aes(x = X, y = Y), data = tsne_df) +geom_text(aes(label=seq(1,nrow(tsne_df),1)),size=3)+
  geom_point(aes(color = cluster))

# check the agreement #
Cluster.Assignment=data.frame(as.vector(dbscanclusters),as.vector(h.eucl.clust),as.vector(km.clusters),as.vector(pam_gower.clust))
colnames(Cluster.Assignment)<-c("DBScan","Hierachical","KMeans", "PAM-gower")
rownames(Cluster.Assignment)<-as.factor(seq(1,nrow(scaled.data),1))

# See which clusters are most similar #
similarity.matrix=matrix(0,ncol(Cluster.Assignment),ncol(Cluster.Assignment))
for(i in 1:ncol(Cluster.Assignment))
{
  for(j in 1:ncol(Cluster.Assignment))
  {
    similarity.matrix[i,j]=rand.index(Cluster.Assignment[,i],Cluster.Assignment[,j])
  }
}
similarity.matrix
rownames(similarity.matrix)=colnames(Cluster.Assignment)
colnames(similarity.matrix)=colnames(Cluster.Assignment)

# Plot heat map
similarity.matrix
plot(hclust(as.dist(1-similarity.matrix),method = "complete"))
heatmap(similarity.matrix,Rowv = NA, Colv = NA,scale = "none")

########################################
#Post Clustering - Regression Analysis #
########################################
model_clusters <- function(clusters, k) {
  # Create each regression model for each cluster                     #
  # Select the regression model with the lowest RMSE for each cluster #
  
  # Initialize results table and model lists #
  results.table <- data.frame(matrix(ncol = 8, nrow = k))
  colnames(results.table) <- c("knn-rmse", "tree-rmse", "bag-rmse", "rf-rmse", "knn-rsq", "tree-rsq", "bag-rsq", "rf-rsq")
  knn.mod = tree.mod = bag.mod = rf.mod = list()
  trans.cluster = list()
  
  # For each cluster, generate a knn, tree, bagged tree, and random forest model #
  ctrl=trainControl(method = "repeatedcv", number = 10, repeats=5, savePredictions = TRUE)
  for (i in 1:k) {
    # Create predictors and response for each cluster
    mlb.clust = subset(mlb.data, clusters == i)
    cluster.predictors=subset(mlb.clust, select = - c(salaries))
    cluster.response = data.frame(mlb.clust$salaries)
    colnames(cluster.response) = "salaries"
    
    # Create training and test sets. Use mean salary for selecting balanced partitions
    cluster.hilow = ifelse (cluster.response$salaries <  median(cluster.response$salaries),0, 1)
    tr=createDataPartition(cluster.hilow,p=.9,list = F)
    tr.cluster.pred=cluster.predictors[tr,]
    test.cluster.pred=cluster.predictors[-tr,]
    tr.cluster.response=cluster.response[tr,]
    test.cluster.response=cluster.response[-tr,]
    
    # Scale and center the data
    trans=preProcess(tr.cluster.pred, method = c('knnImpute','center','scale')) #creating the function to transform)
    trans.cluster[[i]] = trans
    trans.tr.cluster.pred=predict(trans,tr.cluster.pred)
    trans.test.cluster.pred=predict(trans,test.cluster.pred)
    
    # Model KNN, Tree, Bagged Tree, Random Forest #
    knn.mod[[i]] = train(x=trans.tr.cluster.pred,y=tr.cluster.response,method="knn",trControl=ctrl)
    tree.mod[[i]] = train(x=trans.tr.cluster.pred,y=tr.cluster.response,method='rpart',trControl=ctrl)
    bag.mod[[i]] = train(x=trans.tr.cluster.pred,y=tr.cluster.response,method='treebag',trControl=ctrl)
    rf.mod[[i]] = train(x=trans.tr.cluster.pred,y=tr.cluster.response,method='cforest',trControl=ctrl)
    
    #Save the results in a list. Training models can't be stored in a data.frame #
    results.table[i,"knn-rmse"] = min(knn.mod[[i]]$results$RMSE)
    results.table[i,"tree-rmse"] = min(tree.mod[[i]]$results$RMSE)
    results.table[i,"bag-rmse"] = bag.mod[[i]]$results$RMSE
    results.table[i,"rf-rmse"] = min(rf.mod[[i]]$results$RMSE)
    results.table[i,"knn-rsq"] = max(knn.mod[[i]]$results$Rsquared)
    results.table[i,"tree-rsq"] = max(tree.mod[[i]]$results$Rsquared)
    results.table[i,"bag-rsq"] = max(bag.mod[[i]]$results$Rsquared)
    results.table[i,"rf-rsq"] = max(rf.mod[[i]]$results$Rsquared)
  }
  
  # Create a list of the best model, based on lowest RMSE, for each cluster #
  best.model = list()
  for (i in 1:k) {
    best.rmse = min(results.table[i,1:4])
    if (results.table[i,1] == best.rmse) {
      best.model[[i]] = knn.mod[[i]]
    } else if (results.table[i,2] == best.rmse) {
      best.model[[i]] = tree.mod[[i]]
    } else if (results.table[i,3] == best.rmse) {
      best.model[[i]] = bag.mod[[i]]
    } else if (results.table[i,4] == best.rmse) {
      best.model[[i]] = rf.mod[[i]]
    }
    print(best.model[[i]]$method)
  }
  print(results.table)
  return(list("bestmodel" = best.model, "trans" = trans.cluster))
} # End of model_clusters #

predict_salaries <- function(clusters, best.model, trans) {
  # Predict salaries of arbitration players in each cluster                #
  # Store the actual and predicted salaries plus the difference in a table #
  
  # Initialize table of actual vs. predicted salaries
  predict.table <- data.frame(matrix(ncol = 5, nrow = 0))
  colnames(predict.table) <- c("Player", "Cluster", "Actual_Salary", "Predicted_Salary", "Delta")
  
  # For each arbitration eligible player, predict their salary using their clusters model and transformation
  for (i in 1:length(MLB[,1])) {
    if (MLB[i,"Arbitration"] == 1) {
      new.player = MLB[i,5:(length(MLB)-1)]
      cluster.number = clusters[i]
      trans.player=predict(trans[[cluster.number]],new.player) # keep in mind: the same transformation!! #
      predicted.salary=predict(best.model[[cluster.number]], newdata = trans.player)*1000
      actual.salary = MLB[i,"Salary"]*1000
      one.result = data.frame("Player"=MLB[i,"Player"], "Cluster"=cluster.number, "Actual_Salary"=actual.salary, 
                              "Predicted_Salary"=predicted.salary, "Delta"=round((actual.salary-predicted.salary)/actual.salary,digits=2))
      predict.table = rbind(predict.table, one.result)
    }
  }
  row.names(predict.table) = NULL
  return(predict.table)
} # End of predict_salaries #

# Predict baseline salaries without using clusters #
baseline.cluster = rep(1, length(MLB[,1]))
baseline.model = model_clusters(baseline.cluster, 1)

# Extract model and transformation from the return function #
model = baseline.model$bestmodel
trans = baseline.model$trans
model

# Predict salaries of arbitration players without using clusters #
baseline.salary.predict = predict_salaries(baseline.cluster, model,trans)
baseline.salary.predict
mean(baseline.salary.predict$Delta)
sd(baseline.salary.predict$Delta)

# Create models for each cluster (PAM, Hier, or K-Means) #
clusters = pam_gower_data$clustering
model_list = model_clusters(clusters, pam.k)
# model_list = model_clusters(h.eucl.clust, h.k)
# model_list = model_clusters(km.output$cluster, km.k)

# Extract models and transformations from the returned list #
cluster.model = model_list$bestmodel
cluster.trans = model_list$trans

# Predict salaries of arbitration players using clusters, models, transformations #
cluster.salary.predict = predict_salaries(clusters, cluster.model,cluster.trans)
cluster.salary.predict

# List the arbitration players actual vs. predicted salaries plus statistics on mean, median, sd #
favstats(cluster.salary.predict$Delta ~ cluster.salary.predict$Cluster)
mean(cluster.salary.predict$Delta)
sd(cluster.salary.predict$Delta)


## Perform PCA dimensional reduction ##
stress.data=mlb.data

pr.out=prcomp(stress.data,scale=TRUE)
pr.out$center #--they are de-centered that way--#
pr.out$scale #--they are de-scaled that way---#
pr.out$rotation

par(mfrow=c(1,1))
biplot(pr.out,scale = 0)

pr.out$sdev
pr.var=pr.out$sdev^2

pve=pr.var/sum(pr.var)
pve

par(mfrow=c(2,1))
plot(pve,xlab="Principal Component", ylab="Proportion of variance explained", ylim=c(0,1),type='b')
plot(cumsum(pve),xlab="Principal Component", ylab="Cumulative proportion of variance explained", ylim=c(0,1),type='b')

components <- pr.out[["x"]]
components <- data.frame(components)
components$PC2 <- -components$PC2
components$PC3 <- -components$PC3

## Create an SEM model ##
sem.model <- '
Speed =~ CS+SB+X3B
Efficiency =~ OBP+OPS+SLG
Plays =~ G+AB
Power =~ RBI+HR+X2B
Onbase =~ H+R+BB
Experience =~ Age+salaries
salaries ~ Speed+Efficiency+Plays+Power+Onbase
'

model.estimation <- sem(
  model = sem.model,
  data = scale(mlb.data)
)

par(mfrow=c(1,1))
summary(model.estimation)
summary(model.estimation, fit.measures = TRUE)

#--library(semPlot)---#
semPaths(
  object = model.estimation,
  what = "path",
  whatLabels = "par"
)


# The End #