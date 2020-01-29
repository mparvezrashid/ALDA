###########
# HW1
# Mention your team details here
#
#
#
############


# You may use the following librarie(s):
require(plyr)
# If you get an error when running these lines, 
# make sure to install the respective libraries
#setwd('~/Downloads/ALDA/HW/HW1/HW1_Code')
# read data matrix
read_data <- function(path = "./iris.csv") {
  # Note 1: DO NOT change the function arguments.
  # Input: path: type: string, output: a matrix containing data from iris.csv
  # Write code here to read the csv file as a data frame and return it.
  return(read.csv("./iris.csv"))
}

# Part 1: Distance Measurement
calculate_euclidean <- function(p, q) {
  # Input: p, q are numeric vectors of the same length
  # output: a single value of type double, containing the euclidean distance between p and q.
  return(sqrt(sum((p-q)^2)))
  
}

calculate_cosine <- function(p, q) {
  # Input: p, q are numeric vectors of the same length
  # output: a single value of type double, containing the cosine distance between p and q.
  return(1- (sum(p*q)/sqrt(sum(p^2)*sum(q^2))))
  
}

calculate_l_inf <- function(p, q) {
  # Input: p, q are numeric vectors of the same length
  # output: a single value of type double, containing the l_inf distance between p and q.
  return(max(abs(p-q)))
  
}

# Part 2: principal Component Analysis
principal_component_analysis <- function(data, n){
  # Input: data: the Iris dataframe, with 4 numeric attributes and a 5th nominal class variable
  #        n: the number of the principle component to calculate (e.g. 1 for first principal component)
  # output: a 1 x 4 vector of type double, containing the weights (eigenvector) of the 
  # nth principal component of the dataset.
  #pc<-prcomp(data[,1:4])
  return(prcomp(data[,1:4],rank. = n))
}

principal_component_calculation <- function(p, component_weights){
  # Input: p is a numeric vector of of length n, e.g. representing a row from the original dataset.
  #        component_weights is a vector length n, containing the weights of a principal component
  #        (e.g. the output from running principal_component_analysis)
  # Output: a single value of type double, containing the first principal component value of the sample.
  print(p)
  return(sum(p*component_weights$rotation))
  
}

pc1_distance <- function(p, q, component_weights) {
  # Input: p, q are numeric vectors of of length n, e.g. representing rows from the original dataset.
  #        component_weights is a vector length n, containing the weights of a principal component
  #        (e.g. the output from running principal_component_analysis)
  # output: a single value of type double, containing the distance between p and q, projected onto 
  # the first principal component (i.e. |PC1_p - PC1_q|)
  return(abs(sum(p*component_weights$rotation)-sum(q*component_weights$rotation)))

}
