##############
# Use this function to check your results
# DO NOT submit this script
# TA will use something similar to this 
# (in addition to other verifications and possibly a different dataset) to grade your script
#############

rm(list = ls(all = T))

# set working directory to point to the folder containing hw1.R

# source hw1.R
source('./hw1.R')


# Function for calculating inter- and intra- species distance.
inter_intra_species_dist <- function(data, distance_func_name) {
  # NOTE: This function has already been implemented for you.
  # DO NOT modifiy this function.
  # Input: data: type: dataframe with dimension n_samples x attribute number, 
  # where n_samples is total # of samples (60 in the dataset supplied to you) and
  # attribute number is the total # of attributes (5 in the dataset supplied to you with the final one as the class value).
  # Input: method_name: type: string, can be one of the following values: ('euclidean', 'cosine', 'l_inf')
  # This function will display a matrix, with the mean intra-species-distance, the mean inter-species-distance, the ratio of the two distances.
  # This function has already been implemented for you. It takes the data matrix and method name, outputs the distance
  n = nrow(data)
  distances <- data.frame(distance=numeric(n*n), species=rep(data$Species, n), same_species=logical(n*n))
  if (distance_func_name == "pc1") {
    pc1 <- principal_component_analysis(data, 1)
  }
  for (i in 1:n) {
    for (j in 1:n) {
      if (distance_func_name == "euclidian") {
        dis <- calculate_euclidean(data[i,1:4], data[j,1:4])
        print("Euclid")
        print(dis)
      } else if (distance_func_name == "cosine") {
        dis <- calculate_cosine(data[i,1:4], data[j,1:4])
      } else if (distance_func_name == "l_inf") {
        dis <- calculate_l_inf(data[i,1:4], data[j,1:4])
      } else if (distance_func_name == "pc1") {
        dis <- pc1_distance(data[i,1:4], data[j,1:4], pc1)
      }
      
      distances[(i-1)*n+j,]$distance <- dis
      distances[(i-1)*n+j,]$same_species <- data$Species[i] == data$Species[j]
    }
  }
  print(ddply(distances, c("species"), summarize,
        mean_intra_dis=mean(distance[same_species]),
        mean_inter_dis=mean(distance[!same_species]),
        ratio=mean_inter_dis/mean_intra_dis))
}

# Part 1

# read data in matrix format
iris <- read_data('./iris.csv')

# make sure you read the dataset correctly
summary(iris)

# Should output 2.236068
calculate_euclidean(c(3, 6), c(1, 7))

# Make your own testcase!
calculate_cosine(c(3, 6), c(1, 7))

# Make your own testcase!
calculate_l_inf(c(3, 6), c(1, 7))

# Investigate how well each distance metric distinguishes between species
inter_intra_species_dist(iris, 'euclidian')
inter_intra_species_dist(iris, 'cosine')
inter_intra_species_dist(iris, 'l_inf')

# Part 2: PCA

# PC1's weight for Septal.Length should be 0.3582955
principal_component_analysis(iris, 1)
# Investigate the other PCs
principal_component_analysis(iris, 2)
principal_component_analysis(iris, 3)
principal_component_analysis(iris, 4)

pc1 <- principal_component_analysis(iris, 1)
# calculate PC1 for the first object in the iris dataset: Should be 2.734055
principal_component_calculation(iris[1,1:4], pc1)

# Part 3: PC1 distance

# Calculate PC1 distance between the first and second objects in the iris dataset
# Should be 1.774914
pc1_distance(iris[1,1:4], iris[2,1:4], pc1)

# Part 4: Compare euclidian and pc1 distances

inter_intra_species_dist(iris, 'euclidian')
inter_intra_species_dist(iris, 'pc1')
