##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if(!require(stringr)) install.packages("stringr", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip
library(data.table)
library(dplyr)
library(caret)
library(lubridate)
library(stringr)


dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# # if using R 3.6 or earlier
# movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
#                                            title = as.character(title),
#                                            genres = as.character(genres))
# if using R 4.0 or later
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)


# Develop the algorithm using the edx set. For a final test of the algorithm, predict 
# movie ratings in the validation set (the final hold-out test set) as if they were unknown. 

# Extract the movie release year from the title
movie_year <- edx$title %>% str_extract("\\(\\d{4}\\)") %>%
  str_extract("(\\d{4})")
# Calculate the age of the movie
edx        <-  mutate(edx, movie_age = 2020-as.numeric(movie_year))

# Categorizing movies based on their age
edx$age_bins <- cut(edx$movie_age, breaks=c(0,20,40,60,80,105), labels=c("1-20","21-40","41-60","61-80","81-105"))

# Create training and test sets
test_index <- createDataPartition(y = edx$rating, times = 1,
                                  p = 0.2, list = FALSE)
train_set <- edx[-test_index,]
test_set <- edx[test_index,]

# To make sure I don't include users and movies in the test set that do not
# appear in the training set, I removed these using the semi_join function
test_set <- test_set %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# Explore the Movielens Dataset
# Explore Movie age effect
# Plot avg movie rating vs movie age
movie_avg_n <- train_set %>% 
  group_by(movie_age) %>% 
  mutate(movie_avg = mean(rating))
  
movie_avg_n %>% ggplot(aes(movie_age,movie_avg)) + geom_line() + 
  labs(x = "Movie Age", y = "Movie Rating Average")

# Explore genre effect
genre <- edx %>%
  group_by(genres) %>% 
  summarize(mean_genre=mean(rating),n=n()) %>% 
  arrange(-mean_genre)
# Plot avg movie rating vs movie genre
genre %>%
  ggplot(aes(1:nrow(genre),mean_genre)) +
  geom_line() +
  labs(x = "Movie Genre Index", y = "Movie Rating Average")  
  
head (genre)

### Building the Recommendation System

# Function to calaculate RMSE
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# Calculate the average of all movies across all users
mu_hat <- mean(train_set$rating)

# Calculating predicted ratings
predicted_ratings <- test_set %>% 
  mutate(pred = mu_hat) %>%
  .$pred 

# Calculating the model rmse
model_1_rmse <- RMSE(predicted_ratings, test_set$rating)
print(model_1_rmse)

# Calculate movie effect b_i
mu <- mean(train_set$rating) 
movie_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

# Calculating predicted ratings
predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  mutate(pred = mu + b_i) %>%
  .$pred 

# Calculating the model rmse
model_2_rmse <- RMSE(predicted_ratings, test_set$rating)
print(model_2_rmse)

# Calaulate user effect b_u
 user_avgs <- train_set %>% 
   left_join(movie_avgs, by='movieId') %>%
   group_by(userId) %>%
   summarize(b_u = mean(rating - mu - b_i))
 
 # Calculating predicted ratings
 predicted_ratings <- test_set %>% 
   left_join(movie_avgs, by='movieId') %>%
   left_join(user_avgs, by='userId') %>%
   mutate(pred = mu + b_i + b_u) %>%
   .$pred 
 
 # Calculating the model rmse
 model_3_rmse <- RMSE(predicted_ratings, test_set$rating)
 print(model_3_rmse)
 
 # Calaulate genre effect b_g
 genre_avgs <- train_set %>% 
   left_join(movie_avgs, by='movieId') %>%
   left_join(user_avgs, by='userId') %>%
   group_by(genres) %>%
   summarize(b_g = mean(rating - mu - b_i - b_u))
 
 # Calculating predicted ratings
 predicted_ratings <- test_set %>% 
   left_join(movie_avgs, by='movieId') %>%
   left_join(user_avgs, by='userId') %>%
   left_join(genre_avgs, by='genres') %>%
   mutate(pred = mu + b_i + b_u + b_g) %>%
   .$pred 
 
 # Calculating the model rmse
 model_4_rmse <- RMSE(predicted_ratings, test_set$rating)
 print(model_4_rmse)
 
 # Calaulate movie age effect b_a
 age_avgs <- train_set %>% 
   left_join(movie_avgs, by='movieId') %>%
   left_join(user_avgs, by='userId') %>%
   left_join(genre_avgs, by='genres') %>%
   group_by(age_bins) %>%
   summarize(b_a = mean(rating - mu - b_i - b_u - b_g))
 
 # Calculating predicted ratings
 predicted_ratings <- test_set %>% 
   left_join(movie_avgs, by='movieId') %>%
   left_join(user_avgs, by='userId') %>%
   left_join(genre_avgs, by='genres') %>%
   left_join(age_avgs, by='age_bins') %>%
   mutate(pred = mu + b_i + b_u + b_g + b_a) %>%
   .$pred 
 
 # Calculating the model rmse
 model_5_rmse <- RMSE(predicted_ratings, test_set$rating)
 print(model_5_rmse)

 # Spliting training set for testing the Model Validation algorithm and selcting lamdba
 valid_index <- createDataPartition(y = train_set$rating, times = 1,
                                   p = 0.2, list = FALSE)
 valid_train_set <- train_set[-valid_index,]
 valid_test_set <- train_set[valid_index,]
 
 # To make sure I don't include users and movies in the valid_test set that do not
 # appear in the valid_training set, I removed these using the semi_join function
 valid_test_set <- valid_test_set %>% 
   semi_join(valid_train_set, by = "movieId") %>%
   semi_join(valid_train_set, by = "userId")
 
 # Regularization combined movie/user/genre/movie age effect 
 # Choosing lambda
 lambdas <- seq(0, 10, 0.25)
 rmses <- sapply(lambdas, function(l){
   mu <- mean(valid_train_set$rating)
   b_i <- valid_train_set %>% 
     group_by(movieId) %>%
     summarize(b_i = sum(rating - mu)/(n()+l))
   b_u <- valid_train_set %>% 
     left_join(b_i, by="movieId") %>%
     group_by(userId) %>%
     summarize(b_u = sum(rating - b_i - mu)/(n()+l))
   b_g <- valid_train_set %>% 
     left_join(b_i, by="movieId") %>%
     left_join(b_u, by="userId") %>%
     group_by(genres) %>%
     summarize(b_g = sum(rating - b_i - b_u - mu)/(n()+l))
   b_a <- valid_train_set %>%
     left_join(b_i, by="movieId") %>%
     left_join(b_u, by="userId") %>%
     left_join(b_g, by="genres") %>%
     group_by(age_bins) %>%
     summarize(b_a = sum(rating - b_i - b_u - b_g - mu)/(n()+l))
   predicted_ratings <- 
     valid_test_set %>% 
     left_join(b_i, by = "movieId") %>%
     left_join(b_u, by = "userId") %>%
     left_join(b_g, by = "genres") %>%
     left_join(b_a, by = "age_bins") %>%
     mutate(pred = mu + b_i + b_u + b_g + b_a) %>%
     .$pred
   return(RMSE(predicted_ratings, valid_test_set$rating))
 })
 
 qplot(lambdas, rmses)  
 
 lambda <- lambdas[which.min(rmses)]
 
 print(lambda)
 rmses[lambdas == lambda]
 
 # Extract the movie release year from the title for Validation Set
 movie_year_validation <- validation$title %>% str_extract("\\(\\d{4}\\)") %>%
   str_extract("(\\d{4})")
 # Calculate the age of the movie
 validation <-  mutate(validation, movie_age = 2020-as.numeric(movie_year_validation))
 
 # Categorizing movies based on their age
 validation$age_bins <- cut(validation$movie_age, breaks=c(0,20,40,60,80,105), labels=c("1-20","21-40","41-60","61-80","81-105"))
 
 
 # Calculate rmse for the validation set
 
 mu <- mean(validation$rating)
 b_i <- validation %>% 
   group_by(movieId) %>%
   summarize(b_i = sum(rating - mu)/(n()+lambda))
 b_u <- validation %>% 
   left_join(b_i, by="movieId") %>%
   group_by(userId) %>%
   summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))
 b_g <- validation %>% 
   left_join(b_i, by="movieId") %>%
   left_join(b_u, by="userId") %>%
   group_by(genres) %>%
   summarize(b_g = sum(rating - b_i - b_u - mu)/(n()+lambda))
 b_a <- validation %>%
   left_join(b_i, by="movieId") %>%
   left_join(b_u, by="userId") %>%
   left_join(b_g, by="genres") %>%
   group_by(age_bins) %>%
   summarize(b_a = sum(rating - b_i - b_u - b_g - mu)/(n()+lambda))
 predicted_ratings <- 
   validation %>% 
   left_join(b_i, by = "movieId") %>%
   left_join(b_u, by = "userId") %>%
   left_join(b_g, by = "genres") %>%
   left_join(b_a, by = "age_bins") %>%
   mutate(pred = mu + b_i + b_u + b_g + b_a) %>%
   .$pred
 print('The RMSE of precidtions for the Validation set is:')
 RMSE(predicted_ratings, validation$rating)