#############################################################
# Create edx set, validation set, and submission file
#############################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl, timeout = 60)

ratings <- read.table(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                      col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Final hold-out test set will be 10% of MovieLens data

set.seed(1)
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,] 

# Make sure userId and movieId in validation set are also in edx set validation set 

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

###################################################################################
#################################################

#Save the edx and validation files
save(edx, file="edx.RData")
save(validation, file = "validation.RData")

#Loading library 
library(tidyverse)
library(ggplot2)
library(dplyr)
library(markdown)
library(knitr)
library(caret)

### Exploratory analysis

#First, we explore the data 
str(edx)
str(validation)

# Verifying the data for any NA values 
anyNA(edx)
# Obtaining the count of movies and users in the dataset: 
edx %>% 
  summarize(n_users = n_distinct(userId),
            n_movies = n_distinct(movieId))

# Let's identify the movies with the highest number of ratings:
edx %>% group_by(rating) %>% summarize(count = n()) %>% top_n(5) %>%
  arrange(desc(count))  

#Determining the number of ratings received by each movie: 
edx %>% count(movieId) %>% ggplot(aes(n))+
  geom_histogram(color = "black" , fill= "light blue")+
  scale_x_log10()+
  ggtitle("Number of ratings per movie")+
  theme_gray()
#Visualising the distribution of ratings for each user:
edx %>% count(userId) %>% ggplot(aes(n))+
  geom_histogram(color = "red" , fill= "light blue")+
  ggtitle(" Number of ratings per user")+
  scale_x_log10()+
  theme_gray()

#Calculating the number of ratings for each movie genre:
edx %>% separate_rows(genres, sep = "\\|") %>%
  group_by(genres) %>%
  summarize(count = n()) %>%
  arrange(desc(count)) %>% ggplot(aes(genres,count)) + 
  geom_bar(aes(fill =genres),stat = "identity")+ 
  labs(title = "Number of ratings for each genre")+
  theme(axis.text.x  = element_text(angle= 90, vjust = 50 ))+
  theme_light()
 
#Partition the data 
set.seed(1)
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.2, list = FALSE)
train_set <- edx[-test_index,]
test_set <- edx[test_index,]
#To ensure that we don't include users and movies in the test set that are not present in the training set, 
#we remove these entries using the semi_join function:
test_set <- test_set %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")
  
#RMSE calculation Function 

RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2, na.rm = TRUE))
}

# The first model:
Mu_1 <- mean(train_set$rating)
Mu_1

naive_rmse <- RMSE( Mu_1,test_set$rating)
naive_rmse

#This code generates a table to store the RMSE results obtained from each method, 
#facilitating comparison between different approaches.

rmse_results <- data_frame(method = "Just the average", RMSE = naive_rmse)

#The second model 
#As observed during the exploratory analysis certain movies received more ratings compared to others. 
#We can enhance our previous model by incorporating the term b_i to denote the average rating for movie i. 
#We can once again utilize least squares to estimate the movie effect
#$Y~u,i~ = ?? + b~i~ + ??~u,i~$ 
#Due to the large number of parameters $b~i$ corresponding to each movie, 
#employing the lm() function directly can lead to significant computational slowdown. 
#Therefore, we opt for a more efficient approach by computing it using the average, as follows:

Mu_2 <- mean(train_set$rating) 
movie_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - Mu_2))
movie_avgs

#We observe variability in the estimate, as depicted in the plot presented here:
movie_avgs %>% qplot(b_i, geom ="histogram", bins = 10, data = ., color = I("black"))

#Let's examine how the prediction accuracy improves after modifying the equation $Y~u,i~ = ?? + b~i$:
predicted_ratings <- Mu_2 + test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)

model_1_rmse <- RMSE(predicted_ratings, test_set$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie Effect Model",  
                                     RMSE = model_1_rmse))
rmse_results 
                   
# The third model 
# Comparing users who have rated more than 100 movies:
train_set %>% 
  group_by(userId) %>% 
  summarize(b_u = mean(rating)) %>% 
  filter(n()>=100) %>%
  ggplot(aes(b_u)) + 
  geom_histogram(bins = 30, color = "black")

#There is significant variability observed across user ratings as well. 
#This suggests that further improvement to our model may be nessesary such as: $Y~u,i~ = ?? + b~i~ + ??~u,i~$ 
#We could fit this model by using use the lm() function but as mentioned earlier, 
#it would be very slow due to large dataset lm(rating \~ as.factor(movieId) + as.factor(userId)) 

#Here is the code:

user_avgs <- train_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - Mu_2 - b_i))
user_avgs

#Now, let's examine how the RMSE has improved this time:
predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = Mu_2 + b_i + b_u) %>%
  .$pred

model_3_rmse <- RMSE(predicted_ratings, test_set$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie + User Effects Model",  
                                     RMSE = model_3_rmse))
rmse_results

##The RMSE of the validation set is:

valid_pred_rating <- validation %>%
  left_join(movie_avgs, by = "movieId" ) %>% 
  left_join(user_avgs , by = "userId") %>%
  mutate(pred = Mu_2 + b_i + b_u ) %>%
  pull(pred)

model_3_valid <- RMSE(validation$rating, valid_pred_rating)
model_3_valid
rmse_results <- bind_rows( rmse_results, 
                           data_frame(Method = "Validation Results" , RMSE = model_3_valid))
rmse_results

