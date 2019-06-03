install.packages('Lahman') #Baseball database
install.packages('dplyr')  #data frame manipulation
install.packages('ggplot2')

library(Lahman)
library(dplyr)
library(ggplot2)

#Loads batting table from Lahman package, filters yearID to 2016, and creates a new column called BatAvg
bat <- Batting %>% 
  filter(yearID == 2016) %>% 
  mutate(BatAvg = round(H/AB, 3))

#Next, will filter down to two teams: the New York Yankees and the Boston Red Sox. We'll also keep batters who had more than 100 ABs
bat2 <- bat %>% 
  filter(teamID %in% c('NYA', 'BOS') & AB >= 100)

#Let's check the data
str(bat2)
summary(bat2)
head(bat2)

#Now, we'll compare the Batting Average (BatAvg) spread for both teams
ggplot(bat2, aes(x = BatAvg, fill = as.factor(teamID))) + 
  geom_histogram(alpha = 0.3) +  #alpha determines tranparency. It goes from 0 to 1. 
  labs(x = 'Batting Average', y = 'Count', fill = 'TeamID') + 
  ggtitle('NYA vs BOS Batting Average Spread', subtitle = '2016 Season')

#This visual is not very appealing, so we'll create another one based on geom_density 
ggplot(bat2, aes(x = BatAvg, fill = as.factor(teamID))) + geom_density(alpha = 0.3) + 
  labs(x = 'Batting Average', y = 'Count', fill = 'TeamID') + 
  ggtitle('NYA vs BOS Batting Average Spread', subtitle = '2016 Season')

#This one is more visually appealing and makes more sense! 
#Boston has a more "even" distribution, while New York has a right-skewed distribution.