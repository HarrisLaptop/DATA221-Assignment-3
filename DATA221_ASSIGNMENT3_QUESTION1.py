# Harris Khan
# February 10, 2026
# DATA221, Assignment 3, Question 1

# Import the pandas package under the alias pd
import pandas as pd
# Import the numpy package under the alias np
import numpy as np

crime_stats_data_frame = pd.read_csv("crime1.csv")

mean_of_violent_crimes_per_population = np.mean(crime_stats_data_frame["ViolentCrimesPerPop"])
print(mean_of_violent_crimes_per_population)

median_of_violent_crimes_per_population = np.median(crime_stats_data_frame["ViolentCrimesPerPop"])
print(median_of_violent_crimes_per_population)

standard_deviation_of_violent_crimes_per_population = np.std(crime_stats_data_frame["ViolentCrimesPerPop"])
print(standard_deviation_of_violent_crimes_per_population)

minimum_value_of_violent_crimes_per_population = np.min(crime_stats_data_frame["ViolentCrimesPerPop"])
print(minimum_value_of_violent_crimes_per_population)

maximum_value_of_violent_crimes_per_population = np.max(crime_stats_data_frame["ViolentCrimesPerPop"])
print(maximum_value_of_violent_crimes_per_population)

# Question 1:
# From my assessment, the distribution seems to be slightly skewed to the right. If we take a look
# at the mean and median, we see that the mean holds a value of approximately 0.44 whereas the median
# is equal to 0.39. Since the mean is more affected by outliers than median (which is discussed more
# in the next question), then we can say that because the mean is a bit greater in value than the
# median, the distribution does not closely resemble a normal model. (However, since the mean and
# median are not more than 1 standard deviation away from each other either, the distribution is
# probably not that heavily skewed either).

# Question 2:
# The mean is affected by extreme values more than the median. This is because the mean must take
# an average of data values. Therefore, if an extreme value is added to the calculations of mean,
# it may heavily influence the value of the mean. On the other hand, the median is not as affected
# by extreme values since it only checks for what the middle value of it's values is. Therefore, if
# we have a large collection of numbers that are close to some value, and one extreme value, the
# median will still hold the value of the numbers that are close to the middle.