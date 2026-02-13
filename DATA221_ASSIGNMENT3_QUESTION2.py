# Harris Khan
# February 10, 2026
# DATA221, Assignment 3, Question 2

# Import matplotlib.pyplot under the alias plt
import matplotlib.pyplot as plt
# Import pandas package under the alias pd
import pandas as pd

# Use the pandas package to create a dataframe out of the crime stats data
crime_stats_data_frame = pd.read_csv("crime1.csv")

# Store the column "ViolentCrimesPerPop" in this variable
violent_crimes_per_population_stats = crime_stats_data_frame["ViolentCrimesPerPop"]

# Create a frequency histogram measuring the ViolentCrimesPerPop variable
plt.hist(x = violent_crimes_per_population_stats, color = "orange") # Creates the histogram
plt.title("Frequency Histogram of Violent Crimes per Population") # Creates the title
plt.xlabel("Violent Crimes per Population") # Creates the x-axis label
plt.ylabel("Frequency") # Creates the y-axis label
plt.show() # Reveals the histogram

# Create a boxplot measuring the ViolentCrimesPerPop variable
plt.boxplot(violent_crimes_per_population_stats) # Creates the boxplot
plt.title("Boxplot of Violent Crimes per Population") # Creates the title
plt.xlabel("Violent Crimes per Population") # Creates the x-axis label
plt.ylabel("Percentage of Violent Crimes") # Creates the y-axis label
plt.show() # Reveals the boxplot

# Question 1:
# The histogram shows that the distribution of violent crimes per population are slightly skewed to the right.
# The bars for each bin are at their highest near the left, indicating a higher concentration of percentages on
# the lower side. On the other hand, the bars for each bin are at their lowest near the right 'tail end' of the
# distribution. Although there is some skew towards the right, there are still a good portion of values located
# on the right side of the distribution. Therefore, based on these data, one can infer that there is a loose
# skewedness towards the right of the distribution.

# Question 2:
# The boxplot tells us that the median is approximately equal to 0.39 or 0.40. This tells us that the distribution
# of values are more commonly found at the lower end of percentages rather than the higher percentages. The
# median line being in the middle of the Interquartile Range box also tells us that the median is closer to 0.50,
# or 50%. This is because the IQR measures the middle 50% of the distribution (75th percentile - 25th percentile).
# However, although the median line is close to the middle of the IQR box, there are also longer whiskers at the
# higher end of the distribution, indicating that there is still some skewedness to the right.

# Question 3:
# The boxplot does not suggest any presence of outliers. If we look at the whiskers of the boxplot, the
# lower whiskers extend downwards close to 0. The upper whiskers extend uppwards close to 1. Since the boundaries this
# variable tracks is limited to 0% and 100% (0.00 to 1.00), then there are few values, if any, that could lie outside
# the whisker range and still be within the bounds of the numerical variable. Box plots also explicitly tell you
# if there are any outliers. Any dots that extend outside of each end of whiskers are outlier points. Since there are
# no such dots in our box plot, there are not outliers in this distribution.