
data <- read.csv("data.csv", stringsAsFactors = TRUE)
print(head(data))
print(summary(data))

# What's the average price of BMW cars in the dataset?

print(mean(subset(data, data$Make == 'BMW')$MSRP))

# Number of missing values in "Engine HP" after 2015

print(sum(is.na(subset(data, data$Year >= 2015)$Engine.HP)))

# Does the mean change after filling missing values?

avg <- mean(subset(data, data$Year >= 2015)$Engine.HP, na.rm = TRUE)
print(avg)
data[is.na(data$Engine.HP),]$Engine.HP <- avg
print(mean(subset(data, data$Year >= 2015)$Engine.HP, na.rm = TRUE))

rr_cars <- data[data$Make == 'Rolls-Royce', c("Engine.HP", "Engine.Cylinders", "highway.MPG")]
rr_cars2 <- unique(rr_cars)