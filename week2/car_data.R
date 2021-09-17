library(tidyverse)
library(stats)
library(car)
library(GGally)

if (!file.exists('../data/car_data.csv')) {
    download.file('https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-02-car-price/data.csv', 
                  destfile = '../data/car_data.csv', method = 'curl')    
}
stopifnot(file.exists('../data/car_data.csv'))

raw_data <- read_csv('../data/car_data.csv')
colnames(raw_data) <- str_replace_all(tolower(colnames(raw_data)), "\\s", "_")

car_data <- mutate(raw_data, 
                   make = as.factor(str_replace_all(tolower(make), "\\s", "_")),
                   model = as.factor(str_replace_all(tolower(model), "\\s", "_")),
                   engine_fuel_type = as.factor(str_replace_all(tolower(engine_fuel_type), "\\s", "_")),
                   engine_cylinders = as.factor(str_replace_all(tolower(engine_cylinders), "\\s", "_")),
                   transmission_type = as.factor(str_replace_all(tolower(transmission_type), "\\s", "_")),
                   driven_wheels = as.factor(str_replace_all(tolower(driven_wheels), "\\s", "_")),
                   number_of_doors = as.factor(number_of_doors),
                   market_category = as.factor(str_replace_all(tolower(market_category), "\\s", "_")),
                   vehicle_size = as.factor(str_replace_all(tolower(vehicle_size), "\\s", "_")),
                   vehicle_style = as.factor(str_replace_all(tolower(vehicle_style), "\\s", "_")),
                   age = max(year) - year) %>%
    select(-c('year', 'model', 'market_category', 'engine_cylinders')) %>%
    relocate(msrp, .after = last_col())
print(str(car_data))
print(summary(car_data))

# remove Makes with less than 10 records
car_makes <- count(car_data, make) %>% filter(n >= 10)
car_engine_fuel_types <- count(car_data, engine_fuel_type) %>% filter(n >= 10)
#car_engine_cylinders <- count(car_data, engine_cylinders) %>% filter(n >= 10)
car_data <- filter(car_data, 
                   make %in% car_makes$make, 
#                  engine_cylinders %in% car_engine_cylinders$engine_cylinders,
                   engine_fuel_type %in% car_engine_fuel_types$engine_fuel_type)
print(summary(car_data))

ggcorr(car_data[,sapply(car_data, function(x) !is.factor(x))], label = TRUE, label_size = 2.9, hjust = 1, layout.exp = 2)

 
init_lm <- lm(msrp ~ ., data = car_data)
print(summary(init_lm))
print(alias(init_lm))

final_lm <- step(init_lm, direction = "backward")
print(summary(final_lm))
