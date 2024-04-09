### Clear memory
rm(list=ls())

### Set working directory
setwd("/Users/noelfranklin/Downloads")
data = read.csv("query_result.csv")

t1 <- data.frame(data)
summary(t1)

t1$riskfactor_sc= (t1$velocity/100)*15.57

model <- lm(t1$riskfactor ~  t1$events + t1$velocity + t1$totmiles , data=data)
summary(model)

# Get the predicted values
data$predicted <- predict(model, data)

# Plot the actual and predicted values
ggplot(data, aes(x = t1$riskfactor, y = predicted)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE, color = "green") +
  xlab("Actual Values") +
  ylab("Predicted Values") +
  ggtitle("Multiple Linear Regression")

