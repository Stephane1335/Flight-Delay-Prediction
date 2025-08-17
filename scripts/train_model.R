###############################################################
# Flight Delay Prediction - Model Training Script
# Author: Jean-Stephane YAPO
# Description:
#   This script loads historical flight data, performs feature
#   engineering, trains a regression model (XGBoost) to predict
#   arrival delay in minutes (positive = delay, negative = early),
#   evaluates the model, and saves it for later use.
###############################################################



# 1) Load packages
suppressPackageStartupMessages({
  library(tidyverse)
  library(lubridate)
  library(tidymodels)
  library(vip)   # Variable importance visualization
})

set.seed(42)

# 2) Load dataset
getwd()
data_path <- "GitHub/Flight-Delay-Prediction/data/flight_delays.csv"
raw <- read_csv(data_path, show_col_types = FALSE)

# 3) Clean and preprocess
parse_dt_safe <- function(x) {
  out <- suppressWarnings(lubridate::ymd_hms(x, tz = "UTC"))
  if (all(is.na(out))) out <- suppressWarnings(lubridate::ymd_hm(x, tz = "UTC"))
  if (all(is.na(out))) out <- suppressWarnings(lubridate::ymd(x, tz = "UTC"))
  out
}

df <- raw %>%
  filter(!Cancelled, !Diverted) %>%
  mutate(
    ScheduledDeparture_dt = parse_dt_safe(ScheduledDeparture),
    ScheduledArrival_dt   = parse_dt_safe(ScheduledArrival),
    ActualArrival_dt      = parse_dt_safe(ActualArrival),
    arrival_delta_min = as.numeric(difftime(ActualArrival_dt, ScheduledArrival_dt, units = "mins")),
    dep_hour   = hour(ScheduledDeparture_dt),
    dep_min    = minute(ScheduledDeparture_dt),
    dep_wday   = wday(ScheduledDeparture_dt, week_start = 1),
    dep_month  = month(ScheduledDeparture_dt),
    sched_block_min = as.numeric(difftime(ScheduledArrival_dt, ScheduledDeparture_dt, units = "mins"))
  ) %>%
  drop_na(arrival_delta_min)

# 4) Select features
features <- c("Airline", "Origin", "Destination", "Distance",
              "AircraftType", "dep_hour", "dep_min", "dep_wday",
              "dep_month", "sched_block_min")

df_model <- df %>%
  select(all_of(c(features, "arrival_delta_min"))) %>%
  mutate(across(c(Airline, Origin, Destination, AircraftType), as.factor))

# 5) Train/Test split
set.seed(42)
split <- initial_split(df_model, prop = 0.8, strata = arrival_delta_min)
train <- training(split)
test  <- testing(split)

# 6) Preprocessing pipeline
rec <- recipe(arrival_delta_min ~ ., data = train) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  step_other(all_nominal_predictors(), threshold = 0.01, other = "OTHER") %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  step_zv(all_predictors())

# 7) Model specification (XGBoost)
xgb_spec <- boost_tree(
  trees = tune(),
  tree_depth = tune(),
  learn_rate = tune(),
  loss_reduction = tune(),
  min_n = tune(),
  mtry = tune()
) %>%
  set_engine("xgboost") %>%
  set_mode("regression")

wf <- workflow() %>%
  add_model(xgb_spec) %>%
  add_recipe(rec)

# 8) Cross-validation + hyperparameter tuning
set.seed(42)
cv_folds <- vfold_cv(train, v = 5, strata = arrival_delta_min)

prep_rec <- prep(rec)
x_train <- bake(prep_rec, new_data = NULL) %>% select(-arrival_delta_min)

grid <- grid_latin_hypercube(
  trees(),
  tree_depth(),
  learn_rate(range = c(0.01, 0.3)),
  loss_reduction(),
  min_n(),
  finalize(mtry(), x_train),
  size = 20
)

tuned <- tune_grid(
  wf,
  resamples = cv_folds,
  grid = grid,
  metrics = metric_set(rmse, mae, rsq),
  control = control_grid(save_pred = TRUE)
)

best <- select_best(tuned, metric = "rmse")

# 9) Train final model
final_wf <- finalize_workflow(wf, best)
final_fit <- fit(final_wf, data = train)

# 10) Evaluate on test set
test_preds <- predict(final_fit, new_data = test) %>%
  bind_cols(test %>% select(arrival_delta_min))

metrics_test <- metrics(test_preds, truth = arrival_delta_min, estimate = .pred)
print(metrics_test)

# 11) Save the trained model
saveRDS(final_fit, "../models/arrival_delay_xgb_model.rds")
cat("✅ Model saved to models/arrival_delay_xgb_model.rds\n")

# 12) Optional: plot variable importance
try({
  final_fit %>%
    extract_fit_parsnip() %>%
    vip(num_features = 15)
}, silent = TRUE)
