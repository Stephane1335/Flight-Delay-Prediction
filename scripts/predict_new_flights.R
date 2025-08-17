###############################################################
# Flight Delay Prediction - Inference Script
# Description:
#   Loads the trained model and predicts arrival delay (in minutes)
#   for new upcoming flights (no actual arrival time yet).
###############################################################

suppressPackageStartupMessages({
  library(tidyverse)
  library(lubridate)
})

# Utility: safe datetime parser
parse_dt_safe <- function(x) {
  out <- suppressWarnings(lubridate::ymd_hms(x, tz = "UTC"))
  if (all(is.na(out))) out <- suppressWarnings(lubridate::ymd_hm(x, tz = "UTC"))
  if (all(is.na(out))) out <- suppressWarnings(lubridate::ymd(x, tz = "UTC"))
  out
}

# Load trained model
model_path <- "../models/arrival_delay_xgb_model.rds"
if (!file.exists(model_path)) stop("Trained model not found. Run train_model.R first.")
model <- readRDS(model_path)

# Load new upcoming flights
new_data <- read_csv("../data/upcoming_flights.csv", show_col_types = FALSE)

# Feature engineering (same as training)
new_data_prep <- new_data %>%
  mutate(
    ScheduledDeparture_dt = parse_dt_safe(ScheduledDeparture),
    ScheduledArrival_dt   = parse_dt_safe(ScheduledArrival),
    dep_hour   = hour(ScheduledDeparture_dt),
    dep_min    = minute(ScheduledDeparture_dt),
    dep_wday   = wday(ScheduledDeparture_dt, week_start = 1),
    dep_month  = month(ScheduledDeparture_dt),
    sched_block_min = as.numeric(difftime(ScheduledArrival_dt, ScheduledDeparture_dt, units = "mins"))
  ) %>%
  select(Airline, Origin, Destination, Distance, AircraftType,
         dep_hour, dep_min, dep_wday, dep_month, sched_block_min) %>%
  mutate(across(c(Airline, Origin, Destination, AircraftType), as.factor))

# Predict
preds <- predict(model, new_data = new_data_prep)
result <- new_data %>%
  bind_cols(predicted_delay_min = round(preds$.pred, 0))

print(result)
write_csv(result, "../data/predictions.csv")
cat("✅ Predictions saved to data/predictions.csv\n")
