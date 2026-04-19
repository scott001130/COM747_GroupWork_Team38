#set to path where german.data is saved & load as dataframe
setwd("C:\\Users\\614566589\\OneDrive - BT Plc\\Documents")
german_df <- read.table("german.data", header = FALSE, sep = "")

#retain raw file for validation (will build this next week)
german_raw <- german_df

#label columns in dataframe (interpreted from UCI)
colnames(german_df) <- c(
  "account_status",
  "loan_duration_mths",
  "credit_history",
  "loan_purpose",
  "credit_amount",
  "savings",
  "employment_duration",
  "installment_rate",
  "marital_status",
  "other_debtors",
  "residence_since",
  "property",
  "age_yrs",
  "other_installment_plans",
  "housing",
  "existing_credits",
  "occupation",
  "people_liable",
  "telephone",
  "foreign_worker",
  "target")

#4 functions below
#first 2 map all A-codes to (un)ordered factors
#last 2 map integers that require factorisation to (un)ordered factors

a_code_to_factor <- function(x, mapping) {
  x <- as.character(x)
  mapped_label <- unname(mapping[x])
  factor(mapped_label, levels = unique(unname(mapping)))}

a_code_to_ordered_factor <- function(x, mapping) {
  x <- as.character(x)
  mapped_label <- unname(mapping[x])
  factor(mapped_label, levels = unique(unname(mapping)), ordered = TRUE)}

int_to_factor <- function(x, mapping) {
  x <- as.integer(as.character(x))
  mapped_label <- unname(mapping[as.character(x)])
  factor(mapped_label, levels = unique(unname(mapping)))}

int_to_ordered_factor <- function(x, mapping) {
  x <- as.integer(as.character(x))
  mapped_label <- unname(mapping[as.character(x)])
  factor(mapped_label, levels = unique(unname(mapping)), ordered = TRUE)}

#these features don't need factorised as they're measured quants & not categorical
german_df$loan_duration_mths <- as.integer(as.character(german_df$loan_duration_mths))
german_df$credit_amount <- as.integer(as.character(german_df$credit_amount))
german_df$age_yrs <- as.integer(as.character(german_df$age_yrs))

#builds mapping tables for all featured that need factorisation
#order of each map dictates level order of factors (levels=unique in functions)
account_status_map <- c(
  "A11" = "No checking account",
  "A12" = "< 0 DM",
  "A13" = "0 to < 200 DM",
  "A14" = ">= 200 DM / salary assignments for at least 1 year")

credit_history_map <- c(
  "A30" = "Delay in paying off in the past",
  "A31" = "Critical account / other credits elsewhere",
  "A32" = "No credits taken / all credits paid back duly",
  "A33" = "Existing credits paid back duly till now",
  "A34" = "All credits at this bank paid back duly")

loan_purpose_map <- c(
  "A40"  = "Other",
  "A41"  = "Car (new)",
  "A42"  = "Car (used)",
  "A43"  = "Furniture / equipment",
  "A44"  = "Radio / television",
  "A45"  = "Domestic appliances",
  "A46"  = "Repairs",
  "A47"  = "Education",
  "A48"  = "Vacation",
  "A49"  = "Retraining",
  "A410" = "Business")

savings_map <- c(
  "A61" = "Unknown / no savings account",
  "A62" = "< 100 DM",
  "A63" = "100 to < 500 DM",
  "A64" = "500 to < 1000 DM",
  "A65" = ">= 1000 DM")

employment_duration_map <- c(
  "A71" = "Unemployed",
  "A72" = "< 1 year",
  "A73" = "1 to < 4 years",
  "A74" = "4 to < 7 years",
  "A75" = ">= 7 years")

marital_status_map <- c(
  "A91" = "Male divorced / separated",
  "A92" = "Female non-single or male single",
  "A93" = "Female non-single or male single",
  "A94" = "Male married / widowed",
  "A95" = "Female single")

other_debtors_map <- c(
  "A101" = "None",
  "A102" = "Co-applicant",
  "A103" = "Guarantor")

property_map <- c(
  "A121" = "Unknown / no property",
  "A122" = "Car or other property",
  "A123" = "Building society savings agreement / life insurance",
  "A124" = "Real estate")

other_installment_plans_map <- c(
  "A141" = "Bank",
  "A142" = "Stores",
  "A143" = "None")

housing_map <- c(
  "A151" = "For free",
  "A152" = "Rent",
  "A153" = "Own")

occupation_map <- c(
  "A171" = "Unemployed / unskilled non-resident",
  "A172" = "Unskilled resident",
  "A173" = "Skilled employee / official",
  "A174" = "Manager / self-employed / highly qualified employee")

telephone_map <- c(
  "A191" = "No",
  "A192" = "Yes (registered under customer's name)")

foreign_worker_map <- c(
  "A201" = "No",
  "A202" = "Yes")

installment_rate_map <- c(
  "1" = ">= 35% of disposable income",
  "2" = "25% to < 35%",
  "3" = "20% to < 25%",
  "4" = "< 20%")

residence_since_map <- c(
  "1" = "< 1 year",
  "2" = "1 to < 4 years",
  "3" = "4 to < 7 years",
  "4" = ">= 7 years")

existing_credits_map <- c(
  "1" = "1",
  "2" = "2 to 3",
  "3" = "4 to 5",
  "4" = ">= 6")

people_liable_map <- c(
  "1" = "0 to 2",
  "2" = "3 or more")

target_map <- c(
  "1" = "Good",
  "2" = "Bad")

#functions apply correct level of factorisation per feature
german_df$account_status <- a_code_to_ordered_factor(german_df$account_status, account_status_map)
german_df$credit_history <- a_code_to_ordered_factor(german_df$credit_history, credit_history_map)
german_df$loan_purpose <- a_code_to_factor(german_df$loan_purpose, loan_purpose_map)
german_df$savings <- a_code_to_ordered_factor(german_df$savings, savings_map)
german_df$employment_duration <- a_code_to_ordered_factor(german_df$employment_duration, employment_duration_map)
german_df$marital_status <- a_code_to_factor(german_df$marital_status, marital_status_map)
german_df$other_debtors <- a_code_to_factor(german_df$other_debtors, other_debtors_map)
german_df$property <- a_code_to_ordered_factor(german_df$property, property_map)
german_df$other_installment_plans <- a_code_to_factor(german_df$other_installment_plans, other_installment_plans_map)
german_df$housing <- a_code_to_factor(german_df$housing, housing_map)
german_df$occupation <- a_code_to_ordered_factor(german_df$occupation, occupation_map)
german_df$telephone <- a_code_to_factor(german_df$telephone, telephone_map)
german_df$foreign_worker <- a_code_to_factor(german_df$foreign_worker, foreign_worker_map)

german_df$installment_rate <- int_to_ordered_factor(german_df$installment_rate, installment_rate_map)
german_df$residence_since <- int_to_ordered_factor(german_df$residence_since, residence_since_map)
german_df$existing_credits <- int_to_ordered_factor(german_df$existing_credits, existing_credits_map)
german_df$people_liable <- int_to_ordered_factor(german_df$people_liable, people_liable_map)
german_df$target <- int_to_factor(german_df$target, target_map)

#turns target into a binary outcome, 1=bad
german_df$target_bad <- ifelse(german_df$target == "Bad", 1L, 0L)

#use the csv to look at the data but load the rds file into R, not the csv
#loading the csv will wipe the factorisation and ordering
write.csv(german_df, "german_credit_clean.csv", row.names = FALSE)
saveRDS(german_df, "german_credit_clean.rds")

#use below to load clean df into R
#setwd("C:\\Users\\614566589\\OneDrive - BT Plc\\Documents")
#gc_df <- readRDS("german_credit_clean.rds")
