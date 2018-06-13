# Import libraries
library("tidyverse")
library("xgboost")
library("randomForest")
library("here")

# Randn
set.seed(1337)

# Import database
# Database from the challenge
df <-
  read_csv(
    file = "~/Downloads/trainingData-release.csv",
    na = c("", "NA", "NotDone", "ND")
  )

# Database from Moodle (train)
AML_train_data <-
  read_csv(
    file = here("data", "AMLtraindata.csv"),
    na = c("", "NA", "NotDone")
  ) %>%
  mutate(
    SEX = ifelse(SEX == "M", 1, ifelse(SEX == "F", 0, NA)),
    PRIOR.MAL = ifelse(PRIOR.MAL == "YES", 1, ifelse(PRIOR.MAL == "NO", 0, NA)),
    PRIOR.CHEMO = ifelse(PRIOR.CHEMO == "YES", 1, ifelse(PRIOR.CHEMO == "NO", 0, NA)),
    PRIOR.XRT = ifelse(PRIOR.XRT == "YES", 1, ifelse(PRIOR.XRT == "NO", 0, NA)),
    Infection = ifelse(Infection == "Yes", 1, ifelse(Infection == "No", 0, NA)),
    cyto.cat = as.factor(cyto.cat),
    ITD = ifelse(ITD == "POS", 1, ifelse(ITD == "NEG", 0, NA)),
    D835 = ifelse(D835 == "POS", 1, ifelse(D835 == "NEG", 0, NA)),
    Ras.Stat = ifelse(Ras.Stat == "POS", 1, ifelse(Ras.Stat == "NEG", 0, NA)),
    Chemo.Simplest = as.factor(Chemo.Simplest),
    resp.simple = as.factor(ifelse(resp.simple == "CR", 1, ifelse(resp.simple == "RESISTANT", 0, NA)))
  )

train_clinical <-
  AML_train_data %>%
  select(SEX:CD19)

train_proteomic <-
  AML_train_data %>%
  select(ACTB:ZNF346)

# Database from Moodle (test)
AML_test_data <- read_csv(
 file = "~/Downloads/AMLtestdata.csv",
 na = c("", "NA", "NotDone")
)

# Impute the dataframe
train_clinical_imputed <- rfImpute(
  x = resp.simple ~ .,
  data = train_clinical
)

# write out data

readr::write_rds(train_clinical, here("data", "aml_train_clinical.rds"))
readr::write_rds(train_clinical_imputed, here("data", "aml_train_clinical_imputed.rds"))
readr::write_rds(train_proteomic, here("data", "aml_train_proteomic.rds"))

# Are the patient id's in the test set in the original dataset?
AML_test_data$X.Patient_id %in% pull(df, 1)
