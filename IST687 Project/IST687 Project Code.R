#Load the required packages
require(pacman)
p_load(readxl, dplyr, ggplot2, tidyr, lubridate, gridExtra)

#Load the dataset into R 
dat <- read_excel('IST687 Data.xlsx')

#Check the structure of the data, head, and see that there are no NAs.
str(dat)
head(dat)
summary(dat)

#Column names need to be changed to better understand the data. 
colnames(dat)
names <- c("person_id", "credit_score", "loan_application_date", "requested_loan_term", "requested_loan_amount",
           "preapproved_loan_amount", "preapproved_loan_term",  "person_birthdate", "person_marital_status", "person_gender", 
           "person_degree_type_desc", "person_employment_type_desc", "industry", "estimated_net_income", "actual_net_income", 
           "monthly_debt_capacity", "decision_status", "loan_disbursement_date", "first_repayment_date", "loan_amount", 
           "approved_loan_rate", "approved_nominal_rate", "approved_interest_amount")
colnames(dat) <- names

#Create month variable to better bin the data for month-to-month analysis.
#We'll also calculate the age and bin them in groups of ten years.
dat <- dat %>% mutate(age = as.numeric((Sys.Date() - as.Date(person_birthdate))/365),
                      age_bins = as.character(cut(age, breaks = c(21, 31, 41, 51, 61, 71, 81, 91), 
                                     labels = c('21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90'))), 
                      loan_application_code_month = ifelse(month(dat$loan_application_date) < 10, 
                                                           paste0(year(dat$loan_application_date), 0, month(dat$loan_application_date)),
                                                           paste0(year(dat$loan_application_date), month(dat$loan_application_date))),
                      loan_disbursement_code_month = ifelse(month(dat$loan_disbursement_date) < 10, 
                                                            paste0(year(dat$loan_disbursement_date), 0, month(dat$loan_disbursement_date)),
                                                            paste0(year(dat$loan_disbursement_date), month(dat$loan_disbursement_date))))

#Demographic distributions

#Applications per month by gender
ggplot(dat, aes(loan_application_code_month, fill = as.factor(person_gender))) + 
  geom_bar(position = 'stack') + 
  labs(x = 'YearMonth', y = 'Number of Applications', fill = 'Gender') + 
  ggtitle('Applications per Month', subtitle = 'Aug 2017 - Jan 2018') +
  theme(plot.title = element_text(hjust = 0.5), plot.subtitle = element_text(hjust = 0.5))

#Application per month by gender - percentage 
ggplot(dat, aes(loan_application_code_month, fill = as.factor(person_gender))) + 
  geom_bar(position = 'fill') + 
  labs(x = 'YearMonth', y = 'Percentage of Applications', fill = 'Gender') + 
  ggtitle('Applications per Month', subtitle = 'Aug 2017 - Jan 2018') +
  theme(plot.title = element_text(hjust = 0.5), plot.subtitle = element_text(hjust = 0.5))

#Applicant age distributions by gender
ggplot(dat, aes(age, fill = as.factor(person_gender))) + 
  geom_density(alpha = 0.3) + 
  facet_wrap(~ person_gender, ncol = 1) + 
  labs(x = 'Customer Age', y = 'Density', fill = 'Gender') + 
  ggtitle('Age Distribution by Gender') + 
  theme(legend.position = 'bottom', plot.title = element_text(hjust = 0.5), plot.subtitle = element_text(hjust = 0.5))

#
ggplot(dat, aes(reorder(person_degree_type_desc, table(person_degree_type_desc)[person_degree_type_desc]))) + 
  geom_bar() +
  coord_flip() +
  labs(y = 'Customer Count', x = 'Degree')

ggplot(dat, aes(reorder(industry, table(industry)[industry]))) + 
  geom_bar() + 
  coord_flip() +
  labs(x = 'Industry', y = 'Count')

ggplot(dat, aes(reorder(person_marital_status, table(person_marital_status)[person_marital_status]))) + 
  geom_bar() +
  coord_flip() + 
  labs(x= 'Marital Status', y = 'Count')

ggplot(dat, aes(credit_score)) + geom_histogram()
