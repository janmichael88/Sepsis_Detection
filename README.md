# Sepsis_Detection
### Predictive Septic Shock for ICU patients

This was a project I did for my Insight September 2020 session. The purpose of this project is to create an algorithm that will predict whether or not a patient will go into septic shock prematurely! The idea is to identify as early as possible if an ICU patient will go into septic shock.

### Data
The date comes from the MIMIC IV repository
* https://mimic-iv.mit.edu/

For this project approximately 10 million rows were obtained from MIMIC IV. This was narrowed down to 2 million total entries. With approximately 40,000 patients. Root files in this directoy include data aggregation and EDA

### Folder Files
* v1 lstm with 3 units for a time i window of 100 - model just predicts the moajority class\
* v2 same lstm, but with the correct labels, still the same as v1
* v3 experimenting with xgboost to identify feature importance values 25Sep20
