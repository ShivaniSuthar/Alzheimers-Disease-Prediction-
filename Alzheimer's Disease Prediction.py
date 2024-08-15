#!/usr/bin/env python
# coding: utf-8

# # UC San Diego: Neural Data Science
# ## Investigating Influential Factors of Alzheimer's Disease

# # Name(s)
# 
# - Shivani Suthar

# # Overview

# * In this study, I investigated the degree of influence that estimated total intracranial volume, normalized whole brain volume, gender, and socioeconomic status have on predicting Alzheimer's Disease. In order to do so, I ran 3 machine learning models - logistic regression, random forest, and XGBoost using a dataset with both demented and non-demented patients and chose the model with the best balance between accuracy and not underdiagnosing demented patients. In order to compare the degrees of influence that the variables have, I looked at the "feature importance" values of the variables outputted by the model. The reason I decided to conduct this investigation is because the causes of Alzheimer's Disease are not yet fully known so I wanted to contribute to the current bed of research by looking further into how the variables of interest can impact the risk of developing Alzherimer's since there are some preliminary studies suggesting their impact.

# <a id='research_question'></a>
# # Research Question

# * How do estimated total intracranial volume (eTIV), normalized whole brain volume (nWBV), gender, and socioeconomic status compare in their degree of influence in determining whether or not someone would be diagnosed with Alzheimer's Disease (AD)?
# 
# 

# <a id='background'></a>
# 
# ## Background & Prior Work

# According to the National Institute on Aging, “Experts suggest that more than 6 million Americans, most of them age 65 or older, may have Alzheimer's (NIH, 2023). In addition, according to Alzheimer's Disease International, “There are over 55 million people worldwide living with dementia in 2020. This number will almost double every 20 years, reaching 78 million in 2030 and 139 million in
# 2050” (Alzherimer's Disease International, n.d.). Clearly, Alzheimer's (which leads to dementia), is a very prevalent disease and is only increasing in frequency so it is important that it is understood further. This way, proper treatments and medications can be developed to help mitigate and manage the disease. Alzheimer's is also a progressive disease and has no cure so that further emphasizes the urgency and necessity of studying it; as of now, it is something that can just be managed. Additionally, according to the Alzheimer's Society, women are twice as likely compared to men to develop this deadly disease. Even when controlled for race, this seems to be the case (CDC, 2019). Another study found that "Women are disproportionately affected by Alzheimer's disease where nearly two-thirds of the more than 5 million Americans living with Alzheimer's are women" (Carrillo, 2016). However, there still aren't solidified reasons as to why this may be (Reed-Geaghan, 2022). Clearly however, gender does seem to be a large influencer on the diagnosis of AD.
# 
# In addition, in a study from the paper “Association of Socioeconomic Status With Dementia Diagnosis Among Older Adults In Denmark" (Wehberg et al., 2021), 10,191 individuals in Denmark from various socioeconomic backgrounds were gathered for a dementia diagnostic evaluation between 2017 and 2018. They found that individuals with a higher household income were less likely to receive a diagnosis for dementia and if they did, they were in a less severe cognitive stage compared to middle and lower-SES individuals. It was also seen in the paper "Education and Socioeconomic Determinants of Dementia and Alzheimer's Disease", that low SES in adult life is associated with poor nutrition, infectious diseases, chronic illness, and other factors that may lead to Alzheimer's (Mortimer et al., 1993). Similar to that found in the Wehberg paper, this paper also concluded that there seems to be a pattern of correlation between SES and Alzheimer's.
# 
# Lastly, according to the American Medical Association, "It has been suggested previously that a larger brain volume provides a greater cerebral reserve against the effects of Alzheimer's disease (AD), maintaining cognitive function in the presence of neurodegeneration and thereby delaying the onset of symptoms" (Jenkins, 2020). Another study by JMIR biomedical engineering found that "Attributes such as eTIV, nWBV, and ASF have a greater correlation in the prevalence of AD in women compared with men" demonstrating the the potential impact brain volume may have on AD and again, demonstrating the influence that gender has (Khan et al., 2020).
# 
# Given the current statistics on Alzheimer's and its increasing prevalence, I decided it would be interesting and crucial to better understand these influential factors of Alzheimer's disease. By understanding their specific degrees of influence and comparing them along biological and demographic lines, I can gain insight into how physical and social factors can influence one's likelihood of getting Alzheimer's disease.
# 
# References:
# - 1) “Alzheimer’s Disease Fact Sheet.” National Institute of Aging, 5 Apr. 2023., https://www.nia.nih.gov/health/alzheimers-and-dementia/alzheimers-disease-fact-sheet#:~:text=Estimates%20vary%2C%20but%20experts%20suggest,of%20dementia%20among%20older%20adults.
# - 2) Alzheimer's Disease International. “Dementia Statistics.” Alzheimer’s Disease International, 2020, https://www.alzint.org/about/dementia-facts-figures/dementia-statistics/#:~:text=Someone%20in%20the%20world%20develops,will%20be%20in%20developing%20countries.
# - 3) Carrillo M. Why Does Alzheimer’s Disease Affect More Women than Men? New Alzheimer’s Association Grant Will Help. Alzheimer’s Disease and Dementia. Published 2020. https://www.alz.org/blog/alz/february_2016/why_does_alzheimer_s_disease_affect_more_women_tha
# - 4) CDC. The Truth About Aging and Dementia | CDC. www.cdc.gov. Published September 26, 2019. https://www.cdc.gov/aging/publications/features/Alz-Greater-Risk.html
# - 5) Jenkins R, Fox NC, Rossor AM, Harvey RJ, Rossor MN. Intracranial Volume and Alzheimer Disease. Archives of Neurology. 2000;57(2):220. doi:https://doi.org/10.1001/archneur.57.2.220
# - 6) Khan A, Zubair S. Longitudinal Magnetic Resonance Imaging as a Potential Correlate in the Diagnosis of Alzheimer Disease: Exploratory Data Analysis. JMIR Biomedical Engineering. 2020;5(1):e14389. doi:https://doi.org/10.2196/14389
# - 7) “Mortimer, Education and Socioeconomic Determinants of Dementia and Alzheimer’s Disease.” Research Gate,
# www.researchgate.net/publication/232547140_Education_and_socioeconomic_determinants_of_
# dementia_and_Alzheimer’s_disease.
# www.alzint.org/about/dementia-facts-figures/dementia-statistics/.
# - 8) Petersen, Jindong Ding, et al. “Association of Socioeconomic Status with Dementia Diagnosis among
# Older Adults in Denmark.” JAMA Network Open, vol. 4, no. 5, 18 May 2021, pp.
# e2110432–e2110432, jamanetwork.com/journals/jamanetworkopen/fullarticle/2779936,
# https://doi.org/10.1001/jamanetworkopen.2021.10432.
# - 9) Reed-Geaghan, Erin. “Why Does Alzheimer’s Disease Affect More Women than Men? | BrightFocus
# Foundation.” Www.brightfocus.org, 1 Sept. 2022,
# www.brightfocus.org/alzheimers/article/why-does-alzheimers-disease-affect-more-women-men.

# # Hypothesis
# 

# I hypothesize that gender will have the most influence in determining whether or not someone will be diagnosed with Alzheimer's followed by socioeconomic status, and lastly, eTIV/nWBV (where I believe they will rank similarly). This is because as I wwas conducting research, I found there to be the greatest amount of conclusive studies demonstrating the impact of gender on AD, followed by socioeconomic status, and then eTIV/nWBV - the most critical of which were mentioned in the background section.

# # Dataset

# - Dataset Name: oasis_longitudinal (renamed to original_AD_dataset for clarity)
# - Link to the dataset:
#   - Link to Kaggle from which dataset was obtained: https://www.kaggle.com/datasets/jboysen/mri-and-alzheimers/data.
#   - Link to csv file uploaded to my google drive for more direct access: https://drive.google.com/file/d/11-lQk9P7_0GyB7-aR57hdUfeltZaieu2/view?usp=sharing
# - Number of observations: 373
# 
# - Dataset Description (from Kaggle):
# 
#   -                                                                       Kaggle obtained this dataset from "The Open Access Series of Imaging Studies"
#   (OASIS). According to Kaggle, "OASIS is a project aimed at making MRI data sets of the brain freely available to the scientific community". More specifically, this dataset , "Longitudinal MRI Data in Nondemented and Demented Older Adults", was acquired through a longitudinal collection of 150 subjects aged 60 to 96. Each subject was given a brain MRI scan on two or more
#   visits, separated by at least one year for a total of 373 imaging sessions. Each subject in this dataset is right-handed, including both men and women, and for each, 3 or 4 individual T1-weighted MRI scans obtained in single scan sessions are in the dataset. 72 of the subjects were characterized as
#   nondemented throughout the study. 64 of the included subjects were characterized as demented at the time of their initial visits and remained so for subsequent scans, including 51 individuals with mild to moderate Alzheimer’s disease. Another 14 subjects were characterized as nondemented at
#   the time of their initial visit and were subsequently characterized as demented at a later visit.
# 
# - Dataset Acquisition:
#   - I obtained this dataset directly from the Kaggle website linked above.
# 
# - Description of Columns:
#   - Subject ID: Patient identification number (e.g. OAS2_0005)
#   - MRI ID: Image identification number of an individual subject; No direct description
#   provided on website, can be inferred that it indicates which MRI scan it was for that
#   patient (e.g. OAS2_0005_MR3 can indicate that it’s the third MRI scan for patient
#   OAS2_0005)
#   - Group: Whether the patient is characterized as demented or non-demented (e.g.
#   “Demented”, “Nondemented”)
#   - Visit: Number of subject visits; No direct description provided on website, can be
#   inferred that it indicates the visit number of the patient (e.g. “4” would indicate it’s the
#   patient’s fourth MRI scan)
#   - MR Delay: Magnetic resonance delay; No direct description given on website, can be
#   inferred that it indicates the delay time that is prior to the MRI image procurement
#   (unsure of the units, could be seconds)
#   - M/F: Gender (e.g. “M”, “F”, where “M” is male and “F” is female)
#   - Hand: Handedness (e.g. “R” indicates the patients is right-handed) (all right handed in
#   this dataset)
#   - Age: The age of the patient in years
#   - EDUC: Subject education level (in years) (No direct description on website, but I can
#   infer that it is the number of years of education)
#   - SES: Socioeconomic status (No direct description of values provided on website but it
#   can be inferred that it’s a 1-5 scale where a “1” indicates a low SES and a “5” indicates a high SES
#   - MMSE: Mini-Mental State Examination (No direct description on website, can be
#   inferred from google that it’s a simple pen‐and‐paper test of cognitive function based on
#   a total possible score of 30 points where a higher score indicates higher cognitive
#   function)
#   - CDR: Clinical Dementia Rating (The CDR was measured as follows:{0= non-demented; 0.5 = very mild dementia; 1 = mild dementia; 2 = moderate dementia; Extreme—3})
#   - eTIV: Estimated Total Intracranial Volume (The eTIV variable estimates intracranial
#   brain volume including the brain, meninges (membranes of the brain), and CSF (cerebrospinal fluid)) (in mm^3)
#   - nWBV: Normalize Whole Brain Volume (This variable measures the volume of the brain tissue relative to the total intracranial volume (TIV) of an individual. It accounts for individual variations in head size and overall brain volume. In other words, nWBV represents the proportion of brain tissue volume compared to the total volume of the brain and surrounding structures within the skull, including the brain, meninges (membranes of the brain), and cerebrospinal fluid (CSF)) (in mg)
#   - ASF: Atlas scaling factor (No direct description on website, can be inferred that it’s a
#   one-parameter scaling factor that allows for comparison of the estimated total intracranial
#   volume (eTIV) based on differences in human anatomy (from google search))
# 
# 

# # Data Wrangling

# To pull the data in Python, I first downloaded the csv file from the link above and ran the following code in order to get the data into a dataframe. Below, a preview of the dataset can be seen:

# In[ ]:


from google.colab import files

uploaded = files.upload()


# In[ ]:


import pandas as pd

original_AD_dataset = pd.read_csv("/content/oasis_longitudinal.csv")
original_AD_dataset.head()


# # Data Cleaning

# In order to clean the data, what I first did is I removed all the rows except for the rows that had each patient's latest MRI scan details:

# In[ ]:


# Making sure the dataframe is sorted by 'Subject ID' and 'Visit' in ascending order
original_AD_dataset_sorted = original_AD_dataset.sort_values(by=['Subject ID', 'Visit'])

# Keeping only the rows with the latest visit for each patient
original_AD_dataset_latest_visits = original_AD_dataset_sorted.groupby('Subject ID').tail(1).reset_index(drop=True)

# Displaying the first few rows of the resulting dataframe
original_AD_dataset_latest_visits.head()


# Next, I removed any rows of data that had missing/NaN values for the variables of interest (eTIV, nWBV, gender, socioeconomic status, and "group") as those rows would not be useful:

# In[ ]:


columns_of_interest = ['eTIV', 'nWBV', 'M/F', 'SES', 'Group']
original_AD_dataset_cleaned = original_AD_dataset_latest_visits.dropna(subset=columns_of_interest)

original_AD_dataset_cleaned.head()


# Lastly, I changed any value of "converted" to "demented" in the "group" column as I am simply interested in whether or not someone eventually became demented:

# In[ ]:


original_AD_dataset_cleaned.loc[original_AD_dataset_cleaned['Group'] == 'Converted', 'Group'] = 'Demented'

original_AD_dataset_cleaned.head()


# Now, the resulting dataset has 149 rows and looks like the following:

# In[ ]:


original_AD_dataset_cleaned


# Generating summary statistics for the cleaned dataset:

# In[ ]:


summary_stats = original_AD_dataset_cleaned.describe()

summary_stats


# # Data Visualization

# For the EDA, first I wanted to see the distributions of the variables of interest:

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 2, figsize=(15, 15))

# Distribution plot for "Group" (Count Plot)
sns.countplot(data=original_AD_dataset_cleaned, x='Group', ax=axes[0, 0])
axes[0, 0].set_title('Distribution of Group')

# Distribution plot for "M/F" (Count Plot)
sns.countplot(data=original_AD_dataset_cleaned, x='M/F', ax=axes[0, 1])
axes[0, 1].set_title('Distribution of Gender')

# Distribution plot for "SES" (Count Plot)
sns.countplot(data=original_AD_dataset_cleaned, x='SES', ax=axes[1, 0])
axes[1, 0].set_title('Distribution of SES')

# Distribution plot for "eTIV" (Histogram)
sns.histplot(data=original_AD_dataset_cleaned, x='eTIV', bins=20, ax=axes[1, 1])
axes[1, 1].set_title('Distribution of eTIV')

# Distribution plot for "nWBV" (Histogram)
sns.histplot(data=original_AD_dataset_cleaned, x='nWBV', bins=20, ax=axes[2, 0])
axes[2, 0].set_title('Distribution of nWBV')

# Distribution plot for "Age" (Histogram)
sns.histplot(data=original_AD_dataset_cleaned, x='Age', bins=20, ax=axes[2, 1])
axes[2, 1].set_title('Distribution of Age')

plt.tight_layout()
plt.show()


# From the plots above, I can see that the "demented" and "non demented" groups are about evenly distributed, and the "eTIV", "nWBV", and "Age" variables look to have an approximate normal distribution shape (although the eTIV seems slightly right-skewed". However, more importantly, there seems to be more females than males and the SES graph looks to be right skewed. These are important observations to note to inform the analysis.

# Second, I wanted to see interactions between certain variables of interest that could potentially provide initial insights into addressing my research question/hypothesis:

# Let's start with looking at the interaction between gender and group:
# 
# 

# In[ ]:


sns.set(style="whitegrid")

plt.figure(figsize=(8, 6))
sns.countplot(data=original_AD_dataset_cleaned, x='M/F', hue='Group')
plt.title('Distribution of Demented and Non-Demented Individuals by Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.legend(title='Group')
plt.show()


# From the plot above, it can be seen that for men, there are many more demented patients than non-demented patients and for females, there are many more nondemented patients than demented patients indicating that gender could play a role in predicting whether or not someone would be demented or not. Interestingly, this is contrary to prior research that states that women are a lot more likely to develop AD.

# Now, let's look at the interaction between SES and group:

# In[ ]:


sns.set(style="whitegrid")

plt.figure(figsize=(10, 6))
sns.countplot(data=original_AD_dataset_cleaned, x='SES', hue='Group')
plt.title('Distribution of Demented and Non-Demented Individuals by Socioeconomic Status')
plt.xlabel('Socioeconomic Status (SES)')
plt.ylabel('Count')
plt.legend(title='Group')
plt.show()


# The plot above doesn't seem to present any clear relationships between SES and AD as for almost every SES group, there seem to be more demented patients than non-demented patients. However, for the SES group "2" (which would be classified as lower-SES), there seems to be many more non-demented patients than demented patients which is again interesting because according to prior research, people in lower SES groups are more likely to get AD.

# Now, let's look at the interaction between eTIV and group:

# In[ ]:


plt.figure(figsize=(10, 6))
sns.boxplot(data=original_AD_dataset_cleaned, x='Group', y='eTIV')
plt.title('Interaction between Estimated Total Intracranial Volume (eTIV) and Group')
plt.xlabel('Group')
plt.ylabel('Estimated Total Intracranial Volume (eTIV)')
plt.show()


# From the plot above, it can be seen that for demented and nondemented patients, they have a similar proportions in their spread of values however, the nondemented group has higher upper quartile values which is congruent with prior research suggesting that larger brain volume can protect against AD.

# Now, let's look at the interaction between nWBV and group:

# In[ ]:


plt.figure(figsize=(10, 6))
sns.boxplot(data=original_AD_dataset_cleaned, x='Group', y='nWBV')
plt.title('Interaction between Normalized Whole Brain Volume (nWBV) and Group')
plt.xlabel('Group')
plt.ylabel('Normalized Whole Brain Volume (nWBV)')
plt.show()


# The plot above shows that the median nWBV for non-demented patients is much higher than that of demented patients which is congruent with prior research which indicates that larger brain volume can protect against AD. Also, when compared to the eTIV graph where the medians were similar, it indicates that it might not necessarily be the simple magnitude of the brain volume that can protect against AD but the volume of the brain tissue relative to the eTIV that does.

# Given this finding and the fact that men tend to have larger heads than women, let's look at the interaction between nWBV and gender:
# 
# 

# In[ ]:


plt.figure(figsize=(10, 6))
sns.boxplot(x='M/F', y='nWBV', data=original_AD_dataset_cleaned)
plt.title('Interaction between Normalized Whole Brain Volume (nWBV) and Gender')
plt.xlabel('Gender')
plt.ylabel('Normalized Whole Brain Volume (nWBV)')
plt.xticks(ticks=[0, 1], labels=['Male', 'Female'])
plt.show()


# Based on these box plots, it seems as if females (or at least the ones in this dataset), have a higher nWBV overall than men given that their median value is much higher and their boxplot overall, is in the higher range. This is an interesting finding when compared to the distribution of demented and non-demented individuals by gender which showed that for men, there were many more demented patients than non-demented patients and for females, there were many more nondemented patients than demented patients. This could indicate a higher significance of nWBV than initially anticipated.

# Let's also compare the age distributions for males and females and the relationship between age and nWBV. This is because the risk of AD increases with age so if there seem to be older men than women in this dataset, that could potentially explain why there are more demented males than females in this dataset.

# In[ ]:


males = original_AD_dataset_cleaned[original_AD_dataset_cleaned['M/F'] == 'M']
females = original_AD_dataset_cleaned[original_AD_dataset_cleaned['M/F'] == 'F']

plt.figure(figsize=(12, 6))
sns.histplot(data=males, x='Age', kde=True, bins=30, color='blue')
plt.title('Age Distribution for Males')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(12, 6))
sns.histplot(data=females, x='Age', kde=True, bins=30, color='purple')
plt.title('Age Distribution for Females')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()


# In[ ]:


plt.figure(figsize=(10, 6))
sns.scatterplot(data=original_AD_dataset_cleaned, x='Age', y='nWBV', hue='Group', palette='viridis', alpha=0.7)

plt.title('Scatter Plot of Age vs. nWBV')
plt.xlabel('Age')
plt.ylabel('nWBV')
plt.legend(title='Group')

plt.show()


# From the plots above, it can be seen that the graph for males seems more left-skewed whereas the graph for females seems more right-skewed indicating that there do seem to be older men than women in this dataset overall. In addition, it seems that for both demented and non-demented individuals, as their age increases, their nWBV decreases further suggesting that nWBV may be more influential than originally anticipated as there are older men in this dataset and more men that are demented.

# Lastly, for observation sake, let's look at the interaction between SES and gender as historically, men tend to make more money than women:

# In[ ]:


plt.figure(figsize=(10, 6))
sns.countplot(x='SES', hue='M/F', data=original_AD_dataset_cleaned)
plt.title('Interaction between Socioeconomic Status and Gender')
plt.xlabel('Socioeconomic Status (SES)')
plt.ylabel('Count')
plt.legend(title='Gender', loc='upper right', labels=['Male', 'Female'])
plt.show()


# Interestingly enough, this plot does not seem to present any clear patterns between SES and gender but it seems that for the lower to middle range SES values, men seem to have a much higher SES than women and women only have a slightly higher SES than men in the upper SES bracket but then in the highest SES bracket, men have the most.

# Overall, from this EDA, I believe that the main takeaway is that gender and nWBV seem to have a high significance when determining whether or not someone would get AD which is contradictory to my original hypothesis where I thought that nWBV would not be that significant.

# # Data Analysis & Results

# For my data analysis, I will take the following steps:
# 
# 1. First, I will build the following ML models - logistic regression, random forest, and XGBoost in order to determine the degree of influence that each variable of interest has on whether or not someone is diagnosed as demented or not. The reason I chose logistic regression is because it is a simple and interpretable model that works well for binary classification problems. I chose random forest for a similar reason and also because it works well with both categorical and numerical features and it is robust to outliers. I chose XGBoost for its traditionally known high accuracy. In addition, based on the EDA, it seems to be important to correct for the fact that there are a fair amount of more females than males in the dataset. In my logistic regression model, I will correct for this by using scikit-learn's  "class_weight" parameter which will appropriately adjust weights to make the gender more balanced. I will do the same for the random forest model. In the XGBoost model, I will correct for this by using weighted sampling which will assign higher weights to males.
# 
# 2. Second, I will determine which model is the best by generating confusion matrices for each one and seeing which model seems to have the best balance between accuracy and a low false positive rate since I would want low false positives so that patients who do have Alzheimer's don't accidentally get delayed treatment. (I say false positive here because in the code below, "non-demented" was actually labeled as the positive class.)
# 
# 3. Third, after determining the best model, I will analyze the degree of influence each variable has on determining whether someone would be demented or not as part of the results. (If logistic regression is the best model, I will analyze the "coefficient" output for each variable of interest. If random forest is the best model, I will analyze the "feature importance score" for each variable of interest. If XGBoost is the best model, then I will analyze the "feature importance score" for each variable).

# Defining target and predictor variables:

# In[ ]:


X = original_AD_dataset_cleaned[['M/F', 'SES', 'eTIV', 'nWBV']]
y = original_AD_dataset_cleaned['Group']


# One-hot encoding the gender category:

# In[ ]:


X_encoded = pd.get_dummies(X, columns=['M/F'])


# Defining a function to plot confusion matrices:

# In[ ]:


def plot_confusion_matrix(matrix, labels):
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()


# Logistic Regression Model:

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Creating and training the logistic regression model
logreg = LogisticRegression(class_weight='balanced')
logreg.fit(X_train, y_train)

# Predicting on the test set
y_pred_logreg = logreg.predict(X_test)

# Feature importance
logreg_coefficients = pd.DataFrame({'Feature': X_train.columns, 'Coefficient': logreg.coef_[0]})
print("Logistic Regression Coefficients:")
print(logreg_coefficients)

# Generating confusion matrix and classification report
conf_matrix_logreg = confusion_matrix(y_test, y_pred_logreg)
class_report_logreg = classification_report(y_test, y_pred_logreg)

print("\nClassification Report (Logistic Regression):\n", class_report_logreg)

plot_confusion_matrix(conf_matrix_logreg, logreg.classes_)


# Random Forest Model:

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

# Creating and training the random forest model
rf = RandomForestClassifier(class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)

# Predicting on the test set
y_pred_rf = rf.predict(X_test)

# Feature importance
rf_feature_importance = pd.DataFrame({'Feature': X_train.columns, 'Importance': rf.feature_importances_})
print("Random Forest Feature Importance:")
print(rf_feature_importance)

# Generating confusion matrix and classification report
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
class_report_rf = classification_report(y_test, y_pred_rf)

print("\nClassification Report (Random Forest):\n", class_report_rf)

plot_confusion_matrix(conf_matrix_rf, rf.classes_)


# XGBoost Model:

# In[ ]:


from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.utils import compute_sample_weight

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)

# Computing sample weights to handle class imbalance
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

# Creating and training the XGBoost model
xgb = XGBClassifier(random_state=42)
xgb.fit(X_train, y_train, sample_weight=sample_weights)

# Predicting on the test set
y_pred_xgb = xgb.predict(X_test)

# Feature importance
xgb_feature_importance = pd.DataFrame({'Feature': X_train.columns, 'Importance': xgb.feature_importances_})
print("XGBoost Feature Importance:")
print(xgb_feature_importance)

# Generating confusion matrix and classification report
conf_matrix_xgb = confusion_matrix(y_test, y_pred_xgb)
class_report_xgb = classification_report(y_test, y_pred_xgb, target_names=label_encoder.classes_)

print("\nClassification Report (XGBoost):\n", class_report_xgb)

plot_confusion_matrix(conf_matrix_xgb, label_encoder.classes_)


# ### Results Summary:

# From the outputs of the model above, the following can be seen:
# 
# Logistic Regression -
# 
#   - SES: As someone's SES increases, their likelihood of being non-demented decreases. (Importance value, about -0.209978)
#   - eTIV: As someone's eTIV increases their likelihood of being non-demented increases. (Importance value, about 0.000835)
#   - nWBV: As someone's nWBV increases, their likelihood of being non-demented increases. (Importance value, about 0.393316)
#   - M/F_F: If someone is a female, the likelihood of them being non-demented increases. (Importance value, about 0.104351)
#   - M/F_M: If someone is a male, the likelihood of them being non-demented decreases. (Importance value, about -0.810052)
#   - Accuracy: This model accurately predicts about 55% of the time whether or not someone would be demented
#   - False Positive Rate: This model has a false positive rate of approximately .55 meaning that out of all the people that are demented, about 55% of them will be told that they're not.
#   - Overall: According to this model, the degree of importance that each variable has in order to determine if someone is demented or not in order from most important to least important is:
#     1. Being a male (increases your chances of being demented)
#     2. nWBV (the greater the nWBV, the greater your chance of being non-demented)
#     3. SES (the greater your SES, the greater your chance of being demented)
#     4. Being a female (decreases your chance of being demented)
#     5. eTIV (the greater the eTIV, the greater your chance of being non-demented)
# 
# 
# Random Forest -
# 
#    - SES: As someone's SES increases, their likelihood of being non-demented increases. (Importance value, about 0.125902)
#    - eTIV: As someone's eTIV increases, the likelihood of them being non-demented increases. (Importance value, about 0.345007)
#    - nWBV: As someone's nWBV increases, the likelihood of them being non-demented increases. (Importance value, about 0.472558)
#    - M/F_F: If someone is a female, the likelihood of them being non-demented increases. (Importance value, about 0.028365)
#    - M/F_M: If someone is a male, the likelihood of them being non-demented increases. (Importance value, about 0.028169)
#    - Accuracy: This model accurately predicts about 45% of the time whether or not someone would be demented
#    - False Positive Rate: This model has a false positive rate of approximately .72 meaning that out of all the people that are demented, about 72% of them will be told that they're not.
#    - Overall: According to this model, the degree of importance that each variable has in order to determine if someone is demented or not in order from most important to least important is:
#     1. nWBV (the greater the nWBV, the greater your chance of being non-demented)
#     2. eTIV (the greater the eTIV, the greater your chance of being non-demented)
#     3. SES (the greater your SES, the greater your chance of being non-demented)
#     4. Being a female (decreases your chance of being demented)
#     5. Being a male (decreases your chances of being demented)
# 
# XGBoost -
# 
#    - SES: As someone's SES increases, their likelihood of being non-demented increases. (Importance value, about 0.233999)
#    - eTIV: As someone's eTIV increases, the likelihood of them being non-demented increases. (Importance value, about 0.192318)
#    - nWBV: As someone's nWBV increases, the likelihood of them being non-demented increases. (Importance value, about 0.258324)
#    - M/F_F: If someone is a female, the likelihood of them being non-demented increases. (Importance value, about 0.315358)
#    - M/F_M: There is no correlation between being male and being demented/non-demented (Importance value, 0)
#    - Accuracy: This model accurately predicts about 55% of the time whether or not someone would be demented
#    - False Positive Rate: This model has a false positive rate of approximately .39 meaning that out of all the people that are demented, about 39% of them will be told that they're not.
#    - Overall: According to this model, the degree of importance that each variable has in order to determine if someone is demented or not in order from most important to least important is:
#     1. Being a female (decreases your chance of being demented)
#     2. nWBV (the greater the nWBV, the greater your chance of being non-demented)
#     3. SES (the greater your SES, the greater your chance of being non-demented)
#     4. eTIV (the greater the eTIV, the greater your chance of being non-demented)
#     5. Being a male (no correlation)

# # Conclusion & Discussion

# **Conclusion of Results:**
# 
# In conclusion, based on the results above and the context of your investigation, there are many ways in which the "best" model can be chosen and the variables can be ranked in terms of their importance in predicting AD. However, there are some general findings which seem to be consistent across all models. First, nWBV always ranked within the top 2 for significance. Second, SES always ranked as number 3. nWBV ranking within the top 2 for significance aligns with the findings in my EDA from the box plots of nWBV vs. group and nWBV vs. gender where there were clear differences in the box plots for nWBV between the demented and non-demented groups and the genders. There didn't seem to be any consistent findings for gender across the models which I found interesting because I thought the findings would likely either align with prior research in which if you're a woman you're more likely to get AD, or with the findings from my EDA in which it seemed that if you're a woman, you'd be less likely to get AD. However, there were no clear patterns in terms of gender across the models. This could potentially be due to the gender imbalance in the dataset or the difference in age distributions between the genders. Only in the logistic regression model did the findings align with my EDA in which being a woman decreases your chances of getting AD and being a man increases it but the model was lacking in other areas.
# 
# My original goal however, was to answer the question "How do estimated total intracranial volume (eTIV), normalized whole brain volume (nWBV), gender, and socioeconomic status compare in their degree of influence in determining whether or not someone would be diagnosed with Alzheimer's Disease (AD)" by determining the best model and looking at its results where the "best" model is defined by the model that has the best balance between accuracy and a low false positive rate (to avoid underdiagnosing positive patients). With this goal in mind, I believed that the best model would be the XGBoost model with an accuracy of 55% and a FPR of around 39%. Based on this model, I would answer my question as follows: The degree of importance that eTIV, nWBV, gender, and SES have in order to determine if someone is demented or not in order from most influential to least influential is:
# 
# 1. Being a female (decreases your chance of being demented)
# 2. nWBV (the greater the nWBV, the greater your chance of being non-demented)
# 3. SES (the greater your SES, the greater your chance of being non-demented)
# 4. eTIV (the greater the eTIV, the greater your chance of being non-demented)
# 5. Being a male (no correlation)
# 
# This does not align with my hypothesis in which I believed the ranking might be gender, socioeconomic status, and lastly, eTIV/nWBV (where I thought they would rank similarly). It seems as though both biological and social factors play a role in determining Alzheimer's with biological being slightly more influential. However, it does align with my EDA which hinted towards the influence that nWBV and being female would have.
# 
# Finally, the logistic regression model also performed similarly to the XGBoost model with an accuracy of 55% but had a much higher false positive rate of .55 so I believe that the results from my XGBoost model would be the most beneficial to move forward with for future work whether optimizing for accuracy, a low false positive rate, or both.
# 
# **Discussion of Limitations:**
# 
# While this study was successful in establishing preliminary ideas of what variables are the most important when predicting AD, there are certainly limitations of the study that need to be addressed.
# 
# First, as with any study, a larger sample size could have been useful in making the results of my study more generalizable. Second, including more predictors such as age could have given me a more complete view of the influencing factors of AD. Third, the results found from the study do not establish causation of AD but rather, provide insights into what factors could be correlated with AD which would then need to be further studied in the future. Lastly, there could have been potential confounds between variables in the study such as the effects of the patient's ages and ethnicities which were not accounted for.
# 
# **Future Directions:**
# 
# In the future, I believe that investigating age along with the other variables would greatly benefit this study as it has been shown that the likelihood of developing AD increases with age and there was a clear difference in ages between the men and women in this dataset which could have also been affecting the results of the gender importance.
# 
# I also think it would be interesting to incorporate data about patient's family history and lifestyles as it has been shown that genetics and certain lifestyle habits such as sleep patterns and smoking can cause Alzheimer's. This would allow for better preventive care solutions for patients to help protect them from AD.
# 
# Overall, I hope to continue this research in investigating the causes of Alzheimer's Disease to help further preventive care and treatment options.
