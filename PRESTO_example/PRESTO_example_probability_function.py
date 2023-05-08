#import numpy as np
import pandas as pd
#import csv
'''from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc, f1_score, precision_score, recall_score, precision_recall_curve#, plot_precision_recall_curve 
from sklearn.ensemble import RandomForestClassifier                                             
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold, StratifiedKFold 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier'''
#import lightgbm as lgb
'''from sklearn.feature_selection import RFE,RFECV
import scipy.io
from scipy import stats
import matplotlib.pyplot as plt'''
#import seaborn as sns
'''import sklearn
import math'''
import pickle
import warnings
warnings.filterwarnings("ignore")
'''sklearn.__version__
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)'''


#DATA_PATH = input("Enter the path to local destination folder: ") # /home/zahra/.../PRESTO_example
#DATA_PATH = "/Users/leonardo/code/django/Predictive-models-of-pregnancy/PRESTO_example/"

#import os
#os.listdir(str(DATA_PATH))#DATA_PATH+

def pregnancy_probability():
    
    Menstrual_cycle_length_days = input("Enter Menstrual cycle length (days): ")# for example: 30

    Female_age_at_baseline_years = input("Enter Female age at baseline (years): ")# for example: 30

    Urbanization_of_residential_area__rural = input("Enter 1 if Urbanization of residential area is rural, otherwise enter 0:  ")

    Previously_tried_to_conceive_for_greater_than_or_equal_to_12_months__yes = input("Enter 1 if Previously tried to conceive for ≥12 months, otherwise enter 0:  ")

    One_menstrual_cycle_of_attempt_time_at_study_entry = input("Enter 1 if One menstrual cycle of attempt time at study entry, otherwise enter 0: ")

    Daily_use_of_multivitamins_or_folic_acid = input("Enter 1 if Daily use of multivitamins/folic acid, otherwise enter 0: ")

    Last_method_of_contraception__hormonal_IUD = input("Enter 1 if Last method of contraception: hormonal IUD, otherwise enter 0: ")

    Female_BMI_kg_m2 = input("Enter Female BMI (kg/m2): ")# for example: 27

    Ever_breastfed_an_infant = input("Enter 1 if Ever breastfed an infant, otherwise enter 0: ")

    Ever_been_pregnant = input("Enter 1 if Ever been pregnant, otherwise enter 0: ")

    Female_education_years = input("Enter Female education (years): ")# for example: 16

    Received_influenza_vaccine_in_the_past_year = input("Enter 1 if Received influenza vaccine in the past year, otherwise enter 0: ")

    Perceived_Stress_Scale_score = input("Enter Stress (Perceived Stress Scale score): ")# for example: 15

    Total_number_of_pregnancies = input("Enter Total number of pregnancies: ")# for example: 2

    Urbanization_of_residential_area__Canada = input("Enter 1 if Urbanization of residential area is Canada, otherwise enter 0:  ")

    Urbanization_of_residential_area__urban_cluster = input("Enter 1 if Urbanization of residential area is urban cluster, otherwise enter 0:  ")

    Previously_tried_to_conceive_for_greater_than_or_equal_to_months__no_never_tried_before = input("Enter 1 if the answer to question (Previously tried to conceive for ≥12 months) is “no, never tried before” (ref = “no, tried for < 12 months”), otherwise enter 0: ")

    
    variable_dict = {
        'ageatqstn' : (float(Female_age_at_baseline_years)-29.805008)/3.751832, #standardization
        'b_everpregnant': (float(Ever_been_pregnant)-0.501721)/0.500075,
        'b_gravid': (float(Total_number_of_pregnancies)-0.964632)/1.392237,
        'b_fluvacc': (float(Received_influenza_vaccine_in_the_past_year)-0.525196)/0.499443,
        'pregsupp': (float(Daily_use_of_multivitamins_or_folic_acid)-0.836620)/0.36977,
        'pss_score': (float(Perceived_Stress_Scale_score)-15.451330)/5.76934,
        'b_breastfeedever': (float(Ever_breastfed_an_infant)-0.307042)/0.461339,
        'Cont_hormoneiud': (float(Last_method_of_contraception__hormonal_IUD)-0.115493)/0.319666,
        'MenstruationCyclus': (float(Menstrual_cycle_length_days)-29.596017)/3.982703,
        'ttp_entry': (float(One_menstrual_cycle_of_attempt_time_at_study_entry)-0.578091)/0.493941,
        'bmi': (float(Female_BMI_kg_m2)-26.613678)/6.519338,
        'b_conteduc': (float(Female_education_years)-16.012207)/1.244465,
        'b_trypregnant_1.0': (float(Previously_tried_to_conceive_for_greater_than_or_equal_to_12_months__yes)-0.046009)/0.209538,
        'ua_cat_1.0': (float(Urbanization_of_residential_area__rural)-0.043505)/0.204024,
        'b_trypregnant_3.0': (float(Previously_tried_to_conceive_for_greater_than_or_equal_to_months__no_never_tried_before)-0.416275)/0.493018,
        'ua_cat_2.0': (float(Urbanization_of_residential_area__urban_cluster)-0.079499)/0.270559,
        'ua_cat_4.0': (float(Urbanization_of_residential_area__Canada)-0.183412)/0.387064
    }
    
    
    
    df = pd.DataFrame(columns=list(list(variable_dict.keys())))

    df = df.append(variable_dict, ignore_index=True)
    
    my_seed = 2020
    prob = []
    pred = []
    my_seeds=range(my_seed, my_seed+5) # the random_state that controls the shuffling applied to the data before applying the split
    for seed in my_seeds:
        filename = f'clf_{seed}'
        clf = f'clf_{seed}'
        clf = pickle.load(open(filename, 'rb'))
        pred.append(clf.predict(df.values))
        prob.append(clf.predict_proba(df.values)[:,1])

    print()
    mean = sum(prob) / len(prob)
    print('The probability of pregnancy within 12 menstural cycles of pregnancy attempt time is: ' + str(round(mean[0],2)))


def pregnancy_probability_w_input( input_data ):
    
    Menstrual_cycle_length_days = input_data[0]

    Female_age_at_baseline_years = input_data[1]

    Urbanization_of_residential_area__rural = input_data[2]

    Previously_tried_to_conceive_for_greater_than_or_equal_to_12_months__yes = input_data[3]

    One_menstrual_cycle_of_attempt_time_at_study_entry = input_data[4]

    Daily_use_of_multivitamins_or_folic_acid = input_data[5]

    Last_method_of_contraception__hormonal_IUD = input_data[6]

    Female_BMI_kg_m2 = input_data[7]

    Ever_breastfed_an_infant = input_data[8]

    Ever_been_pregnant = input_data[9]

    Female_education_years = input_data[10]

    Received_influenza_vaccine_in_the_past_year = input_data[11]

    Perceived_Stress_Scale_score = input_data[12]

    Total_number_of_pregnancies = input_data[13]

    Urbanization_of_residential_area__Canada = input_data[14]

    Urbanization_of_residential_area__urban_cluster = input_data[15]

    Previously_tried_to_conceive_for_greater_than_or_equal_to_months__no_never_tried_before = input_data[16]

    
    variable_dict = {
        'ageatqstn' : (float(Female_age_at_baseline_years)-29.805008)/3.751832, #standardization
        'b_everpregnant': (float(Ever_been_pregnant)-0.501721)/0.500075,
        'b_gravid': (float(Total_number_of_pregnancies)-0.964632)/1.392237,
        'b_fluvacc': (float(Received_influenza_vaccine_in_the_past_year)-0.525196)/0.499443,
        'pregsupp': (float(Daily_use_of_multivitamins_or_folic_acid)-0.836620)/0.36977,
        'pss_score': (float(Perceived_Stress_Scale_score)-15.451330)/5.76934,
        'b_breastfeedever': (float(Ever_breastfed_an_infant)-0.307042)/0.461339,
        'Cont_hormoneiud': (float(Last_method_of_contraception__hormonal_IUD)-0.115493)/0.319666,
        'MenstruationCyclus': (float(Menstrual_cycle_length_days)-29.596017)/3.982703,
        'ttp_entry': (float(One_menstrual_cycle_of_attempt_time_at_study_entry)-0.578091)/0.493941,
        'bmi': (float(Female_BMI_kg_m2)-26.613678)/6.519338,
        'b_conteduc': (float(Female_education_years)-16.012207)/1.244465,
        'b_trypregnant_1.0': (float(Previously_tried_to_conceive_for_greater_than_or_equal_to_12_months__yes)-0.046009)/0.209538,
        'ua_cat_1.0': (float(Urbanization_of_residential_area__rural)-0.043505)/0.204024,
        'b_trypregnant_3.0': (float(Previously_tried_to_conceive_for_greater_than_or_equal_to_months__no_never_tried_before)-0.416275)/0.493018,
        'ua_cat_2.0': (float(Urbanization_of_residential_area__urban_cluster)-0.079499)/0.270559,
        'ua_cat_4.0': (float(Urbanization_of_residential_area__Canada)-0.183412)/0.387064
    }
    
    
    
    df = pd.DataFrame(columns=list(list(variable_dict.keys())))

    df = df.append(variable_dict, ignore_index=True)
    
    my_seed = 2020
    prob = []
    pred = []
    my_seeds=range(my_seed, my_seed+5) # the random_state that controls the shuffling applied to the data before applying the split
    for seed in my_seeds:
        filename = f'clf_{seed}'
        clf = f'clf_{seed}'
        clf = pickle.load(open(filename, 'rb'))
        pred.append(clf.predict(df.values))
        prob.append(clf.predict_proba(df.values)[:,1])

    print()
    mean = sum(prob) / len(prob)
    #print('The probability of pregnancy within 12 menstural cycles of pregnancy attempt time is: ' + str(round(mean[0],2)))
    num = int( round(mean[0]*100,0) )
    #print('Age: ' + str(input_data[1]) + ', probability: ' + str(num) + '%' )
    print(str(input_data[1]) + '        ' + str(num) + '%' )

    print(prob)
    print("\n")
    print(pred[0])
    print("\n")
    print(dir(clf))
    
    #return round(mean[0],2)



'''
Menstrual_cycle_length_days = input_data[0]
Female_age_at_baseline_years = input_data[1]
Urbanization_of_residential_area__rural = input_data[2]
Previously_tried_to_conceive_for_greater_than_or_equal_to_12_months__yes = input_data[3]
One_menstrual_cycle_of_attempt_time_at_study_entry = input_data[4]
Daily_use_of_multivitamins_or_folic_acid = input_data[5]
Last_method_of_contraception__hormonal_IUD = input_data[6]
Female_BMI_kg_m2 = input_data[7]
Ever_breastfed_an_infant = input_data[8]
Ever_been_pregnant = input_data[9]
Female_education_years = input_data[10]
Received_influenza_vaccine_in_the_past_year = input_data[11]
Perceived_Stress_Scale_score = input_data[12]
Total_number_of_pregnancies = input_data[13]
Urbanization_of_residential_area__Canada = input_data[14]
Urbanization_of_residential_area__urban_cluster = input_data[15]
Previously_tried_to_conceive_for_greater_than_or_equal_to_months__no_never_tried_before = input_data[16]
'''

input_scenario = [30, 30, 1, 1, 0, 0, 1, 25, 0, 0, 18, 0, 50, 0, 0, 0, 0]

mcd = 0     # more cycle days
bmi = 5     # more bmi
pss = 0     # more pss
years = 0   # more education years
can = 0     # live in Canada

input_array = [
    [28+mcd, 18, 0, 0, 0, 1, 1, 20+bmi, 0, 0, 18+years, 1, 15+pss, 0, can, 1, 1],
    [28+mcd, 19, 0, 0, 0, 1, 1, 20+bmi, 0, 0, 18+years, 1, 15+pss, 0, can, 1, 1],
    [28+mcd, 20, 0, 0, 0, 1, 1, 20+bmi, 0, 0, 18+years, 1, 15+pss, 0, can, 1, 1],
    [28+mcd, 21, 0, 0, 0, 1, 1, 20+bmi, 0, 0, 18+years, 1, 15+pss, 0, can, 1, 1],
    [28+mcd, 22, 0, 0, 0, 1, 1, 20+bmi, 0, 0, 18+years, 1, 15+pss, 0, can, 1, 1],
    [28+mcd, 23, 0, 0, 0, 1, 1, 20+bmi, 0, 0, 18+years, 1, 15+pss, 0, can, 1, 1],
    [28+mcd, 24, 0, 0, 0, 1, 1, 20+bmi, 0, 0, 18+years, 1, 15+pss, 0, can, 1, 1],
    [28+mcd, 25, 0, 0, 0, 1, 1, 20+bmi, 0, 0, 18+years, 1, 15+pss, 0, can, 1, 1],
    [28+mcd, 26, 0, 0, 0, 1, 1, 20+bmi, 0, 0, 18+years, 1, 15+pss, 0, can, 1, 1],
    [28+mcd, 27, 0, 0, 0, 1, 1, 20+bmi, 0, 0, 18+years, 1, 15+pss, 0, can, 1, 1],
    [28+mcd, 28, 0, 0, 0, 1, 1, 20+bmi, 0, 0, 18+years, 1, 15+pss, 0, can, 1, 1],
    [28+mcd, 29, 0, 0, 0, 1, 1, 20+bmi, 0, 0, 18+years, 1, 15+pss, 0, can, 1, 1],
    [28+mcd, 30, 0, 0, 0, 1, 1, 20+bmi, 0, 0, 18+years, 1, 15+pss, 0, can, 1, 1],
    [28+mcd, 31, 0, 0, 0, 1, 1, 20+bmi, 0, 0, 18+years, 1, 15+pss, 0, can, 1, 1],
    [28+mcd, 32, 0, 0, 0, 1, 1, 20+bmi, 0, 0, 18+years, 1, 15+pss, 0, can, 1, 1],
    [28+mcd, 33, 0, 0, 0, 1, 1, 20+bmi, 0, 0, 18+years, 1, 15+pss, 0, can, 1, 1],
    [28+mcd, 34, 0, 0, 0, 1, 1, 20+bmi, 0, 0, 18+years, 1, 15+pss, 0, can, 1, 1],
    [28+mcd, 35, 0, 0, 0, 1, 1, 20+bmi, 0, 0, 18+years, 1, 15+pss, 0, can, 1, 1],
    [28+mcd, 36, 0, 0, 0, 1, 1, 20+bmi, 0, 0, 18+years, 1, 15+pss, 0, can, 1, 1],
    [28+mcd, 37, 0, 0, 0, 1, 1, 20+bmi, 0, 0, 18+years, 1, 15+pss, 0, can, 1, 1],
    [28+mcd, 38, 0, 0, 0, 1, 1, 20+bmi, 0, 0, 18+years, 1, 15+pss, 0, can, 1, 1],
    [28+mcd, 39, 0, 0, 0, 1, 1, 20+bmi, 0, 0, 18+years, 1, 15+pss, 0, can, 1, 1],
    [28+mcd, 40, 0, 0, 0, 1, 1, 20+bmi, 0, 0, 18+years, 1, 15+pss, 0, can, 1, 1]
]
print('\nCycle days = ' + str(28+mcd))
print('BMI = ' + str(20+bmi))
print('PSS = ' + str(15+pss))
print('Education = ' + str(18+years))
print('Canada = ' + str(can))
print()
print('Age      Probability')

#for input_data in input_array:
#    pregnancy_probability_w_input( input_data )

pregnancy_probability_w_input( input_array[0] )

print()

#for i in range(18, 41):
#    print('[28+mcd, ' + str(i) + ', 0, 0, 0, 1, 1, 20+bmi, 0, 0, 18+years, 1, 15+pss, 0, 0, 1, 1],')