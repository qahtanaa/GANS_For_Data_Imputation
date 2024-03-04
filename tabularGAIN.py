#Code written by Ouassim Bannany for completion of the Master Thesis at Universiteit Utrecht

#Required Libraries
import datetime
import numpy as np
import pandas as pd
from collections import Counter
import warnings
from sdv.tabular import CTGAN
from scipy.spatial import distance
from UtilFunctions import type_determinator, normalize_and_remap, denormalize_and_remap, imputer
warnings.filterwarnings('ignore')

'''
args: 
  - Data set including missing values (pandas df)
  - number of epochs to train each CTGAN model (default is 300)
  - Distinct ratio threshold (for data type detection)
  - generator learning rate (default is 0.0002)
  - discriminator learning rate (default is 0.0002)
  - verbose (True/False), whether to output the model's loss per iteration

output:
  - Imputed complete data frame
'''
#
#output: Imputed data set without missing values, ready to be used for analysis


def data_imputation_completer(df_with_NA_rows, nr_epochs, distinct_ratio_threshold, generator_lr, discriminator_lr, verbose):
  
  #Throw error if input is not a dataframe, or dataframe is empty
  if(isinstance(df_with_NA_rows, pd.DataFrame) == False or (isinstance(df_with_NA_rows, pd.DataFrame) and df_with_NA_rows.empty == True)):
    raise Exception('Input needs to be a pandas dataframe and may not be empty')
  
  #Step 1: automatically determine datatypes of the columns
  data_types = {}
  for col in df_with_NA_rows.columns:
    datatype = type_determinator(df_with_NA_rows[col],distinct_ratio_threshold)
    data_types.update({col:datatype})

  #print("Detected data types:\n", data_types)
  #Step 2: normalize/remap (with the ability to revert this back later)
  categorical_columns = []
  numerical_columns = []
  datetime_columns = []
  categorical_unique_columns = []

  for column, data_type in data_types.items():
    if(data_type == 'categorical'):
      categorical_columns.append(column)
    elif(data_type == 'numerical'):
      numerical_columns.append(column)
    elif(data_type == 'datetime'):
      datetime_columns.append(column)
    elif(data_type == 'categorical_unique'): #We don't do anything with this datetype group, because it hinders training and imputation (most times unique ID in string format, e.g. case number)
      categorical_unique_columns.append(column)

  #print("Number of categorical variables detected: ", len(categorical_columns))
  normalized_df = normalize_and_remap(df_with_NA_rows, categorical_columns, numerical_columns, datetime_columns, categorical_unique_columns)
  print('dataframe has been normalized...\n')

  #Step 3: split complete rows (to be used for training) from rows with atleast 1 missing value (to be imputed)
  rows_with_missing_values = normalized_df[normalized_df.isnull().any(axis=1)]
  complete_rows = normalized_df.dropna()
  
  if(len(rows_with_missing_values) == 0 or len(complete_rows) == 0):
    raise Exception('Dataframe has no complete rows to train on, or has no incomplete rows to impute')

  #Step 4: train the models 
  print('Training GAN Model 1: \n')
  model1 = CTGAN(verbose=verbose, epochs = nr_epochs, generator_lr=generator_lr, discriminator_lr=discriminator_lr)  #generator_lr and discriminator_lr(float): Learning rate for the generator. Both defaults to 2e-4.
  model1.fit(complete_rows) 
  print('Training GAN Model 2: \n')
  model2 = CTGAN(verbose=verbose, epochs = nr_epochs, generator_lr=generator_lr, discriminator_lr=discriminator_lr)
  model2.fit(complete_rows) 
  print('Training GAN model 3: \n')
  model3 = CTGAN(verbose=verbose, epochs = nr_epochs, generator_lr=generator_lr, discriminator_lr=discriminator_lr)
  model3.fit(complete_rows)   

  #Step 5: Imputing the missing rows
  print('Imputing missing values Model 1...\n')
  imputed_rows_m1= imputer(model1, rows_with_missing_values, categorical_columns)
  print('Imputing missing values Model 2...\n')
  imputed_rows_m2 = imputer(model2, rows_with_missing_values, categorical_columns)
  print('Imputing missing values Model 3...\n')
  imputed_rows_m3 = imputer(model3, rows_with_missing_values, categorical_columns)


  #step 6: Find out which of the models has the best imputation
  print('Evaluating candidate imputations: \n ')
  imputed_rows = rows_with_missing_values.copy()

  imputed_NA_count = 0
  missing_value_count = 0
  for idx, row in imputed_rows.iterrows():
    colCount = 0
    for value in row:
      if(pd.isnull(value) == True):
        missing_value_count += 1
        m1_imputed_val = imputed_rows_m1.loc[idx][colCount]
        m2_imputed_val = imputed_rows_m2.loc[idx][colCount]
        m3_imputed_val = imputed_rows_m3.loc[idx][colCount]
        print("(Candidate values) M1: ", m1_imputed_val, "M2: ", m2_imputed_val, "M3: ", m3_imputed_val)


        if(m1_imputed_val == m2_imputed_val and m2_imputed_val == m3_imputed_val): #If they are all 3 the same, we can impute it 
          imputed_rows.loc[idx,imputed_rows.columns[colCount]] = m1_imputed_val 

        #else pick majority vote
        elif(m1_imputed_val == m2_imputed_val and m2_imputed_val != m3_imputed_val):
          imputed_rows.loc[idx,imputed_rows.columns[colCount]] = m1_imputed_val    

        elif(m1_imputed_val == m3_imputed_val and m1_imputed_val != m2_imputed_val): 
          imputed_rows.loc[idx,imputed_rows.columns[colCount]] = m1_imputed_val    

        elif(m2_imputed_val == m3_imputed_val and m1_imputed_val != m3_imputed_val): 
          imputed_rows.loc[idx,imputed_rows.columns[colCount]] = m2_imputed_val     

        #If all 3 different, evaluate we just pick the first model. Evaluating which one is best is computationally expensive
        elif(m1_imputed_val != m2_imputed_val and m1_imputed_val != m3_imputed_val and m2_imputed_val != m3_imputed_val): 
          imputed_rows.loc[idx,imputed_rows.columns[colCount]] = m1_imputed_val
      colCount +=1


  #step 7: Combine datasets to have one dataframe without missing values
  total_df = pd.concat([complete_rows, imputed_rows], ignore_index=True, sort=False)

  #Step 8: Denormalize/remap imputed dataframe
  print('Denormalizing dataset...')
  denormalized_df = denormalize_and_remap(total_df, df_with_NA_rows, categorical_columns, numerical_columns, datetime_columns, categorical_unique_columns)
  
  return denormalized_df 
