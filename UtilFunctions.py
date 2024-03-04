#Code written by Ouassim Bannany for completion of the Master Thesis at Universiteit Utrecht


#util functions
#Required Libraries
import datetime
import numpy as np
import pandas as pd
from collections import Counter
import warnings
from sdv.tabular import CTGAN
from sdv.sampling import Condition
from scipy.spatial import distance
from sklearn.metrics import pairwise_distances
import random

def type_determinator(column_values, distinct_ratio_threshold):
  
  '''
  Args:
  - a list of values (column of pandas dataframe)
  - distinct ratio threshold which is used for categorical variables. 
        If the threshold is 0.1 it means that if the number of distinct values in the column is less than 10% of the number of rows, 
        the variable is perceived as categorical. Thus, increasing the threshold might classify more variables as categorical instead of numerical.

  Output:
  - the perceived data type (date_time, categorical, categorical_unique, numerical)

  This function allows for automatically detecting the datatypes of the columns that the user provides for imputation.
  (note categorical-unique are categorical values which have a very large amount of distinct values, and are to difficult to condition on in general)
  '''
  
  date_time_check = [] #As long as every value fits the datetime format, 'datetime' category is returned
  for value in column_values:
    try:
      value = datetime.datetime.strptime(value, '%Y-%m-%d')
    except:
      try:   
        value = datetime.datetime.strptime(value, '%Y-%m-%d %H:%M:%S')
      except:
        try:
          value = datetime.datetime.strptime(value, '%Y-%m-%d %H:%M:%S.%f')
        except:
          value = value
        
    if(isinstance(value, datetime.datetime) == True):
      date_time_check.append(True)
    else:
      date_time_check.append(False)

    if(False not in date_time_check):
      return 'datetime'  

  string_check = [] #If the values contain string, it can not be datetime/numerical. But we do differentiate between categorical/categorical_unique
  for value in column_values:
    string_check.append(isinstance(value, str))

  if(True in string_check and len(column_values.unique()) <= distinct_ratio_threshold*(len(column_values))):
    return 'categorical'
  elif(True in string_check and len(column_values.unique()) > distinct_ratio_threshold*(len(column_values))):
    return 'categorical_unique'


  column_values = np.array(column_values.dropna())
  c = Counter(column_values)

  list_without_most_common = column_values[column_values != c.most_common(1)[0][0]] #We create a new list after removing the most common value
  if(len(list_without_most_common) != 0):
    new_distinct_ratio = len(np.unique(list_without_most_common)) / len(list_without_most_common) #Calculate new distinct ratio after removal of most common value

    if(new_distinct_ratio < distinct_ratio_threshold):
      return 'categorical' #If the distinct ratio is lower than the threshold, we return categorical
  old_distinct_ratio = len(np.unique(column_values))/len(column_values)
  
  if(old_distinct_ratio < distinct_ratio_threshold):
    return 'categorical' #Similarly, if the old distinct ratio is lower than the threshold, we return categorical
  
  else:
    return 'numerical' #If all above checks are negative, the variable must be numerical.


def normalize_and_remap(df_with_NA_rows, categorical_variables, numerical_columns, datetime_columns, categorical_unique_columns):

  """
  Args:
    - The pandas dataframe that the user wants to impute
    - A list of variable names for each datatype category

  Output:
    - A pandas dataframe where:
      numerical values have been normalized within a range of [0,1]
      categorical values have been remapped within a range of [0, number_of_distinct_values]
      datetime values are transformed to a long int format

  This step is necessary such that the data is in the appropriate format to fit the GAN models, and also for fair comparisons
  so that variables with a big range do not dominate when calculating distance metrics.

  """
  temp_df = df_with_NA_rows.copy() 

  #First, remap all categorical variables:
  for variable in categorical_variables:
    temp_df[variable] = temp_df[variable].astype("category").cat.codes
    temp_df[variable] = temp_df[variable].replace({-1: np.nan})

  #Likewise with categorical_unique_columns
  for variable in categorical_unique_columns:
    temp_df[variable] = temp_df[variable].astype("category").cat.codes
    temp_df[variable] = temp_df[variable].replace({-1: np.nan})

  #Secondly, normalize all numerical variables with range [0,1]:
  for variable in numerical_columns:
    min = temp_df[variable].min()
    max = temp_df[variable].max()
    temp_df[variable] = (temp_df[variable] - min) / (max-min)

  #Lastly, transform datetimes to long int
  for variable in datetime_columns:
    datetime_df = pd.DataFrame({'date': temp_df[variable]})
    datetime_df_converted = pd.to_datetime(datetime_df["date"]).values.astype(np.int64)
    temp_df[variable] = np.where(datetime_df_converted == -9223372036854775808, np.nan, datetime_df_converted) #This long int refers to NaN, thus must be remapped back

  return temp_df

def denormalize_and_remap(normalized_df, original_df_with_NA, categorical_variables, numerical_columns, datetime_columns, categorical_unique_columns):
  """
  Args:
    - a pandas dataframe that has been normalized and remapped using the normalize_and_remap() function
    - The original dataframe before it has been normalized
    - a list of variable names for each datatype category
  Output:
    - A pandas dataframe the has been reverted back to the original format that the user has provided. This function is called after imputing all missing values.
  """

  temp_df = normalized_df.copy() #create a temporary dataframe 

  #First, remap all categorical variables:
  for variable in categorical_variables:
    remapped_list = []
    for value in temp_df[variable]:
      if(np.isnan(value) == True):
        remapped_list.append(np.nan)
      else:
        remapped_list.append(original_df_with_NA[variable].astype("category").cat.categories[value])

    temp_df[variable] = remapped_list

  #Likewise remap all categorical unique variables:
  for variable in categorical_unique_columns:
    remapped_list = []
    for value in temp_df[variable]:
      if(np.isnan(value) == True):
        remapped_list.append(np.nan)
      else:
        remapped_list.append(original_df_with_NA[variable].astype("category").cat.categories[value])

    temp_df[variable] = remapped_list

  #Secondly, denormalize all numerical variables:
  #Denormalize continuous variables: normalized * (max-min)+min
  for variable in numerical_columns:

    is_float = False #We need to know whether to denormalize as float (with decimals) or as int, depending on the input of the original dataset
    for value in original_df_with_NA[variable]:
      if(isinstance(value, float)):
        is_float = True

    if(is_float == True):
      temp_df[variable] =  (temp_df[variable] * (original_df_with_NA[variable].max() - original_df_with_NA[variable].min()) + original_df_with_NA[variable].min()).astype('float')
    elif(is_float == False):
      temp_df[variable] =  (temp_df[variable] * (original_df_with_NA[variable].max() - original_df_with_NA[variable].min()) + original_df_with_NA[variable].min()).astype('int')

  #Lastly, transform datetimes
  for variable in datetime_columns:
    old_df = pd.DataFrame({'date': temp_df[variable]})
    converted_df = pd.to_datetime(old_df["date"]).values.astype(str)

    datetimes = []
    for time in converted_df:
      if(pd.isnull(np.datetime64(time)) == True):
        datetimes.append(np.nan)
      else:
        datetimes.append(time[:10])

    temp_df[variable] = datetimes


  return temp_df 

def imputer(model, rows_to_impute, categorical_columns):

  """ 
  Args:
    - a ctgan() model that has been trained.
    - rows with atleast one missing value, that need to be imputed (pandas dataframe format)
    - a list of variable names that are of categorical datatype (on which we condition)

  output:
    - A complete pandas dataframe, where the missing values have been imputed using the model and observed values remain untouched.
  """

  temp_df = rows_to_impute.copy() #Create temporary dataframe
  imputation_count = 1 #Keeps track of the number of rows that have been imputed
  column_names = list(temp_df.columns) #List of column names

  for idx, row in temp_df.iterrows(): #For each row in the validation set with missing values we do:

    row = row.dropna().to_dict() #drop the missing values from the row 
    conditions_discrete_vars = {key: row[key] for key in row if key  in categorical_columns} #Store the column names of any non-missing categorical variable
    #print("nr Conditions ", len(conditions_discrete_vars)) <- can be print to keep track of the number of conditions
    while(len(conditions_discrete_vars) >= 0):
      try: #Create n (=50) samples conditioned on discrete values only, then using euclidean distance check which synthetic row is most similar to original row & impute its values
        if(len(conditions_discrete_vars) == 0):
            new_data = model.sample(num_rows=50) 
        else:   
          conditions = Condition(conditions_discrete_vars, num_rows = 50) #We create a condition object based on all discrete values in the row
          #Create n(=50) samples that satisfy the conditions of the discrete values, with max 100 tries 
          new_data = model.sample_conditions(conditions=[conditions], max_tries= 50) 
          
        variables_to_compare = {key: row[key] for key in row if key not in categorical_columns} #Save the continuous variables we need to compare the synthetic samples
        df_synthetic_data = pd.DataFrame(new_data, columns= [*variables_to_compare]) #create dataframe for all synthetic data samples

        #get euclidean distance per synthetic row
        dist_df = pd.DataFrame(columns=["index", "distance"])
        for _, currRow in df_synthetic_data.iterrows(): #we iterate through the n synthetically generated rows, based on the discrete values
            dist = distance.euclidean(list(variables_to_compare.values()), currRow.to_list()) #We compute the euclidean distance of the continuous variables that have been observed
            dist_df.loc[_,:] = [_, dist] #We save the distance in the dataframe, shorter distance means more similar continuous variables 


        #Find top 3 synthetic rows with shortest euclidean distance, impute based on majority vote, else use value of closest row
        dist_df = dist_df.sort_values(by=['distance']) #We first sort the indices in ascending order based on distance (so smallest distance comes on top)
        dist_df = dist_df.reset_index(drop = True) #We reset index, such that the most similar row will be index 0 
      


        if(len(dist_df) > 0 and len(dist_df) < 3): #If less than 3 rows were sampled, we just impute the synthetic row that has been sampled
          best_generated_row_idx = dist_df['index'][0]
          #print('row used to impute: ', new_data.loc[best_generated_row_idx])
          #print('\n imputation_count :', imputation_count, '( out of ', len(temp_df), ')')

          indx = 0
          for value in np.isnan(temp_df.loc[idx]): #We iterate through the columns of the original row, and if a value is missing, we impute the value of the synthetic row
            if(value == True):
              column_name = column_names[indx]
              first_place_value = new_data.loc[best_generated_row_idx][column_name]
              temp_df.loc[idx,column_name] = first_place_value #impute the value, where it first was missing. 
              #print('column: ', column_name, ' (only) candidate imputation: ', 'A: ', first_place_value)

            indx += 1

        else:
          best_generated_row_idx = dist_df['index'][0] #We save the index of the row in new_data which we need to use for imputation
          second_best_generated_row_idx = dist_df['index'][1]
          third_best_generated_row_idx = dist_df['index'][2]
          
          #print('\n imputation_count :', imputation_count, '( out of ', len(temp_df), ')')

          #Replace old row with new row without missing values
          indx = 0
          for value in np.isnan(temp_df.loc[idx]): #similar to step 1, we only impute the values that are missing, observed values stay untouched
            if(value == True):
              column_name = column_names[indx]

              first_place_value = new_data.loc[best_generated_row_idx][column_name]
              second_place_value = new_data.loc[second_best_generated_row_idx][column_name]
              third_place_value = new_data.loc[third_best_generated_row_idx ][column_name]

              #print('column: ', column_name, ' Candidate imputations: ', 'A: ', first_place_value, 'B: ',second_place_value, 'C: ', third_place_value)

              if((first_place_value != second_place_value) and (second_place_value == third_place_value)):
                #print('row used to impute: ', new_data.loc[second_place_value])
                temp_df.loc[idx,column_name] = second_place_value

              else:
                temp_df.loc[idx,column_name] = first_place_value #impute the value, where it first was missing
                #print('row used to impute: ', new_data.loc[best_generated_row_idx])

              #print('row after imputation:', temp_df.loc[idx])
            indx += 1

      except ValueError:   #if both step 1 and step 2 did not work, we remove the row from the comparison as imputation becomes too unreliable
        if(len(conditions_discrete_vars) >= 1):
          conditions_discrete_vars.pop(random.choice(list(conditions_discrete_vars.keys()))) #Randomly remove one condition
          #print('Retrying with less conditions \n') 
          continue
        else:
          break
          
      imputation_count += 1
      break

      
  
  return temp_df
  

### Functions for evaluation

def calc_avg_dissimilarity(list_of_values):
  list_of_values = list(list_of_values) 
  dissimilarityCount = 0
  similarityCount = 0
  for idx, value in enumerate(list_of_values):
    #print(idx, value, list(list_of_values[idx+1:]))
    for i in range(idx+1, len(list_of_values)-1):
      if(value == list_of_values[i]):
        similarityCount +=1
      else:
        dissimilarityCount +=1

  return dissimilarityCount/(dissimilarityCount+similarityCount)

def create_masks(test_dataframe, data_m, categorical_columns, numerical_columns, datetime_columns, categorical_unique_columns):
  temp_df = test_dataframe.copy()
  nr_row, nr_col = temp_df.shape
  data_m_numerical = np.full((nr_row, nr_col), False)
  data_m_categorical = np.full((nr_row, nr_col), False)

  categorical_variables = categorical_columns + categorical_unique_columns
  numerical_variables = numerical_columns + datetime_columns

  categoricalColIndices = []
  numericalColIndices = []

  idxCount = 0
  for col in temp_df.columns:
    if(col in categorical_variables):
      categoricalColIndices.append(idxCount)
    elif(col in numerical_variables):
      numericalColIndices.append(idxCount)
    idxCount +=1


  #create seperate masks for numerical/categorical

  rowCount = 0
  for row in data_m:
    colCount=0
    for value in row:
      if(colCount in categoricalColIndices):
        if(value == True):
          data_m_categorical[rowCount,colCount] = True
      elif(colCount in numericalColIndices):
        if(value == True):
          data_m_numerical[rowCount,colCount] = True
      colCount+=1
    rowCount+=1

  return data_m_categorical, data_m_numerical



def evaluate_imputations(original_training_data, test_set_complete, data_mask, test_set_imputed, utilized_distinct_ratio_threshold):
  #Step 1: automatically determine datatypes of the columns
  data_types = {}
  for col in original_training_data.columns:
    datatype = type_determinator(original_training_data[col],utilized_distinct_ratio_threshold)
    data_types.update({col:datatype})

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

  #Create masks of missingness for numerical & categorical variables
  data_m_categorical, data_m_numerical = create_masks(test_set_complete, data_mask, categorical_columns, numerical_columns, datetime_columns, categorical_unique_columns)


  #Begin evaluation
  temp_df_unimputed = test_set_complete.copy()
  temp_df_imputed = test_set_imputed.copy()


  #Step 0: normalize dataframes
  temp_df_unimputed = normalize_and_remap(temp_df_unimputed, categorical_columns, numerical_columns, datetime_columns, categorical_unique_columns)
  temp_df_imputed = normalize_and_remap(temp_df_imputed, categorical_columns, numerical_columns, datetime_columns, categorical_unique_columns)

  #normalize datetime from long int to float range [0,1] before comparison 
  for col in datetime_columns:
    min = temp_df_unimputed[col].min()
    max = temp_df_unimputed[col].max()
    temp_df_unimputed[col] = (temp_df_unimputed[col] - min) / (max-min)
    temp_df_imputed[col] = (temp_df_imputed[col] - min)/(max-min)

  #Step 1: average of average pairwise distances
  numerical_variables = numerical_columns + datetime_columns
  
  avgDistances_numerical = []
  for variable in numerical_variables:
    valuesArray = np.array(temp_df_unimputed[variable])
    matrix = pairwise_distances(valuesArray.reshape(-1,1), metric='euclidean')
    avgDistances_numerical.append(np.average(matrix))

  avg_avg_pairwise_dis = np.mean(avgDistances_numerical)

  #Step 2: average distance of categorical variables 
  categorical_variables = categorical_columns + categorical_unique_columns
  
  avg_distances_categorical = []

  for variable in categorical_variables:
    avg_distances_categorical.append(calc_avg_dissimilarity(temp_df_unimputed[variable]))

  avg_avg_distances_categorical = np.mean(avg_distances_categorical)

  #step 3: calculate alpha
  alpha = avg_avg_pairwise_dis / avg_avg_distances_categorical

  #step 4: calculate average prediction error for numerical columns

  rowCount, totalComparisons, error = 0,0,0

  for row in data_m_numerical:
    columnCount = 0
    for value in row:
      if(value == True):
        realValue = temp_df_unimputed.iloc[rowCount][columnCount]
        predictedValue = temp_df_imputed.iloc[rowCount][columnCount]
        error += abs(realValue-predictedValue)
        totalComparisons += 1
      columnCount +=1
    rowCount +=1

  avgErrorNumerical = error/totalComparisons
  
  #step 5: calculate average prediction error for categorical columns
  rowCount, totalComparisons, error = 0,0,0

  for row in data_m_categorical:
    columnCount = 0
    for value in row:
      if(value == True):
        realValue = temp_df_unimputed.iloc[rowCount][columnCount]
        predictedValue = temp_df_imputed.iloc[rowCount][columnCount]
        if(realValue != predictedValue):
          error += 1
        totalComparisons += 1
      columnCount +=1
    rowCount +=1

  avgErrorCategorical = error/totalComparisons


  #step 6: apply formula and return overall prediction error for all imputations
  return ((avgErrorNumerical) + (alpha*avgErrorCategorical))/2
