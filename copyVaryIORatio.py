# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 14:51:30 2024

@author: allis
"""

import pandas as pd
import numpy as np
import concurrent.futures
import new_data_processing_hypoxia as dp
from differential_equations_hypoxia_advanced import radioimmuno_response_model
from BED import get_equivalent_bed_treatment
import pandas as pd
import time
import matplotlib.pyplot as plt
import ast
data = pd.read_csv('hypoxia RT 2 PD 2 CTLA4 2.csv')
#print(type(data['RT Treatment Days'].iloc))
#print(type(data['anti-PD-1 Treatment Days'].iloc))
#print(type(data['anti-CTLA-4 Treatment Days'].iloc))

# data['RT Treatment Days'] = data['RT Treatment Days'].apply(lambda x: list([i for i in x.strip('[]').split()]))
# data['anti-PD-1 Treatment Days'] = data['anti-PD-1 Treatment Days'].apply(lambda x: list([i for i in x.strip('[]').split()]))
# data['anti-CTLA-4 Treatment Days'] = data['anti-CTLA-4 Treatment Days'].apply(lambda x: list([i for i in x.strip('[]').split()]))
# data['RT Treatment Days'].tolist()
# data['anti-PD-1 Treatment Days'].tolist()
# data['anti-CTLA-4 Treatment Days'].tolist()
#print(type(data['RT Treatment Days'].iloc))
#print(type(data['anti-PD-1 Treatment Days'].iloc))
#print(type(data['anti-CTLA-4 Treatment Days'].iloc))

print(data)
TCPs = data['Mean TCP'].to_numpy()
optimal_TCP = np.max(TCPs)
times = data['Mean Time to reach Minimum Tumour Cell Count'].to_numpy()
t_treat_rad_optimal = data[data['Mean TCP'] == optimal_TCP]['RT Treatment Days'].tolist()
#print(data[data['Mean TCP'] == optimal_TCP]['RT Treatment Days'].tolist())
t_treat_rad_optimal = ast.literal_eval(t_treat_rad_optimal[0])
#print(t_treat_rad_optimal)
t_treat_p1_optimal = data[data['Mean TCP'] == optimal_TCP]['anti-PD-1 Treatment Days'].tolist()
t_treat_p1_optimal = ast.literal_eval(t_treat_p1_optimal[0])
#print(t_treat_p1_optimal)
t_treat_c4_optimal = data[data['Mean TCP'] == optimal_TCP]['anti-CTLA-4 Treatment Days'].tolist()
t_treat_c4_optimal = ast.literal_eval(t_treat_c4_optimal[0])
#data_PD = pd.read_csv('hypoxia RT 2 PD 2 a.csv')
#data_PD['anti-PD-1 Dose (mg)'] = 0.4
#data_PD['anti-CTLA-4 Dose (mg)'] = 0
#print(data_PD[(data_PD['RT Treatment Days'] == str([10])) & (data_PD['anti-PD-1 Treatment Days'] == str(t_treat_p1_optimal))].values.tolist())
#PD_TCP = data_PD[(data_PD['RT Treatment Days'] == str(t_treat_rad_optimal)) & (data_PD['anti-PD-1 Treatment Days'] == str(t_treat_p1_optimal))]['Mean TCP'].values
#print(t_treat_c4_optimal)
# for k in range(len(t_treat_c4_optimal) - 1):    
#     t_treat_c4_optimal[k] = int(t_treat_c4_optimal[k].strip(','))
# t_treat_c4_optimal[-1] = int(t_treat_c4_optimal[-1])
#print(t_treat_c4_optimal)
start_time = time.time()
params = pd.read_csv('hypoxia_parameters.csv').values.tolist()
sample_size = len(params)
#sample_size= 3
# Define the number of patients
#num_patients = 10


#recursive function to obtain treatment schedules
def get_treatment_schedules(n, t_optimal, start=10, tolerance=2):
  if n == 1:
    #return [[x] for x in range(start, 16)]
    minimum = max(t_optimal[0] - tolerance, start)
    maximum = min(t_optimal[0] + tolerance, 30)
    return [[x] for x in range(minimum, maximum + 1)]
  else:
    # first_days = []
    # minimum = max(t_optimal[0] - tolerance, 10)
    # maximum = min(t_optimal[0] + tolerance, 30)
    # for x in range(minimum, maximum + 1):
    #     first_days = first_days + [[x]]
    # print(first_days)
    # return first_days
    #print(t_optimal[0])
    minimum = max(t_optimal[0] - tolerance, 10)
    maximum = min(t_optimal[0] + tolerance, 30)
    return [[y] + rest for y in range(minimum, maximum + 1) for rest in get_treatment_schedules(n-1, t_optimal[1:], start=max(t_optimal[0] - tolerance, y+1))]


def get_treatment_and_dose(bioEffDose, numRT, param, numPD, numCTLA4, t_treat_rad_optimal, t_treat_p1_optimal, t_treat_c4_optimal):
  if numRT > 0:
      #print(t_treat_rad_optimal)
      RTschedule_list = get_treatment_schedules(numRT, t_treat_rad_optimal)
  if numPD > 0:
    PDschedule_list = get_treatment_schedules(numPD, t_treat_p1_optimal)
  else:
    PDschedule_list = []
  if numCTLA4 > 0:
    CTLA4schedule_list = get_treatment_schedules(numCTLA4, t_treat_c4_optimal)
    #print('ctla4', CTLA4schedule_list)
    #print(len(CTLA4schedule_list))
  else:
      CTLA4schedule_list = []
  schedule = []
  for x in RTschedule_list:
    for y in PDschedule_list:
        for z in CTLA4schedule_list:
            schedule.append([x, y, z])
  DList = []
  D = get_equivalent_bed_treatment(param, bioEffDose, numRT)
  for i in range(len(schedule)):
    DList.append(D)
  return schedule, DList

def get_TCP(C, times):
    for i in range(len(C[0])):
      if float(C[0][i]) == np.min(C):
        return np.exp(-1*np.min(C)), times[i]
#initialising parameters
free = [1,1,0]
LQL = 0
activate_vd = 0
use_Markov = True
T_0 = 1
dT = 0.98
t_f1 = 0
t_f2 = 50
delta_t = 0.05
# t_treat_c4 = np.zeros(3)
# t_treat_p1 = np.zeros(3)
c4 = 0
p1 = 0
PD_fractions = len(t_treat_p1_optimal)
CTLA4_fractions = len(t_treat_c4_optimal)
         #print('errors', errorMerged)
all_res_list = []
IT = (True, True)
RT_fractions = len(t_treat_rad_optimal)
param = [0, 0, 0.15769230769230763, 0.04269230769230769]
# param[26] = 0.13795567390561228
# param[27] = 0.4073542114448485
# param[28] = 0.04813514085703568
# param[33] = 0.0897670603865841
# param.append(2.2458318956090505*10**80)
bed = 80
file_name = 'RT ' + str(RT_fractions) + ' PD ' + str(PD_fractions) + ' CTLA4 ' + str(CTLA4_fractions) + ' varied IO ratio h.csv'
schedule_list, DList = get_treatment_and_dose(bed, RT_fractions, param, PD_fractions, CTLA4_fractions, t_treat_rad_optimal, t_treat_p1_optimal, t_treat_c4_optimal)
#print(len(schedule_list))
ratios = list(np.linspace(0,1,21)) #the proportion of IO total concentration that is anti-PD-1
print(ratios)
#ratios = ratios[0:3]
#print(ratios)
def evaluate_patient(i, t_rad, t_treat_p1, t_treat_c4, ratio):
  # Create a new random number generator with a unique seed for each patient
  paramNew = params[i]
  if IT == (True, True):
    paramNew[24] = 0.8*(1-ratio)/CTLA4_fractions
    paramNew[37] = 0.8*ratio/PD_fractions
    p1 = 0.8*ratio/PD_fractions
    c4 = 0.8*(1-ratio)/CTLA4_fractions
  elif IT == (False, True):
    paramNew[24] = 0.2/CTLA4_fractions
    paramNew[37] = 0
  elif IT == (True, False):
    paramNew[24] = 0
    paramNew[37] = 0.6/PD_fractions
  else:
    paramNew[24] = 0
    paramNew[37] = 0
    #print(paramNew)
  paramNew[0] = 100000.0
  D = DList[0]
  
  # t_rad = np.array(schedule_list[i][0])
  # #t_treat_c4 = np.zeros(3)
  # #t_treat_p1 = np.zeros(3)
  # t_treat_p1 = np.array(schedule_list[i][1])
  # t_treat_c4 = np.array(schedule_list[i][2])
  #t_f2 = max(schedule_list[i][0][0], schedule_list[i][1][0]) + 30
  t_f2 = max(max(t_rad[-1], t_treat_p1[-1]), t_treat_c4[-1]) +30
  #print('t_f2', t_f2)
  # if not isinstance(t_f2, int):
  #     t_f2 = t_f2[0]
  
  vol, radius, radius_H, t_eq, Time, _, C, *_ = radioimmuno_response_model(paramNew, delta_t, free, t_f1, t_f2, D, t_rad, t_treat_c4, t_treat_p1, LQL, activate_vd, use_Markov)
  #print('C', C)
  # plt.figure()
  # plt.plot(Time, vol[0], color='red')
  # plt.show()
  # plt.close()
  # plt.figure()
  # plt.plot(Time, C[0], color='blue')
  # plt.show()
  # plt.close()
  TCP, min_C_time = get_TCP(C, Time)
  return [TCP, min_C_time]
  # if dp.getTreatmentTime(Time, C) != None:
  #   #print(paramNew)
  #   #     plt.plot(Time, C[0])
  #   treatmentTime = dp.getTreatmentTime(Time, C)
  #   #print(str(t_rad) + " " + str(C) + str(treatmentTime)) 
  #   #print('time', treatmentTime)
  #   return treatmentTime
  # else:
  #   #print(Time[190:215])
  #   #print(C[0][190:215])
  #   #plt.figure()
  #   #plt.plot(Time, C[0])
  #   #plt.close()
  #   return np.nan


def trial_treatment(schedule, file, ratio):
  t_rad = schedule_list[schedule][0]
  #print('rad', t_rad)
  t_treat_p1 = schedule_list[schedule][1]
  t_treat_c4 = schedule_list[schedule][2]
  # print('p1', t_treat_p1)
  t_f2 = max(max(t_rad[-1], t_treat_p1[-1]), t_treat_c4[-1]) + 30
  #print('trial t_f2', t_f2)
  D = DList[schedule]
  #if ratio == 0:
      #treatment_res_list = data_PD[(data_PD['RT Treatment Days'] == str(t_treat_rad_optimal)) & (data_PD['anti-PD-1 Treatment Days'] == str(t_treat_p1_optimal))].values.tolist()[0]
  if ratio == 0.75:
    filtered_data = data[(data['RT Treatment Days'] == str(t_rad)) & 
                         (data['anti-PD-1 Treatment Days'] == str(t_treat_p1)) & 
                         (data['anti-CTLA-4 Treatment Days'] == str(t_treat_c4))]
    
    if not filtered_data.empty:
        treatment_res_list = filtered_data.values.tolist()[0]
        print(treatment_res_list)
    else:
        args = [(j, t_rad, t_treat_p1, t_treat_c4, ratio) for j in range(sample_size)]
  #print('args', args)
        res_list = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            res = list(executor.map(lambda p: evaluate_patient(*p), args))
            res = np.transpose(np.array(res))
            res_list.append(res)
          #print('res', res)
            TCPs = res[0]
            times = res[1]
          #print('tcp', TCPs)
          #print(times)
  #print(res_list)
  #treatment_times = [x for x in treatment_times if np.isnan(x) == False]
        treatment_res_list = [t_rad, D, t_treat_p1, 0.8*ratio/PD_fractions, t_treat_c4, 0.8*(1-ratio)/CTLA4_fractions, np.mean(TCPs), np.mean(times), TCPs, times]
    return treatment_res_list

  #print('args', args)                                                                                                        res_list = []                                                                                                           with concurrent.futures.ThreadPoolExecutor() as executor:                                                                   res = list(executor.map(lambda p: evaluate_patient(*p), args))                                                          res = np.transpose(np.array(res))                                                                                       res_list.append(res)                                                                                                    #print('res', res)                                                                                                      TCPs = res[0]                                                                                                           times = res[1]
          #print('tcp', TCPs)
          #print(times)                                                                                                   #print(res_list)                                                                                                        #treatment_times = [x for x in treatment_times if np.isnan(x) == False]                                                     treatment_res_list = [t_rad, D, t_treat_p1, 0.8*ratio/PD_fractions, t_treat_c4, 0.8*(1-ratio)/CTLA4_fractions, np.mean(TCPs), np.mean(times), TCPs, times]
  else:
      args = [(j, t_rad, t_treat_p1, t_treat_c4, ratio) for j in range(sample_size)]
  #print('args', args)
      res_list = []
      with concurrent.futures.ThreadPoolExecutor() as executor:
          res = list(executor.map(lambda p: evaluate_patient(*p), args))
          res = np.transpose(np.array(res))
          res_list.append(res)
          #print('res', res)
          TCPs = res[0]
          times = res[1]
          #print('tcp', TCPs)
          #print(times)
  #print(res_list)
  #treatment_times = [x for x in treatment_times if np.isnan(x) == False]
      treatment_res_list = [t_rad, D, t_treat_p1, 0.8*ratio/PD_fractions, t_treat_c4, 0.8*(1-ratio)/CTLA4_fractions, np.mean(TCPs), np.mean(times), TCPs, times]
  return treatment_res_list

# Define the treatment schedules and doses
#schedules, DList = get_treatment_and_dose(90, RT_fractions, param, PD_fractions, CTLA4_fractions)
#print(DList)
iterations = len(schedule_list)  # Or any other number of iterations
#param_file = open('parameters.txt', 'w')
# for k in range(min(iterations,50)):
#     print('k', k)
#     print(params[k])
#ratios = [0, 0.5,0.75]
args = [(schedule, params, ratio) for schedule in range(175,min(iterations,200)) for ratio in ratios]
# Use a ThreadPoolExecutor to run the iterations in parallel
with concurrent.futures.ThreadPoolExecutor() as executor:
    outputData = list(executor.map(lambda p: trial_treatment(*p), args))

    #print(data)
    # Retrieve results from completed futures

#print(data)
dataFrame = pd.DataFrame(outputData, columns=["RT Treatment Days", "RT Dose (Gy)", "anti-PD-1 Treatment Days", "anti-PD-1 Dose (mg)", "anti-CTLA-4 Treatment Days", "anti-CTLA-4 Dose (mg)", "Mean TCP", "Mean Time to reach Minimum Tumour Cell Count", "List of TCPs", "List of Times to Reach Minimum Tumour Cell Count"])
print(dataFrame)
dataFrame.to_csv(file_name, index=False)
end_time = time.time()
f = open('time taken RT ' + str(RT_fractions) + ' PD ' + str(PD_fractions) + ' CTLA4 ' + str(CTLA4_fractions) + ' treatment eval a constant seed.txt', 'w')
f.write("TIME TAKEN " + str(end_time - start_time))
print("TIME TAKEN " + str(end_time - start_time))
f.close()
print('iterations', iterations)
