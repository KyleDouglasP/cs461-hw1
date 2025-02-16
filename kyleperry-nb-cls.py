import csv
import math
import numpy as np

# Given a variable, mean, and stdev, computes the probability of the variable based on a normal distribution
def gaussian(variable: float, mean: float, variance: float) -> float:
    return (1/math.sqrt(2*math.pi*variance))*math.exp(-(((variable-mean)**2)/(2*variance)))

SL_INDEX = 1
SW_INDEX = 2
PL_INDEX = 3
PW_INDEX = 4

# Virginica Constants
VI_SL_MEAN = 6.6225
VI_SL_VAR = 0.4679
VI_SW_MEAN = 2.96
VI_SW_VAR = 0.1132
VI_PL_MEAN = 5.6075
VI_PL_VAR = 0.3453
VI_PW_MEAN = 1.99
VI_PW_VAR = 0.0743

# Setosa Constants
SE_SL_MEAN = 5.0375
SE_SL_VAR = 0.1311
SE_SW_MEAN = 3.44
SE_SW_VAR = 0.1326
SE_PL_MEAN = 1.4625
SE_PL_VAR = 0.0296
SE_PW_MEAN = 0.2325
SE_PW_VAR = 0.0099

# Versicolor Constants
VE_SL_MEAN = 6.01
VE_SL_VAR = 0.2737
VE_SW_MEAN = 2.78
VE_SW_VAR = 0.1109
VE_PL_MEAN = 4.3175
VE_PL_VAR = 0.2035
VE_PW_MEAN = 1.35
VE_PW_VAR = 0.0431

correct = 0
count = 0

with open('test.csv') as test:

    reader = csv.reader(test)
    next(reader)

    for num,row in enumerate(reader):
        
        ground_truth = row[5]
        
        virginica_probability = (
            gaussian(float(row[SL_INDEX]),VI_SL_MEAN,VI_SL_VAR)
            *gaussian(float(row[SW_INDEX]),VI_SW_MEAN,VI_SW_VAR)
            *gaussian(float(row[PL_INDEX]),VI_PL_MEAN,VI_PL_VAR)
            *gaussian(float(row[PW_INDEX]),VI_PW_MEAN,VI_PW_VAR)
        )

        setosa_probability = (
            gaussian(float(row[SL_INDEX]),SE_SL_MEAN,SE_SL_VAR)
            *gaussian(float(row[SW_INDEX]),SE_SW_MEAN,SE_SW_VAR)
            *gaussian(float(row[PL_INDEX]),SE_PL_MEAN,SE_PL_VAR)
            *gaussian(float(row[PW_INDEX]),SE_PW_MEAN,SE_PW_VAR)
        )

        versicolor_probability = (
            gaussian(float(row[SL_INDEX]),VE_SL_MEAN,VE_SL_VAR)
            *gaussian(float(row[SW_INDEX]),VE_SW_MEAN,VE_SW_VAR)
            *gaussian(float(row[PL_INDEX]),VE_PL_MEAN,VE_PL_VAR)
            *gaussian(float(row[PW_INDEX]),VE_PW_MEAN,VE_PW_VAR)
        )

        classification = ""

        if virginica_probability > setosa_probability and virginica_probability > versicolor_probability:
            classification = "Iris-virginica"
        elif setosa_probability > virginica_probability and setosa_probability > versicolor_probability:
            classification = "Iris-setosa"
        else:
            classification = "Iris-versicolor"

        if classification==ground_truth: correct+=1

        count+=1

print(f'Testing accuracy: {100*(correct/count)}%')