import csv
import numpy as np

# We need to take each attribute for each classification, computing the mean and variance, to create a Guassian distribution
# With this Guassian distribution we can compute probabilities of being each classification

SL_INDEX = 0
SW_INDEX = 1
PL_INDEX = 2
PW_INDEX = 3

# Iris Virginica
virginica_count = 0
virginica_data = [[],[],[],[]]

# Iris Setosa
setosa_count = 0
setosa_data = [[],[],[],[]]

# Iris Iris versicolor
versicolor_count = 0
versicolor_data = [[],[],[],[]]


with open('train.csv') as train:

    reader = csv.reader(train)
    next(reader)

    for num,row in enumerate(reader):
        type = row[5]
        if type=="Iris-virginica":
            virginica_count+=1
            virginica_data[SL_INDEX].append(float(row[SL_INDEX+1]))
            virginica_data[SW_INDEX].append(float(row[SW_INDEX+1]))
            virginica_data[PL_INDEX].append(float(row[PL_INDEX+1]))
            virginica_data[PW_INDEX].append(float(row[PW_INDEX+1]))
        elif type=="Iris-setosa":
            setosa_count+=1
            setosa_data[SL_INDEX].append(float(row[SL_INDEX+1]))
            setosa_data[SW_INDEX].append(float(row[SW_INDEX+1]))
            setosa_data[PL_INDEX].append(float(row[PL_INDEX+1]))
            setosa_data[PW_INDEX].append(float(row[PW_INDEX+1]))
        elif type=="Iris-versicolor":
            versicolor_count+=1
            versicolor_data[SL_INDEX].append(float(row[SL_INDEX+1]))
            versicolor_data[SW_INDEX].append(float(row[SW_INDEX+1]))
            versicolor_data[PL_INDEX].append(float(row[PL_INDEX+1]))
            versicolor_data[PW_INDEX].append(float(row[PW_INDEX+1]))

total_count=virginica_count+setosa_count+versicolor_count

print(f'P[I=Virginica] = {virginica_count/total_count}')
print(f'\tSL Sample Mean: {np.mean(virginica_data[SL_INDEX])}')
print(f'\tSL Sample Variance: {np.var(virginica_data[SL_INDEX], ddof=1)}')
print(f'\tSW Sample Mean: {np.mean(virginica_data[SW_INDEX])}')
print(f'\tSW Sample Variance: {np.var(virginica_data[SW_INDEX], ddof=1)}')
print(f'\tPL Sample Mean: {np.mean(virginica_data[PL_INDEX])}')
print(f'\tPL Sample Variance: {np.var(virginica_data[PL_INDEX], ddof=1)}')
print(f'\tPW Sample Mean: {np.mean(virginica_data[PW_INDEX])}')
print(f'\tPW Sample Variance: {np.var(virginica_data[PW_INDEX], ddof=1)}')

print(f'P[I=Setosa] = {setosa_count/total_count}')
print(f'\tSL Sample Mean: {np.mean(setosa_data[SL_INDEX])}')
print(f'\tSL Sample Variance: {np.var(setosa_data[SL_INDEX], ddof=1)}')
print(f'\tSW Sample Mean: {np.mean(setosa_data[SW_INDEX])}')
print(f'\tSW Sample Variance: {np.var(setosa_data[SW_INDEX], ddof=1)}')
print(f'\tPL Sample Mean: {np.mean(setosa_data[PL_INDEX])}')
print(f'\tPL Sample Variance: {np.var(setosa_data[PL_INDEX], ddof=1)}')
print(f'\tPW Sample Mean: {np.mean(setosa_data[PW_INDEX])}')
print(f'\tPW Sample Variance: {np.var(setosa_data[PW_INDEX], ddof=1)}')

print(f'P[I=versicolor] = {versicolor_count/total_count}')
print(f'\tSL Sample Mean: {np.mean(versicolor_data[SL_INDEX])}')
print(f'\tSL Sample Variance: {np.var(versicolor_data[SL_INDEX], ddof=1)}')
print(f'\tSW Sample Mean: {np.mean(versicolor_data[SW_INDEX])}')
print(f'\tSW Sample Variance: {np.var(versicolor_data[SW_INDEX], ddof=1)}')
print(f'\tPL Sample Mean: {np.mean(versicolor_data[PL_INDEX])}')
print(f'\tPL Sample Variance: {np.var(versicolor_data[PL_INDEX], ddof=1)}')
print(f'\tPW Sample Mean: {np.mean(versicolor_data[PW_INDEX])}')
print(f'\tPW Sample Variance: {np.var(versicolor_data[PW_INDEX], ddof=1)}')


