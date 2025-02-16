# CS461 HW1 Samples

Some code samples from Machine Learning Principles HW1, relating to the Central Limit Theorem, Naïve Bayes classification, and Data Whitening.

## P2: Central Limit Theorem

p2.py simulates a Galton board 1000 times with numpy.random, with user-inputted depth of M. The results are plotted with matplotlib.

<p align="center">
  <img src="https://raw.githubusercontent.com/KyleDouglasP/cs461-hw1/refs/heads/main/img/GaltonBoard.PNG?token=GHSAT0AAAAAAC7ATRU2TCDUOFVKBWILGZAKZ5RMRUA" />
</p>

The histograms of each result showcase how as M increases (the number of Bernoulli random variables summed), the distribution of the binomial random variable should approach a Gaussian distribution, as per the Central Limit Theorem.

## P3: Naïve Bayes Classification

Some data was sampled from Fisher's Iris data set and split into a training set (train.csv) and test set (test.csv), for the purpose of creating and evaluating a Naïve Bayes classifier.

### Training 

The classifier is trained in kyleperry-nb-train.py, where train.csv is read through to calculate the sample mean and variance of every feature conditioned on each classification.

### Evaluation

The actual classifier is in kyleperry-nb-cls.py, which is set to classify all of the Irises in test.csv, then evaluate itself for accuracy. This model was able to achieve 100% accuracy on the test set utilizing the sample statistics from training for Gaussian distributions

## P5: Data Whitening

p5.py performs data whitening on x.npz, a 3x100000 matrix. This is done via calculating a matrix A and vector b which performs a transformation W=AX+b. W is then the whitened data, where E[W] = 0 and COV[W,W]=I.

The results of this whitening were plotted in 3D with matplotlib, to showcase how the data was effectively centered and scaled to become spherical (unit variance).

<p align="center">
  <img src="https://raw.githubusercontent.com/KyleDouglasP/cs461-hw1/refs/heads/main/img/Whitening.PNG?token=GHSAT0AAAAAAC7ATRU2WZMAXKLHOVMTEZXAZ5RMT6Q" />
</p>
