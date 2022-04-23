# Coin probability estimation using EM algorithm
Estimating probability of two coins through Expectation Maximization (EM) algorithm from an unknown distribution (unknown coin probabilities)
- This was part of a research task
This is how the program flows:
1) First we get the data from through api calls to a website
2) We start with random values for weights, mean and covariance for each coin 
3) Then we run the EM algorithm 

The algorithm needs to be run several times to better probability prediction
