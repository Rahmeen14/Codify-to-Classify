# Codify-to-Classify
A simple nodejs app that uses various Classification Algorithms in Python to predict the admission status to a university based on 2 test scores.

It works upon past data from the university in a .csv format. The contents of the file are

1. Test1 Score
2. Test2 Score
3. Admission Status, 0 or 1 for a negative and positive result respectively


A sample .csv file has been provided in data folder for your kind perusal.

To achieve it, it uses 7 different [Classification Algorithms](http://dataaspirant.com/2016/09/24/classification-clustering-alogrithms/) and makes the final verdict by the results of a simple majority of 4:

These 7 algorithms are :


1. [LogisticRegression](https://en.wikipedia.org/wiki/Logistic_regression)
2. [KNeighbors](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
3. [SVC(linear)](https://en.wikipedia.org/wiki/Support_vector_machine)
4. [SVC(rbf)](https://en.wikipedia.org/wiki/Support_vector_machine)
5. [Gaussian](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)
6. [Decision Tree](http://www.saedsayad.com/decision_tree.htm) 
7. [Random Forest](https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm)

## Installation requirements :computer:
```
Node
Python 3.6 or higher
```

## To see it work:
```
clone this repository
cd into it
Change the path to Python as is in your machine
pip install -r requirements.txt
npm install
node app.js
```
