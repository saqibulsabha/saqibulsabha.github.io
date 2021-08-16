---
layout: post
title: Logistic Regression
published: true
---


Logistic regression is one of the most popular Machine Learning algorithms, which comes under the Supervised Learning technique. It is used for predicting the categorical dependent variable using a given set of independent variables.

Logistic regression predicts the output of a categorical dependent variable. Therefore the outcome must be a categorical or discrete value. It can be either Yes or No, 0 or 1, true or False, etc. but instead of giving the exact value as 0 and 1, it gives the probabilistic values which lie between 0 and 1.

Logistic Regression is much similar to the Linear Regression except that how they are used. Linear Regression is used for solving Regression problems, whereas Logistic regression is used for solving the classification problems
We can call a Logistic Regression a Linear Regression model but the Logistic Regression uses a more complex cost function, this cost function can be defined as the ‘Sigmoid function’ or also known as the ‘logistic function’ instead of a linear function.

The goal of logistic regression, as with any classifier, is to figure out some way to split the data to allow for an accurate prediction of a given observation's class using the information present in the features. (For instance, if we were examining the Iris flower dataset, our classifier would figure out some method to split the data based on the following: sepal length, sepal width, petal length, petal width.) In the case of a generic two-dimensional example, the split might look something like this.
![](https://res.cloudinary.com/saqibulsabha/image/upload/v1629091760/new1_fxtzxy.png)

## Decision Boundary
We expect our classifier to give us a set of outputs or classes based on probability when we pass the inputs through a prediction function and returns a probability score between 0 and 1.

For Example, We have 2 classes, let’s take them like cats and dogs(1 — dog , 0 — cats). We basically decide with a threshold value above which we classify values into Class 1 and of the value goes below the threshold then we classify it in Class 2.
![new3.png](https://res.cloudinary.com/saqibulsabha/image/upload/v1629091761/new3_s0elq1.png)

 
                                                     
                                                      
                                                      
As shown in the above graph we have chosen the threshold as 0.5, if the prediction function returned a value of 0.7 then we would classify this observation as Class 1(DOG). If our prediction returned a value of 0.2 then we would classify the observation as Class 2(CAT).

## Logistic function

Now, let's see the logistic function.

 ![l2.PNG](https://res.cloudinary.com/saqibulsabha/image/upload/v1629092237/log_mshmup.png)
 
 
![new4.png](https://res.cloudinary.com/saqibulsabha/image/upload/v1629091760/new4_hcnk41.png)

 
As you can see, this function is asymptotically bounded between 0 and 1. Further, for very positive inputs our output will be close to 1 and for very negative inputs our output will be close to 0. This will essentially allow us to translate the value we obtain from z into a prediction of the proper class. Inputs that are close to zero (and thus, near the decision boundary) signify that we don't have a confident prediction of 0 or 1 for the observation.
At the decision boundary z=0, the function g(z)=0.5. We'll use this value as a cut-off establishing the prediction criterion:

hθ(xi) ≥ 0.5→ypred =1
hθ(xi) < 0.5→ypred = 0
where ypred denotes the predicted class of an observation, xi, and hθ(x) represents the functional composition, hθ(x) = g(z(x)).
More concretely, we can write our model as:
 
   ![new5.png](https://res.cloudinary.com/saqibulsabha/image/upload/v1629091760/new5_ifwdza.png)

Thus, the output of this function represents a binary prediction for the input observation's class.
 ![new6.png](https://res.cloudinary.com/saqibulsabha/image/upload/v1629091760/new6_dotafr.png)


Another way to interpret this output is to view it in terms of a probabilistic prediction of the true class. In other words,  hθ(x) represents the estimated probability that yi=1 for a given input, xi.

                      hθ(xi)=P(yi=1|xi;θ)
                      
Because the class can only take on values of 0 or 1, we can also write this in terms of the probability that yi=0 for a given input, xi.

                    P(yi=0|xi;θ)=1−P(yi=1|xi;θ)
                    
For example, if hθ(x)=0.85 then we can assert that there is an 85% probability that yi=1 and a 15% probability that yi=0. This is useful as we can not only predict the class of an observation, but we can quantify the certainty of such prediction. In essence, the further a point is from the decision boundary, the more certain we are about the decision.

## The cost function

Next, we need to establish a cost function which can grade how well our model is performing according to the training data. This cost function, J(θ), can be considered to be a summation of individual "grades" for each classification prediction in our training set, comparing hθ(x) with the true class yi. We want the cost function to be large for incorrect classifications and small for correct ones so that we can minimize J(θ) to find the optimal parameters.

   ![l3.PNG](https://res.cloudinary.com/saqibulsabha/image/upload/v1629091759/l3_eob5w3.png)
        
                        
                        
In linear regression, we used the squared error as our grading mechanism. Unfortunately for logistic regression, such a cost function produces a nonconvex space that is not ideal for optimization. There will exist many local optima on which our optimization algorithm might prematurely converge before finding the true minimum.

Using the Maximum Likelihood Estimator from statistics, we can obtain the following cost function which produces a convex space friendly for optimization. This function is known as the binary cross-entropy loss.

   ![l4.PNG](https://res.cloudinary.com/saqibulsabha/image/upload/v1629091759/l4_tmetcu.png)
               

These cost functions return high costs for incorrect predictions.

   ![new7.png](https://res.cloudinary.com/saqibulsabha/image/upload/v1629091760/new7_xznyol.png)
              

More succinctly, we can write this as

   ![l5.PNG](https://res.cloudinary.com/saqibulsabha/image/upload/v1629091759/l5_pwghy9.png)
             

Since the second term will be zero when y=1and the first term will be zero when y=0. Substituting this cost into our overall cost function we obtain:

   ![l6.PNG](https://res.cloudinary.com/saqibulsabha/image/upload/v1629091759/l6_zfx6kk.png)

  
  
### Conclusion
Logistic regression is a powerful machine learning algorithm that utilizes a sigmoid function and works best on binary classification problems, although it can be used on multi-class classification problems through the “one vs. all” method. Logistic regression (despite its name) is not fit for regression tasks.


### References
1.	https://www.javatpoint.com/logistic-regression-in-machine-learning
2.	https://towardsdatascience.com/introduction-to-logistic-regression-66248243c148
3.	https://kambria.io/blog/logistic-regression-for-machine-learning/
4.	https://www.jeremyjordan.me/logistic-regression/






