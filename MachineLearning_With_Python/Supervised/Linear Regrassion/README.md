### Index
* [Definition Of Linear Regression](#definition)
* [What is Regression](#what-is-regression)
* [Assumptions of Linear Regression](#assumptions-of-linear-regression)
  * [Multicollinearity](https://github.com/iAmKankan/MachineLearning_With_Python/blob/master/Supervised/Linear%20Regrassion/correlation.md)
  * [Linear and Additive](#linear-and-additive)
  * [Heteroscedasticity](#heteroscedasticity)
  * [Autocorrelation](#autocorrelation)
  * [Normal Distribution of error terms](#normal-distribution-of-error-terms)
* [Types of Linear Regression](#linear-regression-is-of-2-types)
  * [Simple Linear Regression](#simple-linear-regression)
  * [Multivariate Linear Regression](#multivariate-linear-regression)
* [Regularized Linear Models](https://github.com/iAmKankan/Regularization/blob/master/linear_regularization.md) 
  * [**_Ridge Regression_**](https://github.com/iAmKankan/Regularization/blob/master/linear_regularization.md)
  * [**_Lasso Regression_**](https://github.com/iAmKankan/Regularization/blob/master/linear_regularization.md)
  * [**_Elastic Net_**](https://github.com/iAmKankan/Regularization/blob/master/linear_regularization.md)
* [Error](#error-or-loss-or-cost)
  * [Residual Sum Of Squres(RSS)](#residual-sum-of-squres)
  * [Total Squired Error](#total-squired-error)
  * [Total varience in Y](#total-varience-in-y)
  * [ R Squared Error ( _**R <sup>2**_ ) or Coefficient of Determination](#r-squared-error-or-coefficient-of-determination)
  * [Adjusted R Squared Error _**R <sup>2**_](#adjusted-r-squared-error)

# Linear Regression
![grape](https://user-images.githubusercontent.com/12748752/126882595-d1f5449e-14bb-4ab3-809c-292caf0858a1.png)
## Definition
![plum](https://user-images.githubusercontent.com/12748752/126882596-b9ba4645-7001-435e-9a3c-d4416a2543c1.png)
* **Linear Regression tends to establish a relationship between a dependent variable(Y) and one or more independent variable(X) by finding the best fit of the straight line.**

<img src="https://upload.wikimedia.org/wikipedia/commons/3/3a/Linear_regression.svg" align=right width=30% />

* The equation for the Linear model is _**Y = mX+c**_, where _**m**_ is the slope and _**c**_ is the intercept
* Linear Regression When we want to predict real continuous values as an output
* There is no straight line that runs through all the data points. 
* So, the objective here is to fit the best fit of a straight line that will try to minimize the error between the expected and actual value.




### What is Regression
![plum](https://user-images.githubusercontent.com/12748752/126882596-b9ba4645-7001-435e-9a3c-d4416a2543c1.png)
* [**Please follow the link**](https://github.com/iAmKankan/Statistics/blob/main/commonTerms.md#regression)

## Assumptions of Linear Regression
![grape](https://user-images.githubusercontent.com/12748752/126882595-d1f5449e-14bb-4ab3-809c-292caf0858a1.png)
* Let’s look at the important assumptions in regression analysis-
  * There should be no correlation between the residual (error) terms. Absence of this phenomenon is known as **Autocorrelation**.
  * The independent variables should not be correlated. Absence of this phenomenon is known as [**Multicollinearity**](https://github.com/iAmKankan/MachineLearning_With_Python/blob/master/Supervised/Linear%20Regrassion/correlation.md).
  * The error terms must have constant variance. This phenomenon is known as **Homoskedasticity**. The presence of non-constant variance is referred to **Heteroskedasticity**.
  * **The error terms must be normally distributed**.
  * There should be a **linear and additive relationship between dependent (response) variable and independent (predictor) variable(s)**. 
     * A linear relationship suggests that a change in response _Y_ due to one unit change in _X<sup>1_ is constant, regardless of the value of _X<sup>1_. 
     * An additive relationship suggests that the effect of _X<sup>1_ on _Y_ is independent of other variables.
## What if these assumptions get violated
![grape](https://user-images.githubusercontent.com/12748752/126882595-d1f5449e-14bb-4ab3-809c-292caf0858a1.png)
### Linear and Additive
![plum](https://user-images.githubusercontent.com/12748752/126882596-b9ba4645-7001-435e-9a3c-d4416a2543c1.png)

* If you fit a linear model to a non-linear, non-additive data set, the regression algorithm would fail to capture the trend mathematically, thus resulting in an inefficient model. Also, this will result in erroneous predictions on an unseen data set.

#### How to check
* Look for residual vs fitted value plots.
*  Also, you can include polynomial terms (_X_, _X <sup>2_, _X <sup>3_ ) in your model to capture the non-linear effect.

### Multicollinearity
![plum](https://user-images.githubusercontent.com/12748752/126882596-b9ba4645-7001-435e-9a3c-d4416a2543c1.png)
* [Please follow the link](https://github.com/iAmKankan/MachineLearning_With_Python/blob/master/Supervised/Linear%20Regrassion/correlation.md)
 ### Heteroscedasticity 
 ![plum](https://user-images.githubusercontent.com/12748752/126882596-b9ba4645-7001-435e-9a3c-d4416a2543c1.png)

 * The presence of non-constant variance in the error terms results in heteroskedasticity.
 * Generally, non-constant variance arises in presence of outliers or extreme leverage values.
 * Look like, these values get too much weight, thereby disproportionately influences the model’s performance. 
 * When this phenomenon occurs, the confidence interval for out of sample prediction tends to be unrealistically wide or narrow.

#### How to check
* You can look at residual vs fitted values plot.
*  If heteroskedasticity exists, the plot would exhibit a funnel shape pattern (shown in next section). 
*  Also, you can use Breusch-Pagan / Cook – Weisberg test or White general test to detect this phenomenon.
 ### Autocorrelation 
 ![plum](https://user-images.githubusercontent.com/12748752/126882596-b9ba4645-7001-435e-9a3c-d4416a2543c1.png)

 * It's refers to a perticular data where the error term of one time period is correlated to the error term of anothether time period. 
 * This is very common to time series data analysis.
 #### How to check
 * Durbin – Watson (DW) statistic. 
    *  It must lie between 0 and 4.
    *  If DW = 2, implies no autocorrelation, 
    *  0 < DW < 2 implies positive autocorrelation 
    *  2 < DW < 4 indicates negative autocorrelation
 *  Also, you can see residual vs time plot and look for the seasonal or correlated pattern in residual values.
  
### Normal Distribution of error terms
* If the error terms are non- normally distributed, confidence intervals may become too wide or narrow.
* Once confidence interval becomes unstable, it leads to difficulty in estimating coefficients based on minimization of least squares.
* Presence of non – normal distribution suggests that there are a few unusual data points which must be studied closely to make a better model.

#### How to check: 
* You can look at QQ plot (shown below). 
* You can also perform statistical tests of normality such as Kolmogorov-Smirnov test, Shapiro-Wilk test.


## Linear Regression is of 2 types
![plum](https://user-images.githubusercontent.com/12748752/126882596-b9ba4645-7001-435e-9a3c-d4416a2543c1.png)

  * **Simple Linear Regression** 
  * **Multivariate Linear Regression**
    
     
### Simple Linear Regression
![plum](https://user-images.githubusercontent.com/12748752/126882596-b9ba4645-7001-435e-9a3c-d4416a2543c1.png)

> <img src="https://latex.codecogs.com/svg.image?Y=a&plus;bX" title="Y=a+bX" />

* It contains only one input variable. Only one straight line.

    * Y = Dependent Variable
    * a = Y intercept
    * b = slope of the line
    * X = Independent variable
* Y intercept(a): is the value of the Dependent variable(y) when the value of the indipendent variable is zero(0). This is the point at which the line cuts the y-axis.

* Slope(b): is the change in the Dependent Variable(y) for a unit increase in the indipendent variable. It is the tangent of  the angle made by the line with the x-axis.

<img src="https://latex.codecogs.com/svg.image?b=\frac{y_2-y_1}{x_2-x_1}" title="b=\frac{y_2-y_1}{x_2-x_1}" />

### Multivariate Linear Regression
![plum](https://user-images.githubusercontent.com/12748752/126882596-b9ba4645-7001-435e-9a3c-d4416a2543c1.png)

> <img src="https://latex.codecogs.com/svg.image?Y=\beta&space;_0&plus;\beta_1x_1&plus;\beta_2x_2&plus;\beta_3x_3&plus;...&plus;\beta_nx_n" title="Y=\beta _0+\beta_1x_1+\beta_2x_2+\beta_3x_3+...+\beta_nx_n" />

* Here we need to find out all the <img src="https://latex.codecogs.com/svg.image?\inline&space;\beta" title="\inline \beta" /> coefficients.
* It means the beta is creating some kind of relationship or slope it is creating with respect to the output variable
* Findout the relationshio between beta and the output variables..


![linear](https://user-images.githubusercontent.com/12748752/127488303-78ce74fd-a216-4e55-953d-e97f4ec9798a.png)

## Error or Loss or Cost
![grape](https://user-images.githubusercontent.com/12748752/126882595-d1f5449e-14bb-4ab3-809c-292caf0858a1.png)
### Residual Sum Of Squres
![plum](https://user-images.githubusercontent.com/12748752/126882596-b9ba4645-7001-435e-9a3c-d4416a2543c1.png)

* The aim is to minimize the error which is the distance from the bestfit line to each observation.
* The Deviation or the Error is like-
* Deviation is called Residual.

> <img src="https://latex.codecogs.com/svg.image?Residual&space;=&space;Y_p-Y_a" title="Residual = Y_p-Y_a" />

* **The mail goal is to reduce the error**.

<img src="https://latex.codecogs.com/svg.image?AllResidual&space;=&space;\sum&space;(Y_p-Y_a)" title="AllResidual = \sum (Y_p-Y_a)" />

* -ve Y actuals also there thats why we take the squre of the whole residuals

> <img src="https://latex.codecogs.com/svg.image?ResidualSumOfSqures&space;=&space;\sum&space;&space;(Y_p-Y_a)^2" title="ResidualSumOfSqures = \sum (Y_p-Y_a)^2" />

* **Residual Sum Of Squres(RSS) = Ordinary least squares(OLS) = Error function = Loss Function.**

* Now we have to take the minimum of all errors. That's will give us the beest fit line.

* Inorder to fing out the minimum of a function we need to use first order derivation and second order derivation to find out the minima and maxima.

> <img src="https://latex.codecogs.com/svg.image?ResidualSumOfSqures&space;=&space;\left&space;(min&space;&space;\sum&space;&space;(Y_p-Y_a)^2\right&space;)" title="ResidualSumOfSqures = \left (min \sum (Y_p-Y_a)^2\right )" />

* **Best-fit-line we need to find out when we findout the 'a' and the 'b' that's the line which is the min error.**
---

> <img src="https://latex.codecogs.com/svg.image?ResidualSumOfSqures&space;=&space;\left&space;(min&space;&space;\sum&space;&space;(Y_p-Y_a)^2\right&space;)" title="ResidualSumOfSqures = \left (min \sum (Y_p-Y_a)^2\right )" />






---
* **This function is parabola.or Convex shaped curve**

* if we replace the above function with this <img src="https://latex.codecogs.com/svg.image?Y=X^2" title="Y=X^2" />
    * for X= -1,-2,0,1,2 result will be 1,4,0,1,4 if we ploted over a chart it will be like- 
![Linear2](https://user-images.githubusercontent.com/12748752/127488310-b20c91b5-d450-47ad-9e2b-d490ebc07233.png)





* Inorder to findout the minimum error we use gradient descent

https://www.mladdict.com/linear-regression-simulator
* The learning rate is represented by alpha <img src="https://latex.codecogs.com/svg.image?\alpha" title="\alpha" />

### Total Squired Error
![plum](https://user-images.githubusercontent.com/12748752/126882596-b9ba4645-7001-435e-9a3c-d4416a2543c1.png)

> <img src="https://latex.codecogs.com/svg.image?SE_{Line}&space;=(Y_1-(bX_1&plus;a))^2&plus;(Y_2-(bX_2&plus;a))^2&plus;...&plus;(Y_n-(bX_n&plus;a))^2" title="SE_{Line} =(Y_1-(bX_1+a))^2+(Y_2-(bX_2+a))^2+...+(Y_n-(bX_n+a))^2" />




### Total varience in Y
![plum](https://user-images.githubusercontent.com/12748752/126882596-b9ba4645-7001-435e-9a3c-d4416a2543c1.png)

* Is the Squared Error of the mean, The mean of all Y is represented by <img src="https://latex.codecogs.com/svg.image?\overline{Y}" title="\overline{Y}" />

> <img src="https://latex.codecogs.com/svg.image?SE_{\overline{Y}}&space;=(Y_1-\overline{Y})^2&plus;(Y_2-\overline{Y})^2&plus;...&plus;(Y_n-\overline{Y})^2" title="SE_{\overline{Y}} =(Y_1-\overline{Y})^2+(Y_2-\overline{Y})^2+...+(Y_n-\overline{Y})^2" />
 


#### What % of the variation is NOT Described by the vasriation in X or by the Regression line = <img src="https://latex.codecogs.com/svg.image?\frac{SE_{Line}&space;}{SE_{\overline{Y}}}" title="\frac{SE_{Line} }{SE_{\overline{Y}}}" />
![plum](https://user-images.githubusercontent.com/12748752/126882596-b9ba4645-7001-435e-9a3c-d4416a2543c1.png)

### R Squared Error or Coefficient of Determination
![plum](https://user-images.githubusercontent.com/12748752/126882596-b9ba4645-7001-435e-9a3c-d4416a2543c1.png)

* **Coefficient of Determination = What % of the variation is acctually Described by the vasriation in X or by the Regression line =** <img src="https://latex.codecogs.com/svg.image?1-\frac{SE_{Line}&space;}{SE_{\overline{Y}}}" title="1-\frac{SE_{Line} }{SE_{\overline{Y}}}" />


* If <img src="https://latex.codecogs.com/svg.image?SE_{Line}" title="SE_{Line}" /> is very small the we can say that the regression line is good fit.
* the the <img src="https://latex.codecogs.com/svg.image?R^2" title="R^2" /> is close to 1.
* If <img src="https://latex.codecogs.com/svg.image?SE_{Line}" title="SE_{Line}" />"  is large then  the <img src="https://latex.codecogs.com/svg.image?R^2" title="R^2" /> is close to 0.


![error](https://user-images.githubusercontent.com/12748752/127488864-8d0310bd-02a5-4bab-8ea6-5140917eca64.png)

### Adjusted R Squared Error
![plum](https://user-images.githubusercontent.com/12748752/126882596-b9ba4645-7001-435e-9a3c-d4416a2543c1.png)

* It is possible to get <img src="https://latex.codecogs.com/svg.image?R^2" title="R^2" /> value < 0(-ve)
   * When the the regression line is not better than the average line or Total varience in Y
* If we add independent features the value of R^2 is improves by decresing the value of <img src="https://latex.codecogs.com/svg.image?SE_{Line}" title="SE_{Line}" />. But the features may not to correlated  to the output variable Y.
* So, it is not penalising the new added features.
* In  Adjusted R Square it penalizes attributes that are not correlated with Y.
![adjustedr](https://user-images.githubusercontent.com/12748752/127488214-c98d545a-fad0-4f0e-8176-786836be572d.png)
