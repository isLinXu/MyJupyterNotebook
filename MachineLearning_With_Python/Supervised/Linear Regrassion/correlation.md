## Index
![grape](https://user-images.githubusercontent.com/12748752/126882595-d1f5449e-14bb-4ab3-809c-292caf0858a1.png)
https://statisticsbyjim.com/regression/multicollinearity-in-regression-analysis/
### Multicollinearity
![grape](https://user-images.githubusercontent.com/12748752/126882595-d1f5449e-14bb-4ab3-809c-292caf0858a1.png)
> #### Multicollinearity occurs when independent variables in a regression model are correlated. This correlation is a problem because independent variables should be independent. 
> #### If the degree of correlation between variables is high enough, it can cause problems when you fit the model and interpret the results.


* There are four steps to remove multicollinearity-
  * **Increase the sample size**- By doin g this it might reduce the high colinearity of those two variables.
  * **Use priori information**- We can look into the previous data to make the conclution
  * **Transform the function relationship with dependent variable**
  * **Drop the variable**
 #### How to check
 ![plum](https://user-images.githubusercontent.com/12748752/126882596-b9ba4645-7001-435e-9a3c-d4416a2543c1.png)

 * You can use scatter plot to visualize correlation effect among variables. 
 *  Generally 'dataframe.corr()' shows the colinearity between variables. Where **-0.4 to 0.4** is accectable range for colinearity between variables.
 * Also, you can also use VIF factor. 
   * VIF value <= 4 suggests no multicollinearity whereas a value of >= 10 implies serious multicollinearity.


###  Colinearity
![grape](https://user-images.githubusercontent.com/12748752/126882595-d1f5449e-14bb-4ab3-809c-292caf0858a1.png)

>  <img src="https://latex.codecogs.com/svg.image?Y&space;=\beta_0&plus;\beta_1&space;X_1&plus;\beta_2&space;X_2&plus;\epsilon" title="Y =\beta_0+\beta_1 X_1+\beta_2 X_2+\epsilon" />


* Two independent variables <img src="https://latex.codecogs.com/svg.image?X_1,X_2" title="X_1,X_2" />  are colinear when they are correlated with each other
* In a multiple regression study, we assume theat the X variables are independent to each other.
* We also assube that each X-variable has a unique piece of information about Y.
* We belive that 
  * >  <img src="https://latex.codecogs.com/svg.image?\beta_1=" title="\beta_1=" /> The change of Y for a 1-unit change in <img src="https://latex.codecogs.com/svg.image?X_1" title="X_1" /> , while <img src="https://latex.codecogs.com/svg.image?X_2" title="X_2" /> is held constant
  * > <img src="https://latex.codecogs.com/svg.image?\beta_2=" title="\beta_2=" /> The change of Y for a 1-unit change in <img src="https://latex.codecogs.com/svg.image?X_2" title="X_2" /> , while <img src="https://latex.codecogs.com/svg.image?X_1" title="X_1" /> is held constant

* In the condition when Multicorrelation exists then above two believes are oppositive or negetive.

### Effects of Multicollinearity
![plum](https://user-images.githubusercontent.com/12748752/126882596-b9ba4645-7001-435e-9a3c-d4416a2543c1.png)

> **t statistic for**      <img src="https://latex.codecogs.com/svg.image?\beta_1=\frac{b_1}{StdErrof&space;b_1}" title="\beta_1=\frac{b_1}{StdErrof b_1}" />

1. Variances (and standard errors) of regression coefficient estimators (i.e. the b.) are inflated. This means that Var(b) is too large.
2. The magnitude of the <img src="https://latex.codecogs.com/svg.image?b_1" title="b_1" />, may be different from what we expect.
3. The signs of the <img src="https://latex.codecogs.com/svg.image?b_1" title="b_1" />; may be the opposite of what we expect.
4. Adding or removing any of the X-variables produces large changes in the values of remaining <img src="https://latex.codecogs.com/svg.image?b_1" title="b_1" />, or their signs.
5. Sometimes removing a data point causes large changes in the value of the <img src="https://latex.codecogs.com/svg.image?b_1" title="b_1" />, or their signs.
6. In some cases, F is significant, but the t-values (for the <img src="https://latex.codecogs.com/svg.image?b_1" title="b_1" />) may not be significant.



### Detection
![plum](https://user-images.githubusercontent.com/12748752/126882596-b9ba4645-7001-435e-9a3c-d4416a2543c1.png)

* There are solutions where programmetically detection of colinearity-
   - Bivariate correlation
   - Variance Inflation Factor (VIF)


### Tests for Multicollinearity
![plum](https://user-images.githubusercontent.com/12748752/126882596-b9ba4645-7001-435e-9a3c-d4416a2543c1.png)

:large_blue_diamond: 1. **Calculate the correlation coefficient (r) for each pair of the X-variables.** If any of the r-values is significantly different from zero, then the independent variables involved may be collinear.

> <img src="https://latex.codecogs.com/svg.image?r=\frac{Coverrience&space;X_iX_j}{\sigma_i\sigma_j}" title="r=\frac{Coverrience X_iX_j}{\sigma_i\sigma_j}" />

 * Caveat: Although the r for any two X variables may be too small, three independent variables, X₁, X₂, and X3, may be highly correlated as a group.


* The result falls between **-1 to 1**.
* The numbers fall between **-0.4 to 0.4** cannot be consider as a co-related.
![corr](https://cio-wiki.org/images/2/22/Pearson_Correlation_Coefficent.png)

:large_blue_diamond: 2.  **Check whether the Variance Inflation Factor (VIF) is too high**
 * Mathematically, the VIF for a regression model variable is equal to the ratio of the overall model variance to the variance of a model that includes only that single independent variable. 
 *  This ratio is calculated for each independent variable. 
 * A high VIF indicates that the associated independent variable is highly collinear with the other variables in the model.
 
 * Rule of thumb: Collinearity exists if VIF> 5. A VIF of 10, for example, means Var(b) is 10 times what it should be if no collinearity existed (if no collinearity, VIF should be 1)
 * VIF is a more rigorous check for collinearity than correlation coefficient.

<img src="https://latex.codecogs.com/svg.image?VIF&space;=&space;\frac{1}{1-R^2_i}" title="VIF = \frac{1}{1-R^2_i}" />
VIF =

In the regression model:

<img src="https://latex.codecogs.com/svg.image?Y&space;=\beta_0&plus;\beta_1&space;X_1&plus;\beta_2&space;X_2&plus;\epsilon" title="Y =\beta_0+\beta_1 X_1+\beta_2 X_2+\epsilon" />

R₁2 is obtained from regressing X_1, on X_2, and X_3, as follows: 
<img src="https://latex.codecogs.com/svg.image?X_1=\alpha_0&plus;\alpha_1X_2&plus;\alpha_2X_2&plus;\epsilon" title="X_1=\alpha_0+\alpha_1X_2+\alpha_2X_2+\epsilon" />

Similarly:
* <img src="https://latex.codecogs.com/svg.image?X_2=\alpha_0&plus;\alpha_1X_2&plus;\alpha_2X_3&plus;\epsilon&space;=>X^2_2" title="X_2=\alpha_0+\alpha_1X_2+\alpha_2X_3+\epsilon =>X^2_2" />

* <img src="https://latex.codecogs.com/svg.image?X_3=\alpha_0&plus;\alpha_1X_2&plus;\alpha_2X_2&plus;\epsilon&space;=>X^2_3" title="X_3=\alpha_0+\alpha_1X_2+\alpha_2X_2+\epsilon =>X^2_3" />

### :black_circle: Bibilography
![plum](https://user-images.githubusercontent.com/12748752/126882596-b9ba4645-7001-435e-9a3c-d4416a2543c1.png)

* https://en.wikipedia.org/wiki/Correlation_and_dependence
* https://www.youtube.com/watch?v=pZhm1GMn2GY&t=159s
* https://statisticsbyjim.com/regression/multicollinearity-in-regression-analysis/
