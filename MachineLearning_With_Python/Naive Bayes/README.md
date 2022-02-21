# Naive Bayes
## Definition
* Classification technique based on Bays theorem.
* Assumes that presence of a perticular feature in a class is unrelated to the presence of any other feature.
* Document classification
<img src="https://github.com/iAmKankan/MachineLearning_With_Python/blob/master/Naive%20Bayes/nb.png?raw=true">


A **naive Bayes classifier** is an algorithm that uses Bayes' theorem to classify objects. Naive Bayes classifiers assume strong, or naive, independence between attributes of data points. Popular uses of naive Bayes classifiers include spam filters, text analysis and medical diagnosis. These classifiers are widely used for machine learning because they are simple to implement.

#### What is Bayes Theorem?
* Bayes theorem named after Rev. Thomas Bayes. 
* It works on conditional probability. 
* Conditional probability is the probability that something will happen, given that something else has already occurred. * Using the conditional probability, we can calculate the probability of an event using its prior knowledge.

#### The formula for calculating the conditional probability:

<img src="https://github.com/arijitBhadraMP/MachineLearningAndAI/blob/master/bayes1.png?raw=true">

#### Where:

* P(H) is the probability of hypothesis H being true. This is known as the prior probability.
* P(E) is the probability of the evidence(regardless of the hypothesis).
* P(E|H) is the probability of the evidence given that hypothesis is true.
* P(H|E) is the probability of the hypothesis given that the evidence is there.

#### Problem:

A Path Lab is performing a Test of disease say “D” with two results “Positive” & “Negative.” They guarantee that their test result is 99% accurate: if you have the disease, they will give test positive 99% of the time. If you don’t have the disease, they will test negative 99% of the time. If 3% of all the people have this disease and test gives “positive” result, what is the probability that you actually have the disease?

For solving the above problem, we will have to use conditional probability.
Probability of people suffering from Disease D, P(D) = 0.03 = 3%
Probability that test gives “positive” result and patient have the disease, P(Pos | D) = 0.99 =99%

Probability of people not suffering from Disease D, P(~D) = 0.97 = 97%
Probability that test gives “positive” result and patient does have the disease, P(Pos | ~D) = 0.01 =1%

For calculating the probability that the patient actually have the disease i.e, P( D | Pos) we will use Bayes theorem:


<img src="https://github.com/arijitBhadraMP/MachineLearningAndAI/blob/master/bayes2.png?raw=true">

 

We have all the values of numerator but we need to calculate P(Pos):         
P(Pos) = P(D, pos) + P( ~D, pos)        
= P(pos|D)*P(D) + P(pos|~D)*P(~D)            
= 0.99 * 0.03 + 0.01 * 0.97               
= 0.0297 + 0.0097            
= 0.0394               
         
Let’s calculate, P( D | Pos) = (P(Pos | D) * P(D)) / P(Pos)                  
= (0.99 * 0.03) / 0.0394              
= 0.753807107                    

So, Approximately 75% chances are there that the patient is actually suffering from disease.                

### Naive Bayes Classifier:

* Naive Bayes is a kind of classifier which uses the Bayes Theorem. 
* It predicts membership probabilities for each class such as the probability that given record or data point belongs to a particular class.  
* The class with the highest probability is considered as the most likely class.

---

The formula:
```
 P(A/B)= (P(B/A) * P(A))/P(B)
```

---



Types of Naive Bayes Classifiers:             
**
 1)  Gaussian Naive Bayes      
 2)  Bernoulli Naive Bayes     
 3)  MultiNomial Naive Bayes        
**
### 1)  Gaussian Naive Bayes      
Here, we assume that the features follow a normal distribution. Instead of discrete counts, we have continuous features (e.g., the popular Iris dataset where the features are sepal width, petal width, sepal length, petal length).

### 2)  Bernoulli Naive Bayes    
The binomial model is useful if your feature vectors are binary (i.e., 0s and 1s). One application would be text classification with a bag of words model where the 0s 1s are "word occurs in the document" and "word does not occur in the document"

### 3)  MultiNomial Naive Bayes 
The multinomial naive Bayes model is typically used for discrete counts. E.g., if we have a text classification problem, we can take the idea of bernoulli trials one step further and instead of "word occurs in the document" we have "count how often word occurs in the document", you can think of it as "number of times outcome number x_i is observed over the n trials"






