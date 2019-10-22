## Intoduction to Naive Bayes for Classification



This blog post is about understanding Naive Bayes algorithm for classification tasks.

The tutorial is divided into the following parts.

1. Introduction
2. Conditional Probability
3. Bayes Theorem
4. Naive Bayes Sample Code
5. Laplace Correction
6. Advantages and Disadvantages
7. Applications of Naive Bayes Algorithm
8. Tips to improve the power of Naive Bayes Model
9. References


### 1. Introduction

Naive Bayes classiffier is a probablistic machine learning model that’s used for classification tasks. Assume that we have a situation that we have made features out of a given dataset and the dataset is very large dataset we need a to create some basic machine learning model for this dataset and use it as baseline model to compare benchmarks with other classification models, in this case the best option is to use Naive Bayes classifier. Its extremely fast compared to any other model available and has an training time complexity of *O(Nd)* where N is the number of training examples and d is the dimensionality of the dataset or number of input features. 

So why the algorithm is called **Naive**?

Naive Bayes algorithm makes an assumption that the input features of the datset that is used for model are independent of each other.

### 2. Conditional Probability

Conditional probability is defined as the likelihood of an event or outcome occurring, based on the occurrence of a previous event or outcome. For example the conditional probability of an event B is the probability that the event will occur given the knowledge that an event A has already occurred and it is denoted by P(B|A) probablity of B given A. 

\begin{equation}
P(B|A) = \frac{P(A \cap B)}{P(A)}
\end{equation}

**Example**

Lets take a look at the below table.

|           | Male | Female | Total |
|-----------|------|--------|-------|
| Teachers  | 10   | 15     | 25    |
| Students  | 110  | 90     | 200   |
| Total     | 120  | 105    | 225   |

In the above table we have 10 male teachers and 15 female teachers and 100 male students and 90 female students. Suppose we need find probality of member being a teacher given that the member is male.


\begin{equation}
P(Teacher|Male) = \frac{P(Male \cap Teacher)}{P(Male)} = \frac{10} {120} = 0.0833
\end{equation}


### 3. Bayes Theorem

The Bayes theorem describes the probability of an event based on the prior knowledge of the conditions that might be related to the event. In simple way suppose P(A) is the probablity of event A happens and P(B) is the probablity of event B happens suppose if we know the conditional probability P(B|A), then we can use the bayes rule to find out the P(A|B) .

Conditional probablity of B given A is

\begin{equation}
P(B|A) = \frac {P(A \cap B)}{P(A)}
\end{equation}

\begin{equation}
P(A|B) = \frac {P(A \cap B)}{P(B)}
\end{equation}

\begin{equation}
P(A \cap B) = P(A|B) * P(B) = P(B|A) * P(A)
\end{equation}


\begin{equation}
P(B|A) = P(A|B) * \frac {P(B)}{P(A)}
\end{equation}


**Example**

Jhon says he is itchy. There is a test for Allergy to humans, but this test is not always right

For people that really do have the allergy, the test says "Yes" 80% of the time.

For people that do not have the allergy, the test says "Yes" 10% of the time ("false positive").

If 1% of the population have the allergy, and Jhon's test says "Yes", what are the chances that Jhon really has the allergy?

<b>Given</b>

P(Allergy) is Probability of Allergy = 1%

P(Yes|Allergy) is probability of test saying "Yes" for people with allergy = 0.8 (80%)

1% have the allergy, and the test says "Yes" to 80% of them

99% do not have the allergy and the test says "Yes" to 10% of them

Therefore we can find P(Yes) as P(Yes) = 1% × 80% + 99% × 10% = 10.7%

<b>What do we need to find?</b>

We need to find probablity of Jhon will have allergy given that the test is "Yes".

\begin{equation}
P(Allergy|Yes) = P(Yes|Allergy) * \frac {P(Allergy)}{P(Yes)}
\end{equation}


\begin{equation}
P(Allergy|Yes) = \frac{0.01 * 0.8 }{10.7} = 0.0748
\end{equation}

### 4. Naive Bayes Sample Code
