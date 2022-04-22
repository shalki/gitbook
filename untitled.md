# ML: Multivariate Linear Regression

## Multiple Features

Linear regression with multiple variables is also known as "multivariate linear regression".

We now introduce notation for equations where we can have any number of input variables.

$$
\begin{align*}x_j^{(i)} &= \text{value of feature } j \text{ in the }i^{th}\text{ training example} \newline x^{(i)}& = \text{the input (features) of the }i^{th}\text{ training example} \newline m &= \text{the number of training examples} \newline n &= \text{the number of features} \end{align*}
$$

The multivariable form of the hypothesis function accommodating these multiple features is as follows:

$$h_\theta (x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \theta_3 x_3 + \cdots + \theta_n x_n$$&#x20;

In order to develop intuition about this function, we can think about $$\theta_0$$​ as the basic price of a house, $$\theta_1$$​ as the price per square meter, $$\theta_2$$as the price per floor, etc. $$x_1$$will be the number of square meters in the house, $$x_2$$the number of floors, etc.

Using the definition of matrix multiplication, our multivariable hypothesis function can be concisely represented as:

$$
\begin{align*}h_\theta(x) =\begin{bmatrix}\theta_0 \hspace{2em} \theta_1 \hspace{2em} ... \hspace{2em} \theta_n\end{bmatrix}\begin{bmatrix}x_0 \newline x_1 \newline \vdots \newline x_n\end{bmatrix}= \theta^T x\end{align*}
$$

This is a vectorization of our hypothesis function for one training example; see the lessons on vectorization to learn more.

**Remark**: Note that for convenience reasons in this course we assume $$x_{0}^{(i)} =1 \text{ for } (i\in { 1,\dots, m } )$$. This allows us to do matrix operations with theta and x. Hence making the two vectors '$$\theta$$' and $$x^{(i)}$$match each other element-wise (that is, have the same number of elements: n+1).]\


\[**Note**: So that we can do matrix operations with theta and x, we will set $$x^{(i)}_0 = 1$$, for all values of i. This makes the two vectors 'theta' and $$x_{(i)}$$ match each other element-wise (that is, have the same number of elements: n+1).]

The training examples are stored in X row-wise, like such:

$$
\begin{align*}X = \begin{bmatrix}x^{(1)}_0 & x^{(1)}_1  \newline x^{(2)}_0 & x^{(2)}_1  \newline x^{(3)}_0 & x^{(3)}_1 \end{bmatrix}&,\theta = \begin{bmatrix}\theta_0 \newline \theta_1 \newline\end{bmatrix}\end{align*}
$$

You can calculate the hypothesis as a column vector of size (m x 1) with:

$$h_\theta(X) = X \theta$$

&#x20;**For the rest of these notes, and other lecture notes, X will represent a matrix of training examples** $$x_{(i)}$$ **stored row-wise.**

### **Cost function**

For the parameter vector $$θ$$ (of type $$\mathbb{R}^{n+1}$$or in $$\mathbb{R}^{(n+1) \times 1}$$, the cost function is:

$$J(\theta) = \dfrac {1}{2m} \displaystyle \sum_{i=1}^m \left (h_\theta (x^{(i)}) - y^{(i)} \right)^2$$

The vectorized version is:

$$J(\theta) = \dfrac {1}{2m} (X\theta - \vec{y})^{T} (X\theta - \vec{y})$$

Where $$\vec{y}$$denotes the vector of all y values.

## Gradient Descent For Multiple Variables

The gradient descent equation itself is generally the same form; we just have to repeat it for our 'n' features:

$$
\begin{align*} & \text{repeat until convergence:} \; \lbrace \newline \; & \theta_0 := \theta_0 - \alpha \frac{1}{m} \sum\limits_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_0^{(i)}\newline \; & \theta_1 := \theta_1 - \alpha \frac{1}{m} \sum\limits_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_1^{(i)} \newline \; & \theta_2 := \theta_2 - \alpha \frac{1}{m} \sum\limits_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_2^{(i)} \newline & \cdots \newline \rbrace \end{align*}
$$

In other words:

$$
\begin{align*}& \text{repeat until convergence:} \; \lbrace \newline \; & \theta_j := \theta_j - \alpha \frac{1}{m} \sum\limits_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_j^{(i)} \; & \text{for j := 0...n}\newline \rbrace\end{align*}
$$

The following image compares gradient descent with one variable to gradient descent with multiple variables:

![](<.gitbook/assets/image (18).png>)



### Matrix Notation

The Gradient Descent rule can be expressed as:

$$\theta := \theta - \alpha \nabla J(\theta)$$

Where $$\nabla J(\theta)$$is a column vector of the form:

$$
\nabla J(\theta)  = \begin{bmatrix}\frac{\partial J(\theta)}{\partial \theta_0}   \\ \frac{\partial J(\theta)}{\partial \theta_1}   \\ \vdots   \\ \frac{\partial J(\theta)}{\partial \theta_n} \end{bmatrix}
$$

The j-th component of the gradient is the summation of the product of two terms:

$$
\begin{align*}
\; &\frac{\partial J(\theta)}{\partial \theta_j} &=&  \frac{1}{m} \sum\limits_{i=1}^{m}  \left(h_\theta(x^{(i)}) - y^{(i)} \right) \cdot x_j^{(i)} \newline
\; & &=& \frac{1}{m} \sum\limits_{i=1}^{m}   x_j^{(i)} \cdot \left(h_\theta(x^{(i)}) - y^{(i)}  \right) 
\end{align*}
$$

Sometimes, the summation of the product of two terms can be expressed as the product of two vectors.

Here, $$x_j^{(i)}$$​, for i = 1,...,m, represents the m elements of the j-th column, $$\vec{x_j}$$, of the training set X.

The other term $$\left(h_\theta(x^{(i)}) - y^{(i)} \right)$$is the vector of the deviations between the predictions $$h_\theta(x^{(i)})$$ and the true values $$y^{(i)}$$. Re-writing $$\frac{\partial J(\theta)}{\partial \theta_j}$$​, we have:

$$
\begin{align*}\; &\frac{\partial J(\theta)}{\partial \theta_j} &=& \frac1m  \vec{x_j}^{T} (X\theta - \vec{y}) \newline\newline\newline\; &\nabla J(\theta) & = & \frac 1m X^{T} (X\theta - \vec{y}) \newline\end{align*}
$$

Finally, the matrix notation (vectorized) of the Gradient Descent rule is:

$$
\theta := \theta - \frac{\alpha}{m} X^{T} (X\theta - \vec{y})
$$

## Gradient Descent in Practice I - Feature Scaling

We can speed up gradient descent by having each of our input values in roughly the same range. This is because $$\theta$$ will descend quickly on small ranges and slowly on large ranges, and so will oscillate inefficiently down to the optimum when the variables are very uneven.

The way to prevent this is to modify the ranges of our input variables so that they are all roughly the same. Ideally:

$$−1 ≤ x_{(i)}​ ≤ 1$$

or

$$−0.5 ≤ x_{(i)}​ ≤ 0.5$$

These aren't exact requirements; we are only trying to speed things up. The goal is to get all input variables into roughly one of these ranges, give or take a few.

Two techniques to help with this are **feature scaling** and **mean normalization**. Feature scaling involves dividing the input values by the range (i.e. the maximum value minus the minimum value) of the input variable, resulting in a new range of just 1. Mean normalization involves subtracting the average value for an input variable from the values for that input variable resulting in a new average value for the input variable of just zero. To implement both of these techniques, adjust your input values as shown in this formula:

$$x_i := \dfrac{x_i - \mu_i}{s_i}$$​​

Where $$μ_i$$ is the **average** of all the values for feature (i) and $$s_i$$is the range of values (max - min), or $$s_i$$is the standard deviation.

Note that dividing by the range, or dividing by the standard deviation, give different results. The quizzes in this course use range - the programming exercises use standard deviation.

For example, if $$x_i$$​ represents housing prices with a range of 100 to 2000 and a mean value of 1000, then, $$x_i := \dfrac{price-1000}{1900}$$.

## Gradient Descent in Practice II - Learning Rate

**Debugging gradient descent.** Make a plot with _number of iterations_ on the x-axis. Now plot the cost function, $$J(θ)$$ over the number of iterations of gradient descent. If $$J(θ)$$ ever increases, then you probably need to decrease α.

**Automatic convergence test.** Declare convergence if $$J(θ)$$ decreases by less than E in one iteration, where E is some small value such as $$10^{−3}$$. However in practice it's difficult to choose this threshold value.

![](<.gitbook/assets/image (19).png>)

It has been proven that if learning rate $$α$$ is sufficiently small, then $$J(θ)$$ will decrease on every iteration.

![](<.gitbook/assets/image (20).png>)

To summarize:

If $$\alpha$$ is too small: slow convergence.

If $$\alpha$$ is too large: ￼may not decrease on every iteration and thus may not converge.

## Features and Polynomial Regression

We can improve our features and the form of our hypothesis function in a couple different ways.

We can **combine** multiple features into one. For example, we can combine $$x_1$$​ and $$x_2$$​ into a new feature $$x_3$$by taking $$x_1$$⋅$$x_2$$​.

#### **Polynomial Regression**

Our hypothesis function need not be linear (a straight line) if that does not fit the data well.

We can **change the behavior or curve** of our hypothesis function by making it a quadratic, cubic or square root function (or any other form).

For example, if our hypothesis function is $$h_\theta(x) = \theta_0 + \theta_1 x_1$$ then we can create additional features based on $$x_1$$, to get the quadratic function $$h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_1^2$$or the cubic function $$h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_1^2 + \theta_3 x_1^3$$​

In the cubic version, we have created new features $$x_2$$and $$x_3$$where $$x_2 = x_1^2$$and $$x_3 = x_1^3$$​.

To make it a square root function, we could do: $$h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 \sqrt{x_1}$$​​

One important thing to keep in mind is, if you choose your features this way then feature scaling becomes very important.

eg. if $$x_1$$ has range 1 - 1000 then range of $$x_1^2$$becomes 1 - 1000000 and that of $$x_1^3$$becomes 1 - 1000000000
