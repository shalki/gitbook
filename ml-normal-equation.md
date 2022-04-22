# ML: Normal Equation (Optional)

## Normal Equation

The "Normal Equation" is a method of finding the optimum theta **without iteration.**

$$\theta = (X^T X)^{-1}X^T$$

There is **no need** to do feature scaling with the normal equation.

Mathematical proof of the Normal equation requires knowledge of linear algebra and is fairly involved, so you do not need to worry about the details.

Proofs are available at these links for those who are interested:

[https://en.wikipedia.org/wiki/Linear\_least\_squares\_(mathematics)](https://en.wikipedia.org/wiki/Linear\_least\_squares\_\(mathematics\))

[http://eli.thegreenplace.net/2014/derivation-of-the-normal-equation-for-linear-regression](http://eli.thegreenplace.net/2014/derivation-of-the-normal-equation-for-linear-regression)

The following is a comparison of gradient descent and the normal equation:

| Gradient Descent           | Normal Equation                                    |
| -------------------------- | -------------------------------------------------- |
| Need to choose alpha       | No need to choose alpha                            |
| Needs many iterations      | No need to iterate                                 |
| O ($$kn^2$$)               | O ($$n^3$$), need to calculate inverse of $$X^TX$$ |
| Works well when n is large | Slow if n is very large                            |

With the normal equation, computing the inversion has complexity $$\mathcal{O}(n^3)$$. So if we have a very large number of features, the normal equation will be slow. In practice, when n exceeds 10,000 it might be a good time to go from a normal solution to an iterative process.

## **Normal Equation Non-invertibility**

$$X^TX$$may be **non-invertible**. The common causes are:

* Redundant features, where two features are very closely related (i.e. they are linearly dependent)
* Too many features (e.g. m ≤ n). In this case, delete some features or use "regularization" (to be explained in a later lesson).

Solutions to the above problems include deleting a feature that is linearly dependent with another or deleting one or more features when there are too many features.
