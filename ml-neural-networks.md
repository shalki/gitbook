# ML: Neural Networks

## Non-linear Hypotheses

Performing linear regression with a complex set of data with many features is very unwieldy. Say you wanted to create a hypothesis from three (3) features that included all the quadratic terms:

$$
\begin{align*}& g(\theta_0 + \theta_1x_1^2 + \theta_2x_1x_2 + \theta_3x_1x_3 \newline& + \theta_4x_2^2 + \theta_5x_2x_3 \newline& + \theta_6x_3^2 )\end{align*}
$$

That gives us 6 features. The exact way to calculate how many features for all polynomial terms is the combination function with repetition: [http://www.mathsisfun.com/combinatorics/combinations-permutations.html](http://www.mathsisfun.com/combinatorics/combinations-permutations.html) $$\frac{(n+r-1)!}{r!(n-1)!}$$​. In this case we are taking all two-element combinations of three features: $$\frac{(3 + 2 - 1)!}{(2!\cdot (3-1)!)}​ = \frac{4!}{4} = 6$$. (**Note**: you do not have to know these formulas, I just found it helpful for understanding).

For 100 features, if we wanted to make them quadratic we would get $$\frac{(100 + 2 - 1)!}{(2\cdot (100-1)!)} = 5050$$resulting new features.

We can approximate the growth of the number of new features we get with all quadratic terms with $$\mathcal{O}(n^2/2)$$. And if you wanted to include all cubic terms in your hypothesis, the features would grow asymptotically at $$\mathcal{O}(n^3)$$. These are very steep growths, so as the number of our features increase, the number of quadratic or cubic features increase very rapidly and becomes quickly impractical.

Example: let our training set be a collection of 50 x 50 pixel black-and-white photographs, and our goal will be to classify which ones are photos of cars. Our feature set size is then n = 2500 if we compare every pair of pixels.

Now let's say we need to make a quadratic hypothesis function. With quadratic features, our growth is $$\mathcal{O}(n^2/2)$$. So our total features will be about $$2500^2 / 2 = 3125000$$, which is very impractical.

Neural networks offers an alternate way to perform machine learning when we have complex hypotheses with many features.

## Neurons and the Brain

Neural networks are limited imitations of how our own brains work. They've had a big recent resurgence because of advances in computer hardware.

There is evidence that the brain uses only one "learning algorithm" for all its different functions. Scientists have tried cutting (in an animal brain) the connection between the ears and the auditory cortex and rewiring the optical nerve with the auditory cortex to find that the auditory cortex literally learns to see.

This principle is called "neuroplasticity" and has many examples and experimental evidence.

## Model Representation I

Let's examine how we will represent a hypothesis function using neural networks. At a very simple level, neurons are basically computational units that take inputs (**dendrites**) as electrical inputs (called "spikes") that are channeled to outputs (**axons**). In our model, our dendrites are like the input features $$x_1\cdots x_n$$​, and the output is the result of our hypothesis function. In this model our $$x_0$$​ input node is sometimes called the "bias unit." It is always equal to 1. In neural networks, we use the same logistic function as in classification, $$\frac{1}{1 + e^{-\theta^Tx}}$$​, yet we sometimes call it a sigmoid (logistic) **activation** function. In this situation, our "theta" parameters are sometimes called "weights".

Visually, a simplistic representation looks like:

$$
[x0x1x2]\rightarrow[    ]\rightarrow h_\theta(x)
$$

Our input nodes (layer 1), also known as the "input layer", go into another node (layer 2), which finally outputs the hypothesis function, known as the "output layer".

We can have intermediate layers of nodes between the input and output layers called the "hidden layers."

In this example, we label these intermediate or "hidden" layer nodes $$a^2_0 \cdots a^2_n$$and call them "activation units."

$$
\begin{align*}& a_i^{(j)} = \text{"activation" of unit $i$ in layer $j$} \newline& \Theta^{(j)} = \text{matrix of weights controlling function mapping from layer $j$ to layer $j+1$}\end{align*}
$$

If we had one hidden layer, it would look like:

$$
[x_0x_1x_2x_3]\rightarrow[a^{(2)}_1a^{(2)}_2a^{(2)}_3]\rightarrow h_\theta(x)
$$

The values for each of the "activation" nodes is obtained as follows:

$$
\begin{align*} a_1^{(2)} = g(\Theta_{10}^{(1)}x_0 + \Theta_{11}^{(1)}x_1 + \Theta_{12}^{(1)}x_2 + \Theta_{13}^{(1)}x_3) \newline a_2^{(2)} = g(\Theta_{20}^{(1)}x_0 + \Theta_{21}^{(1)}x_1 + \Theta_{22}^{(1)}x_2 + \Theta_{23}^{(1)}x_3) \newline a_3^{(2)} = g(\Theta_{30}^{(1)}x_0 + \Theta_{31}^{(1)}x_1 + \Theta_{32}^{(1)}x_2 + \Theta_{33}^{(1)}x_3) \newline h_\Theta(x) = a_1^{(3)} = g(\Theta_{10}^{(2)}a_0^{(2)} + \Theta_{11}^{(2)}a_1^{(2)} + \Theta_{12}^{(2)}a_2^{(2)} + \Theta_{13}^{(2)}a_3^{(2)}) \newline \end{align*}
$$

This is saying that we compute our activation nodes by using a 3×4 matrix of parameters. We apply each row of the parameters to our inputs to obtain the value for one activation node. Our hypothesis output is the logistic function applied to the sum of the values of our activation nodes, which have been multiplied by yet another parameter matrix $$\Theta^{(2)}$$ containing the weights for our second layer of nodes.

Each layer gets its own matrix of weights, $$\Theta^{(j)}$$.

The dimensions of these matrices of weights is determined as follows:

$$\text{If network has $s_j$ units in layer $j$ and $s_{j+1}$ units in layer $j+1$, then $\Theta^{(j)}$ will be of dimension $s_{j+1} \times (s_j + 1)$.}$$

The +1 comes from the addition in $$\Theta^{(j)}$$ of the "bias nodes," $$x_0$$and $$\Theta_0^{(j)}$$​. In other words the output nodes will not include the bias nodes while the inputs will. The following image summarizes our model representation:

![](<.gitbook/assets/image (29).png>)

Example: If layer 1 has 2 input nodes and layer 2 has 4 activation nodes. Dimension of $$\Theta^{(1)}$$is going to be 4×3 where $$s_j = 2$$_and_ $$s_{j+1} = 4$$, so $$s_{j+1} \times (s_j + 1) = 4 \times 3$$.

## Model Representation II

To re-iterate, the following is an example of a neural network:

$$
\begin{align*} a_1^{(2)} = g(\Theta_{10}^{(1)}x_0 + \Theta_{11}^{(1)}x_1 + \Theta_{12}^{(1)}x_2 + \Theta_{13}^{(1)}x_3) \newline a_2^{(2)} = g(\Theta_{20}^{(1)}x_0 + \Theta_{21}^{(1)}x_1 + \Theta_{22}^{(1)}x_2 + \Theta_{23}^{(1)}x_3) \newline a_3^{(2)} = g(\Theta_{30}^{(1)}x_0 + \Theta_{31}^{(1)}x_1 + \Theta_{32}^{(1)}x_2 + \Theta_{33}^{(1)}x_3) \newline h_\Theta(x) = a_1^{(3)} = g(\Theta_{10}^{(2)}a_0^{(2)} + \Theta_{11}^{(2)}a_1^{(2)} + \Theta_{12}^{(2)}a_2^{(2)} + \Theta_{13}^{(2)}a_3^{(2)}) \newline \end{align*}
$$

In this section we'll do a vectorized implementation of the above functions. We're going to define a new variable $$z_k^{(j)}$$ ​that encompasses the parameters inside our g function. In our previous example if we replaced by the variable z for all the parameters we would get:

$$
\begin{align*}a_1^{(2)} = g(z_1^{(2)}) \newline a_2^{(2)} = g(z_2^{(2)}) \newline a_3^{(2)} = g(z_3^{(2)}) \newline \end{align*}
$$

In other words, for layer j=2 and node k, the variable z will be:

$$
z_k^{(2)} = \Theta_{k,0}^{(1)}x_0 + \Theta_{k,1}^{(1)}x_1 + \cdots + \Theta_{k,n}^{(1)}x_n
$$

The vector representation of x and $$z^{j}$$is:

$$
\begin{align*}x = \begin{bmatrix}x_0 \newline x_1 \newline\cdots \newline x_n\end{bmatrix} &z^{(j)} = \begin{bmatrix}z_1^{(j)} \newline z_2^{(j)} \newline\cdots \newline z_n^{(j)}\end{bmatrix}\end{align*}
$$

Setting $$x = a^{(1)}$$ , we can rewrite the equation as:

$$
z^{(j)} =Θ^{(j−1)} a^{(j−1)}
$$

We are multiplying our matrix $$\Theta^{(j-1)}$$ with dimensions $$s_j\times (n+1)$$(where $$s_j$$​ is the number of our activation nodes) by our vector $$a^{(j-1)}$$with height (n+1). This gives us our vector $$z^{(j)}$$with height $$s_j$$​. Now we can get a vector of our activation nodes for layer j as follows:

$$a^{(j)} = g(z^{(j)})$$

Where our function g can be applied element-wise to our vector $$z^{(j)}$$.

We can then add a bias unit (equal to 1) to layer j after we have computed $$a^{(j)}$$. This will be element $$a_0^{(j)}$$and will be equal to 1. To compute our final hypothesis, let's first compute another z vector:

$$z^{(j+1)} = \Theta^{(j)}a^{(j)}$$

We get this final z vector by multiplying the next theta matrix after $$\Theta^{(j-1)}$$with the values of all the activation nodes we just got. This last theta matrix $$\Theta^{(j)}$$ will have only **one row** which is multiplied by one column $$a^{(j)}$$ so that our result is a single number. We then get our final result with:

$$h_\Theta(x) = a^{(j+1)} = g(z^{(j+1)})$$

Notice that in this **last step**, between layer j and layer j+1, we are doing **exactly the same thing** as we did in logistic regression. Adding all these intermediate layers in neural networks allows us to more elegantly produce interesting and more complex non-linear hypotheses.

## Examples and Intuitions I\\

A simple example of applying neural networks is by predicting $$x_1$$ AND $$x_2$$​, which is the logical 'and' operator and is only true if both $$x_1$$and $$x_2$$are 1.

The graph of our functions will look like:

$$
\begin{align*}\begin{bmatrix}x_0 \newline x_1 \newline x_2\end{bmatrix} \rightarrow\begin{bmatrix}g(z^{(2)})\end{bmatrix} \rightarrow h_\Theta(x)\end{align*}
$$

Remember that $$x_0$$ is our bias variable and is always 1.

Let's set our first theta matrix as:

$$
Θ^{(1)}  = [−30\space20\space20 ]
$$

This will cause the output of our hypothesis to only be positive if both $$x_1$$ ​and $$x_2$$​are 1. In other words:

$$
\begin{align*}& h_\Theta(x) = g(-30 + 20x_1 + 20x_2) \newline \newline & x_1 = 0 \ \ and \ \ x_2 = 0 \ \ then \ \ g(-30) \approx 0 \newline & x_1 = 0 \ \ and \ \ x_2 = 1 \ \ then \ \ g(-10) \approx 0 \newline & x_1 = 1 \ \ and \ \ x_2 = 0 \ \ then \ \ g(-10) \approx 0 \newline & x_1 = 1 \ \ and \ \ x_2 = 1 \ \ then \ \ g(10) \approx 1\end{align*}
$$

So we have constructed one of the fundamental operations in computers by using a small neural network rather than using an actual AND gate. Neural networks can also be used to simulate all the other logical gates. The following is an example of the logical operator 'OR', meaning either $$x_1$$​\
is true or $$x_2$$ ​is true, or both:

![](<.gitbook/assets/image (30).png>)

Where g(z) is the following:

![](<.gitbook/assets/image (31).png>)

## Examples and Intuitions II

The $$Θ^{(1)}$$matrices for AND, NOR, and OR are:

$$
\begin{align*}AND:\newline\Theta^{(1)} &=\begin{bmatrix}-30 & 20 & 20\end{bmatrix} \newline NOR:\newline\Theta^{(1)} &= \begin{bmatrix}10 & -20 & -20\end{bmatrix} \newline OR:\newline\Theta^{(1)} &= \begin{bmatrix}-10 & 20 & 20\end{bmatrix} \newline\end{align*}
$$

We can combine these to get the XNOR logical operator (which gives 1 if $$x_1$$and $$x_2$$are both 0 or both 1).

$$
\begin{align*}\begin{bmatrix}x_0 \newline x_1 \newline x_2\end{bmatrix} \rightarrow\begin{bmatrix}a_1^{(2)} \newline a_2^{(2)} \end{bmatrix} \rightarrow\begin{bmatrix}a^{(3)}\end{bmatrix} \rightarrow h_\Theta(x)\end{align*}
$$

For the transition between the first and second layer, we'll use a $$Θ^{(1)}$$matrix that combines the values for AND and NOR:

$$
\Theta^{(1)} =[−30\space\space20\space\space2010\space\space-20\space-20]
$$

For the transition between the second and third layer, we'll use a $$Θ^{(2)}$$ matrix that uses the value for OR:

$$
\Theta^{(2)} =[−10\space20\space20]
$$

Let's write out the values for all our nodes:

$$
\begin{align*}& a^{(2)} = g(\Theta^{(1)} \cdot x) \newline& a^{(3)} = g(\Theta^{(2)} \cdot a^{(2)}) \newline& h_\Theta(x) = a^{(3)}\end{align*}
$$

And there we have the XNOR operator using a hidden layer with two nodes! The following summarizes the above algorithm:

![](<.gitbook/assets/image (32).png>)

## Multiclass Classification

To classify data into multiple classes, we let our hypothesis function return a vector of values. Say we wanted to classify our data into one of four categories. We will use the following example to see how this classification is done. This algorithm takes as input an image and classifies it accordingly:

![](<.gitbook/assets/image (33).png>)

We can define our set of resulting classes as y:

![](<.gitbook/assets/image (34).png>)

Each $$y^{(i)}$$ represents a different image corresponding to either a car, pedestrian, truck, or motorcycle. The inner layers, each provide us with some new information which leads to our final hypothesis function. The setup looks like:

![](<.gitbook/assets/image (35).png>)

Our resulting hypothesis for one set of inputs may look like:

$$
h_\Theta(x) =[0\space0\space1\space0]
$$

In which case our resulting class is the third one down, or $$h_\Theta(x)_3$$​, which represents the motorcycle.

## Cost Function



Let's first define a few variables that we will need to use:

* L = total number of layers in the network
* $$s_l$$= number of units (not counting bias unit) in layer l
* K = number of output units/classes

Recall that in neural networks, we may have many output nodes. We denote $$h_\Theta(x)_k$$as being a hypothesis that results in the $$k^{th}$$output. Our cost function for neural networks is going to be a generalization of the one we used for logistic regression. Recall that the cost function for regularized logistic regression was:

$$
J(\theta) = - \frac{1}{m} \sum_{i=1}^m [ y^{(i)}\ \log (h_\theta (x^{(i)})) + (1 - y^{(i)})\ \log (1 - h_\theta(x^{(i)}))] + \frac{\lambda}{2m}\sum_{j=1}^n \theta_j^2
$$

For neural networks, it is going to be slightly more complicated:

$$
\begin{gather*} J(\Theta) = - \frac{1}{m} \sum_{i=1}^m \sum_{k=1}^K \left[y^{(i)}_k \log ((h_\Theta (x^{(i)}))_k) + (1 - y^{(i)}_k)\log (1 - (h_\Theta(x^{(i)}))_k)\right] + \frac{\lambda}{2m}\sum_{l=1}^{L-1} \sum_{i=1}^{s_l} \sum_{j=1}^{s_{l+1}} ( \Theta_{j,i}^{(l)})^2\end{gather*}
$$

We have added a few nested summations to account for our multiple output nodes. In the first part of the equation, before the square brackets, we have an additional nested summation that loops through the number of output nodes.

In the regularization part, after the square brackets, we must account for multiple theta matrices. The number of columns in our current theta matrix is equal to the number of nodes in our current layer (including the bias unit). The number of rows in our current theta matrix is equal to the number of nodes in the next layer (excluding the bias unit). As before with logistic regression, we square every term.

Note:

* the double sum simply adds up the logistic regression costs calculated for each cell in the output layer
* the triple sum simply adds up the squares of all the individual Θs in the entire network.
* the i in the triple sum does **not** refer to training example i

## Backpropagation Algorithm

"Backpropagation" is neural-network terminology for minimizing our cost function, just like what we were doing with gradient descent in logistic and linear regression. Our goal is to compute:

$$\min_\Theta J(\Theta)$$

That is, we want to minimize our cost function J using an optimal set of parameters in theta. In this section we'll look at the equations we use to compute the partial derivative of J(Θ):

$$\dfrac{\partial}{\partial \Theta_{i,j}^{(l)}}J(\Theta)$$

To do so, we use the following algorithm:

![](<.gitbook/assets/image (36).png>)



**Back propagation Algorithm**

Given training set $$\lbrace (x^{(1)}, y^{(1)}) \cdots (x^{(m)}, y^{(m)})\rbrace$$

* Set $$\Delta^{(l)}_{i,j}$$:= 0 for all (l,i,j) (hence you end up having a matrix full of zeros)

For training example t =1 to m:

1. Set $$a^{(1)} := x^{(t)}$$
2. Perform forward propagation to compute $$a^{(l)}$$for l=2,3,…,L

![](<.gitbook/assets/image (37).png>)

3\. Using $$y^{(t)}$$, compute $$\delta^{(L)} = a^{(L)} - y^{(t)}$$

Where L is our total number of layers and $$a^{(L)}$$is the vector of outputs of the activation units for the last layer. So our "error values" for the last layer are simply the differences of our actual results in the last layer and the correct outputs in y. To get the delta values of the layers before the last layer, we can use an equation that steps us back from right to left:

4\. Compute $$\delta^{(L-1)}, \delta^{(L-2)},\dots,\delta^{(2)}$$using $$\delta^{(l)} = ((\Theta^{(l)})^T \delta^{(l+1)})\ .*\ a^{(l)}\ .*\ (1 - a^{(l)})$$&#x20;

The delta values of layer l are calculated by multiplying the delta values in the next layer with the theta matrix of layer l. We then element-wise multiply that with a function called g', or g-prime, which is the derivative of the activation function g evaluated with the input values given by $$z^{(l)}$$.

The g-prime derivative terms can also be written out as:

$$
g'(z^{(l)}) = a^{(l)}\ .*\ (1 - a^{(l)})
$$

5\. $$\Delta^{(l)}_{i,j} := \Delta^{(l)}_{i,j} + a_j^{(l)} \delta_i^{(l+1)}$$or with vectorization, $$\Delta^{(l)} := \Delta^{(l)} + \delta^{(l+1)}(a^{(l)})^T$$

Hence we update our new $$\Delta$$matrix.

* $$D^{(l)}_{i,j} := \dfrac{1}{m}\left(\Delta^{(l)}_{i,j} + \lambda\Theta^{(l)}_{i,j}\right)$$, if j≠0.
* $$D^{(l)}_{i,j} := \dfrac{1}{m}\Delta^{(l)}_{i,j}$$​ If j=0

The capital-delta matrix D is used as an "accumulator" to add up our values as we go along and eventually compute our partial derivative. Thus we get $$\frac \partial {\partial \Theta_{ij}^{(l)}} J(\Theta)= D_{ij}^{(l)}$$​

## Backpropagation Intuition

Recall that the cost function for a neural network is:

$$
\begin{gather*}J(\Theta) = - \frac{1}{m} \sum_{t=1}^m\sum_{k=1}^K \left[ y^{(t)}_k \ \log (h_\Theta (x^{(t)}))_k + (1 - y^{(t)}_k)\ \log (1 - h_\Theta(x^{(t)})_k)\right] + \frac{\lambda}{2m}\sum_{l=1}^{L-1} \sum_{i=1}^{s_l} \sum_{j=1}^{s_l+1} ( \Theta_{j,i}^{(l)})^2\end{gather*}
$$

If we consider simple non-multiclass classification (k = 1) and disregard regularization, the cost is computed with:

$$
cost(t) =y^{(t)} \ \log (h_\Theta (x^{(t)})) + (1 - y^{(t)})\ \log (1 - h_\Theta(x^{(t)}))
$$

Intuitively, $$\delta_j^{(l)}$$​ is the "error" for $$a^{(l)}_j$$(unit j in layer l). More formally, the delta values are actually the derivative of the cost function:

$$
\delta_j^{(l)} = \dfrac{\partial}{\partial z_j^{(l)}} cost(t)
$$

Recall that our derivative is the slope of a line tangent to the cost function, so the steeper the slope the more incorrect we are. Let us consider the following neural network below and see how we could calculate some $$\delta_j^{(l)}$$​:

![](<.gitbook/assets/image (38).png>)

In the image above, to calculate $$\delta2^{(2)}$$_, we multiply the weights_ $$\Theta_{12}^{(2)}$$and \Theta_{22}^{(2)}Θ 22 (2) ​by their respective \deltaδ values found to the right of each edge. So we get \delta\_2^{(2)}δ 2 (2) ​= \Theta_{12}^{(2)}Θ 12 (2) ​_\delta1^{(3)}δ 1 (3) ​+\Theta{22}^{(2)}Θ 22 (2) ​_  \
__\delta_2^{(3)}δ 2 (3) ​ . To calculate every single possible \delta\_j^{(l)}δ j (l) ​_\
_, we could start from the right of our diagram. We can think of our edges as our \Theta_{ij}Θ ij ​\
. Going from right to left, to calculate the value of \delta_j^{(l)}δ j (l) ​_\
_, you can just take the over all sum of each weight times the \deltaδ it is coming from. Hence, another example would be \delta\_2^{(3)}δ 2 (3) ​_\
_=\Theta_{12}^{(3)}Θ 12 (3) ​\
\*\delta\_1^{(4)}δ 1 (4) ​\
.
