# ML:Linear Algebra Review

Khan Academy has excellent Linear Algebra Tutorials ([https://www.khanacademy.org/#linear-algebra](https://www.khanacademy.org/#linear-algebra))

## Matrices and Vectors

Matrices are 2-dimensional arrays:

$$\begin{bmatrix} a & b & c \\ d & e & f \\ g & h & i \\ j & k & l\end{bmatrix}$$

The above matrix has four rows and three columns, so it is a 4 x 3 matrix.

A vector is a matrix with one column and many rows:

$$
\begin{bmatrix} w \\ x \\ y \\ z \end{bmatrix}
$$

So vectors are a subset of matrices. The above vector is a 4 x 1 matrix.

**Notation and terms**:

* $$A_{ij}​$$ refers to the element in the $$i^{th}$$ row and $$j^{th}$$ column of matrix A.
* $$A$$ vector with 'n' rows is referred to as an 'n'-dimensional vector
* $$v_i$$​ refers to the element in the $$i^{th}$$ row of the vector.
* In general, all our vectors and matrices will be 1-indexed. Note that for some programming languages, the arrays are 0-indexed.
* Matrices are usually denoted by uppercase names while vectors are lowercase.
* "Scalar" means that an object is a single value, not a vector or matrix.
* $$\mathbb{R}$$ refers to the set of scalar real numbers
* $$\mathbb{R^n}$$ refers to the set of n-dimensional vectors of real numbers



Run the cell below to get familiar with the commands in Octave/Matlab. Feel free to create matrices and vectors and try out different things.

```
% The ; denotes we are going back to a new row.
A = [1, 2, 3; 4, 5, 6; 7, 8, 9; 10, 11, 12]

% Initialize a vector 
v = [1;2;3] 

% Get the dimension of the matrix A where m = rows and n = columns
[m,n] = size(A)

% You could also store it this way
dim_A = size(A)

% Get the dimension of the vector v 
dim_v = size(v)

% Now let's index into the 2nd row 3rd column of matrix A
A_23 = A(2,3)
```

## Addition and Scalar Multiplication

Addition and subtraction are **element-wise**, so you simply add or subtract each corresponding element:

$$
\begin{bmatrix} a & b \\ c & d \\ \end{bmatrix} +\begin{bmatrix} w & x \\ y & z \\ \end{bmatrix} =\begin{bmatrix} a+w & b+x \\ c+y & d+z \\ \end{bmatrix}
$$

To add or subtract two matrices, their dimensions must be **the same**.

Subtracting Matrices:

$$
\begin{bmatrix} a & b \\ c & d \\ \end{bmatrix} - \begin{bmatrix} w & x \\ y & z \\ \end{bmatrix} =\begin{bmatrix} a-w & b-x \\ c-y & d-z \\ \end{bmatrix}
$$

In scalar multiplication, we simply multiply every element by the scalar value:

$$
\begin{bmatrix} a & b \\ c & d \\ \end{bmatrix} * x =\begin{bmatrix} a*x & b*x \\ c*x & d*x \\ \end{bmatrix}
$$

In scalar division, we simply divide every element by the scalar value:

$$
\begin{bmatrix} a & b \\ c & d \\ \end{bmatrix} / x =\begin{bmatrix} a /x & b/x \\ c /x & d /x \\ \end{bmatrix}
$$

Experiment below with the Octave/Matlab commands for matrix addition and scalar multiplication. Feel free to try out different commands. Try to write out your answers for each command before running the cell below.

```
# importing numpy for matrix operations 
import numpy 
  
# initializing matrices 
x = numpy.array([[1, 2], [4, 5]]) 
y = numpy.array([[7, 8], [9, 10]]) 
  
# using add() to add matrices 
print ("The element wise addition of matrix is : ") 
print (numpy.add(x,y)) 
  
# using subtract() to subtract matrices 
print ("The element wise subtraction of matrix is : ") 
print (numpy.subtract(x,y)) 
  
# using divide() to divide matrices 
print ("The element wise division of matrix is : ") 
print (numpy.divide(x,y)) 

# using multiply() to multiply matrices element wise 
print ("The element wise multiplication of matrix is : ") 
print (numpy.multiply(x,y)) 
  
# using dot() to multiply matrices (matrix multiplication)
print ("The product of matrices is : ") 
print (numpy.dot(x,y)) 

```

## Matrix-Vector Multiplication

We map the column of the vector onto each row of the matrix, multiplying each element and summing the result.

$$
\begin{bmatrix} a & b \newline c & d \\ e & f \end{bmatrix} *\begin{bmatrix} x \\ y \\ \end{bmatrix} =\begin{bmatrix} a*x + b*y \\ c*x + d*y \\ e*x + f*y\end{bmatrix}
$$

The result is a **vector**. The vector must be the **second** term of the multiplication. The number of **columns** of the matrix must equal the number of **rows** of the vector.

An **m x n matrix** multiplied by an **n x 1 vector** results in an **m x 1 vector**.

Below is an example of a matrix-vector multiplication. Make sure you understand how the multiplication works. Feel free to try different matrix-vector multiplications.

```
# importing numpy for matrix operations 
import numpy 
  
# initializing matrices 
x = numpy.array([[1, 2], [4, 5]]) 
y = numpy.array([7, 8]) 

# using dot() to multiply matrices (matrix multiplication)
print ("The product of matrices is : ") 
print (numpy.dot(x,y)) 

```

## Matrix-Matrix Multiplication

We multiply two matrices by breaking it into several vector multiplications and concatenating the result.

$$
\begin{bmatrix} a & b \\ c & d \\ e & f \end{bmatrix} *\begin{bmatrix} w & x \\ y & z \\ \end{bmatrix} =\begin{bmatrix} a*w + b*y & a*x + b*z \\ c*w + d*y & c*x + d*z \\ e*w + f*y & e*x + f*z\end{bmatrix}
$$

An **m x n matrix** multiplied by an **n x o matrix** results in an **m x o** matrix. In the above example, a 3 x 2 matrix times a 2 x 2 matrix resulted in a 3 x 2 matrix.

To multiply two matrices, the number of **columns** of the first matrix must equal the number of **rows** of the second matrix.

For example:

```
# importing numpy for matrix operations 
import numpy 
  
# initializing matrices 
x = numpy.array([[1, 2], [4, 5]]) 
y = numpy.array([[7, 8], [9, 10]]) 

# using dot() to multiply matrices (matrix multiplication)
print ("The product of matrices is : ") 
print (numpy.dot(x,y)) 
```

## Matrix Multiplication Properties

* Not commutative. $$A∗B \neq B∗A$$&#x20;
* Associative. $$(A∗B)∗C=A∗(B∗C)$$

The **identity matrix**, when multiplied by any matrix of the same dimensions, results in the original matrix. It's just like multiplying numbers by 1. The identity matrix simply has 1's on the diagonal (upper left to lower right diagonal) and 0's elsewhere.

$$
\begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \\ \end{bmatrix}
$$

When multiplying the identity matrix after some matrix $$(A∗I)$$, the square identity matrix should match the other matrix's **columns**. When multiplying the identity matrix before some other matrix $$(I∗A)$$, the square identity matrix should match the other matrix's **rows**.



```
# importing numpy for matrix operations 
import numpy 
  
# initializing matrices 
A = numpy.array([[1, 2], [4, 5]]) 
B = numpy.array([[1, 1], [0, 2]]) 

# Initialize a 2 by 2 identity matrix
I = numpy.eye(2, dtype=int)

# The above notation is the same as I = [1,0;0,1]

# What happens when we multiply I*A ? 
IA = numpy.dot(I,A)

# How about A*I ? 
AI = numpy.dot(A,I)

# Compute A*B 
AB = numpy.dot(A,B)

# Is it equal to B*A? 
BA = numpy.dot(B,A)

# Note that IA = AI but AB != BA
```

## Inverse and Transpose

The **inverse** of a matrix A is denoted $$A^{−1}$$. Multiplying by the inverse results in the identity matrix.

A non square matrix does not have an inverse matrix. We can compute inverses of matrices in octave with the $$pinv(A)$$ function and in matlab with the $$inv(A)$$ function. Matrices that don't have an inverse are _singular_ or _degenerate_.

The **transposition** of a matrix is like rotating the matrix 90**°** in clockwise direction and then reversing it. We can compute transposition of matrices in matlab with the transpose(A) function or $$A'$$:

$$
A = \begin{bmatrix} a & b \\ c & d \\ e & f \end{bmatrix}
$$

$$
A^T = \begin{bmatrix} a & c & e \\ b & d & f \\ \end{bmatrix}
$$

In other words:

$$A_{ij} = A^T_{ji}$$

```
# importing numpy for matrix operations 
import numpy 

# Initialize matrix A 
b = numpy.array([[2,3],[4,5]])

# Transpose A 
b_Transpose = b.T
print(b_Transpose)

# Take the inverse of A 
A_inv = numpy.linalg.inv(b)
print(A_inv)

# What is A^(-1)*A? 
A_invA = numpy.dot(A_inv,A)
print(A_invA)
```

