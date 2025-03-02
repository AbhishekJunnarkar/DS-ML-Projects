
# **1. Introduction to Linear Algebra**

##  Defining Linear Algebra
- Linear Algebra vs. Algebra: Linear algebra is not just an advanced version of algebra. While algebra deals with mathematical
symbols and their manipulation, linear algebra focuses on vectors and linear functions.
- Core Concepts: The main building blocks of linear algebra include systems of linear equations, vectors, matrices, linear 
transformations, determinants, and vector spaces.
- Importance in Machine Learning: Linear algebra is crucial for machine learning as it helps in handling large datasets,
which are often represented as matrices. Operations like splitting datasets into training and testing sets involve matrix manipulations.

## Applications of Linear Algebra in ML

- Data Sets and Data Files: Machine learning models are fitted on data sets, which are represented as matrices or vectors.
- Images and Photographs: In computer vision, images are stored as matrices, and operations on images (like cropping and scaling) use linear algebra.
- Data Preparation: Techniques like dimensionality reduction and one-hot encoding rely on linear algebra.
- Linear Regression: Solved using least squares optimization, a method from linear algebra.
- Regularization: Prevents overfitting by minimizing the size of coefficients during model fitting.
- Principal Component Analysis (PCA): Reduces the number of features in a data set for visualization and training.
- Latent Semantic Analysis (LSA): Used in natural language processing to handle text data.
- Recommender Systems: Predictive models that recommend products based on previous purchases.
- Deep Learning: Works with vectors, matrices, and tensors of inputs and coefficients.

----------------------
# **2. Vector Basics**

## Introduction to Vectors
- Scalars vs. Vectors: Scalars are just numbers (e.g., weight, temperature) denoted by lowercase symbols. Vectors have both 
magnitude and direction (e.g., velocity) and are denoted by bolded Roman letters.
- Vector Characteristics: Vectors have dimensionality (number of elements) and orientation (column or row).
- Vector Representation: Vectors can be represented graphically as arrows with a specific length and direction.
- Python Representation: In Python, vectors can be represented as lists or NumPy arrays, with orientation indicated by brackets.

**Python Representation:**

- List: vectorAsList = [a, b, c]
- NumPy Array:
    - Row Vector: rowVector = np.array([[a, b, c]])
    - Column Vector: columnVector = np.array([[a], [b], [c]])
## Vector Arithmatic

- Addition and Subtraction: You can add or subtract vectors elementwise only if they have the same dimension.
- Multiplication and Division: Vectors can be multiplied or divided elementwise if they have the same length.
- Vector-Scalar Multiplication: Multiplying a vector by a scalar involves multiplying each element of the vector by the scalar.
- Python Representation: Differences between lists and NumPy arrays are highlighted, showing how operations differ based on the data type.
- Geometric Interpretation: Addition and subtraction of vectors can be visualized geometrically, with specific rules for 
placing vectors and interpreting their sum or difference.

## Coordinate system

- Chessboard Analogy: The Cartesian coordinate system is likened to a chessboard, where positions are determined by rows and columns.
- Axes and Origin: The x-axis runs left and right, while the y-axis runs up and down. They intersect at the origin (0, 0).
- Point Representation: Points are denoted by (x, y) coordinates. For example, (2, 3) means 2 units right on the x-axis and 3 units up on the y-axis.
- Unit Vectors: Vectors OA and OB, with magnitudes of 1, are called unit vectors along the x and y axes, respectively, and are denoted as I and J.
- Vector Addition: The sum of two vectors, such as OC and OD, is represented as 4I + 3J, following the rule of vector addition.

----------------------
# **3. Vector Projections and Basis**

## Dot Product of Vectors
- Dot Product Definition: The dot product of two vectors is calculated by multiplying their corresponding elements and summing the results.
- Formula: For vectors a and b with elements (a_i) and (b_i), the dot product is (a \cdot b = \sum (a_i \times b_i)).
- Example: For vectors a = [1, 2, 3, 4, 5] and b = [6, 7, 8, 9, 10], the dot product is calculated as (1 \times 6 + 2 \times 7 + 3 \times 8 + 4 \times 9 + 5 \times 10).
- Python Implementation: Using NumPy, the dot product can be calculated with np.dot(a, b).
- Properties:
  - Commutative: (a . b = b . a)
  - Distributive over Addition: (a . (b + c) = a . b + a . c)

Geometric Interpretation: The dot product measures the similarity or mapping between two vectors.
## Scalar and Vector projection

- Vector Magnitude: Also known as the norm or geometric length, it is the distance from the tail to the head of a vector, calculated using the Euclidean distance formula.
  - ||x|| to represent vector magnitude of x
  - There is a function in NumPy called norm
  - Magnitude = np.norm(a)
- Scalar Projection: The scalar projection of vector a onto vector b is the dot product of a and b divided by the magnitude of b.
- Vector Projection: To calculate the vector projection of a onto b, multiply the scalar projection by the unit vector of b.
- Python Implementation: Using NumPy, the norm function (np.linalg.norm) calculates the magnitude, and the projection can be computed using dot products and scalar multiplications.

## Changing basis of vectors

- Basis Vectors: Basis vectors are linearly independent and span the entire space. Any vector can be expressed as a linear combination of basis vectors.
- Changing Basis: You can change from one set of basis vectors to another. This involves expressing a vector in terms of a new set of basis vectors.
- Orthogonality: For the change of basis to be valid, the new basis vectors must be orthogonal.
- Vector Projection: The dot product is used to find the projections of a vector onto the new basis vectors, which helps in expressing the vector in the new basis.

## Basis,linear independence, and span

- Spanning Set: A set of vectors is a spanning set for a vector space if every vector in that space can be written as a linear combination of the set.
- Basis Vectors: Basis vectors are linearly independent and span the entire vector space. They don't need to be unit vectors or orthogonal.
- Linear Independence: Vectors are linearly independent if none of them can be written as a linear combination of the others.

----------------------
# **4. Introduction to Matrices**

## Matrices Introduction

- Definition: A matrix is a collection of numbers ordered in rows and columns, forming a two-dimensional array.
- Elements and Dimensions: Each value in a matrix is called an element. Matrices are defined by their dimensions, such as 3x2 for a matrix with 3 rows and 2 columns.
- Arithmetic Operations: Basic arithmetic operations like addition, subtraction, and multiplication can be performed on matrices.
- Content: Matrices can contain numbers, symbols, or expressions.

![2X2 MATRIX](RESOURCES/IMAGES/2X2MATRIX.png)

## Types of Matrices

- Rectangular Matrix: Has a different number of rows and columns (m by n).
- Square Matrix: A special case of a rectangular matrix with the same number of rows and columns (m by m).
- Symmetric Matrix: A square matrix with elements mirrored across the diagonal.
- Zero Matrix: All elements are zero.
- Identity Matrix: A square matrix with ones on the diagonal and zeros elsewhere.
- Diagonal Matrix: Off-diagonal elements are zero, and diagonal elements can be any number.
- Triangular Matrix: A square matrix with either upper or lower triangular elements being zero.
  - Upper Triangular Matrix
  - Lower Triangular Matrix

## Types of Matrices Transformation

- Linear Transformation: Any linear transformation in a plane or space can be specified using vectors or matrices.
- Basic Transformations: Includes scaling, reflecting, rotating, and projecting vectors.
- Combination of Transformations: Advanced transformations can be created by combining basic transformations, such as stretching and rotating.

## Composition or combination of matrix transformations

- Composition of Linear Transformations: Combining multiple linear transformations into a single transformation is called composition.
- Matrix Representation: Any composed linear transformations can be represented as matrices. The product of matrices represents the composition.
- Efficiency: Composing multiple transformations into one matrix reduces computational complexity, making it more efficient to apply the combined transformation.

----------------------
# **5. Gaussian Elimination**

- Gaussian Elimination Process: This method involves converting a system of linear equations into an augmented matrix and performing row operations to solve for the variables.
- Steps:
  1. Convert the system to a matrix-vector equation.
  2. Augment the coefficient matrix with the vector of constants.
  3. Create a matrix with ones on the diagonals (pivoting).
  4. Map the matrix back to equations.
  5. Substitute to solve for variables.

- Outcomes: The system can have a unique solution, no solution, or infinitely many solutions depending on the final form of the matrix.

## Gaussian elimination and finding the inverse matrix

- Matrix Inversion: The inverse of a matrix (A) is denoted as (A^{-1}), and when multiplied by (A), it results in the identity matrix.
- Solving Linear Equations: If matrix (A) is invertible, the system (Ax = B) has a unique solution (x = A^{-1}B).
- Python Implementation: Using NumPy, the inverse of a matrix can be calculated with the inv function from the linalg module, and matrix multiplication can be performed using the dot method.

## Inverse and determinant

- Determinant: The determinant of a matrix is a scalar value that helps determine if a matrix can be inverted. If the determinant is zero, 
the matrix is singular and cannot be inverted.
- Calculation: For a 2x2 matrix (A) with elements (a, b, c,) and (d), the determinant is calculated as ( \text{det}(A) = ad - bc ).
- Python Implementation: Using NumPy, the determinant can be calculated with np.linalg.det(A), and the inverse can be found with np.linalg.inv(A).

----------------------
# **6. Matrices from Orthogonality to Gram-Schmidt Process**

## Matrices changing basis

- Change of Basis Matrix: A matrix that translates vector representations from one basis to another. It allows transformations when the new basis vectors are not orthogonal.
- Transformation Matrix: Constructed from new basis vectors, it helps convert vectors from an alternative vector space to the standard coordinate system.
- Inverse Transformation: To reverse the transformation, you need to find the inverse of the transformation matrix, which, when multiplied by the original matrix, yields the identity matrix.

## Transforming to the new basis

- Matrix Transformations: These are special functions arising from matrix multiplication, mapping vectors from one space to another.
- Standard Basis Vectors: Any vector in ( R^n ) can be expressed as a linear combination of standard basis vectors ( e_1, e_2, \ldots, e_n ).
- Transformation Process: To transform a vector to a new basis:
  1. Use a transformation matrix ( A ) to convert the vector to the standard coordinate system.
  2. Apply a custom transformation using another matrix ( R ).
  3. Transform the result back to the alternate coordinate system using the inverse of ( A ).

## Orthogonal matrix

- Orthonormal Vectors: These vectors are orthogonal (at right angles) to each other and have a unit norm.
- Orthogonal Matrix: Denoted by ( Q ), it is composed of orthonormal vectors as its rows and columns.
- Transpose Property: For an orthogonal matrix ( Q ), the transpose ( Q^T ) is equal to its inverse ( Q^{-1} ). 
When ( Q^T ) is multiplied by ( Q ), it results in the identity matrix.

## Gram-Schmidt process

- Orthogonal Basis: The Gram-Schmidt process transforms a set of vectors into an orthogonal basis, making calculations simpler.
- Step-by-Step Orthogonalization: Each vector is orthogonalized relative to the previous vectors, ensuring all columns are orthogonal.
- Normalization: After orthogonalization, each vector is normalized to unit length, resulting in an orthonormal basis.

# **7. Eigenvalues and Eigenvectors**


## Introduction to eigenvalues and eigenvectors

- Eigenvalues and Eigenvectors: These are fundamental in eigendecomposition, which is defined for square matrices. 
Each eigenvalue has an associated eigenvector.
- Transformation Representation: The equation ( A \cdot v = \lambda \cdot v ) shows that the transformation matrix ( A ) 
behaves like a scalar ( \lambda ) when applied to eigenvector ( v ).
- Applications: Eigenvalues and eigenvectors simplify machine learning models, making them easier to train and 
understand data correlations, with uses in recommendation systems and financial risk analysis.

## Calculating eigenvalues and eigenvectors

- Eigenvalue Equation: The equation ( A \cdot v = \lambda \cdot v ) defines eigenvalues (( \lambda )) and eigenvectors (( v )) for a matrix ( A ).
- Calculation Steps:
 1. Shift the matrix ( A ) by ( \lambda ) times the identity matrix ( I ).
 2. Set the determinant of ( A - \lambda I ) to zero and solve for ( \lambda ) to find eigenvalues.
 3. Substitute eigenvalues back into ( A - \lambda I ) to solve for eigenvectors.

Application: Understanding eigenvalues and eigenvectors is crucial for techniques like Principal Component Analysis (PCA), which reduces dimensionality in machine learning models.

## Changing to the eigenbasis

- Diagonalization: The process of diagonalizing a matrix involves finding its eigenvalues and eigenvectors, which simplifies the calculation of high powers of the matrix.
- Efficiency: Diagonalizing a matrix makes it easier and more efficient to compute large powers of the matrix, as you only need to raise the diagonal elements to the desired power.
- Eigenbasis Conversion: Creating an eigenbasis conversion matrix involves using eigenvectors as columns, allowing for efficient matrix transformations.

## Google PageRank algorithm

- PageRank Concept: PageRank is a core component of Google's search engine algorithm, determining the importance of web pages based on the number and quality of links to them.
- Probability Calculation: The importance of a page is quantified as a probability value between 0 and 1, representing the likelihood of a user clicking on a link to that page.
- Matrix Representation: The network of links is represented by a stochastic matrix, where each element indicates the probability of transitioning from one page to another. Eigenvalues and eigenvectors are used to solve the system of linear equations that determine the PageRank.


# Conclusion

