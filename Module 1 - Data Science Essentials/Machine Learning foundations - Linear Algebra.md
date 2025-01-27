# Introduction to Linear Algebra
## Defining Linear Algebra
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

# Vector Basics

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

# Vector Projections and Basis

## Dot Product of Vectors

## Scalar and Vector 
# Introduction to Matrices
# Gaussian Elimination
# Matrices from Orthogonality to Gram-Schmidt Process
# Eigenvalues and Eigenvectors
# Conclusion