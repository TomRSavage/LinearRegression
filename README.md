# Linear Regression
## An analysis of shrinkage methods for linear regression (adapted from 'Elements of Statistical Learning')

The classic linear regression problem has a simple cost function
<img src="https://render.githubusercontent.com/render/math?math=min \quad \sum _i (y_i-\beta _0 - \sum _j x_{i,j}\beta _j)^2">
where <img src="https://render.githubusercontent.com/render/math?math=\beta _0"> represents the intercept, <img src="https://render.githubusercontent.com/render/math?math=\beta"> represents the vector of linear coefficients, and <img src="https://render.githubusercontent.com/render/math?math=x,y"> represent the input data matrix and response vector respectively. However how do we balance bias and variance using this approach? Below the data from the book Elements of Statistical Learning is used. 8 medically related variables are used to predict log of prostate specific antigen. The data is first augmented twice using a gaussian random distribution, and the data normalised to have mean 0 and standard deviation of 1. 4 folds were used for validation, and the error averaged over all 4 folds. 

### Ridge Regression 
By adding a constraint <img src="https://render.githubusercontent.com/render/math?math=\sum _j \beta _j ^2 \leq t">, the parameters <img src="https://render.githubusercontent.com/render/math?math=\beta"> are somewhat constrained. By then varying <img src="https://render.githubusercontent.com/render/math?math=t">, the bias, variance trade-off can be determined and an optimum set of parameters can be found.

<img align='center' src="https://github.com/TomRSavage/ShrinkageMethods/blob/master/RidgeLoss.png" width="400"> <img align='center' src="https://github.com/TomRSavage/ShrinkageMethods/blob/master/RidgeParams.png" width="400">

The parameters are seen to vary smoothly. This can be explained by the constraint function limiting the parameters describing a hypersphere in parameter space the parameters must lie within. Assuming the 'best' set of parameters lies outside of this hypersphere, the parameter values at a value of t lie on the boundary. As t is increased and this hypersphere grows in radius , the parameters are smoothly varied on the border of this hypersphere constraint until the optimal set of parameters lie within the hyperspherical contraint. Then the parameters assume a constant value when changing t as the constraint is no longer active. 

### Lasso Regression 
Lasso regression assumes a similar form to ridge regression however instead of a squared constraint, the constraint limits the sum of the absolute value of the parameters, <img src="https://render.githubusercontent.com/render/math?math=\sum _j |\beta _j|  \leq t"> (sometimes called the Taxicab norm for reasons I won't go into). Instead of a smooth hypersphere, this results in a polyhedrally constrained area in parameter space. Due to this now linear volume, the parameter set on the edge of this constraint is more than likely to lie at a vertex. As t is then varied and the polyhedron grows towards the optimal parameter set, the parameter set on the edge of the constraint is more likely to 'snap' into a new parameter dimension, i.e. move to a different vertex. This explains the more discretized parameter values, and the behaviour that certain parameters don't appear at all in the set until t is sufficiently large. 

<img align='center' src="https://github.com/TomRSavage/ShrinkageMethods/blob/master/LassoLoss.png" width="400"> <img align='center' src="https://github.com/TomRSavage/ShrinkageMethods/blob/master/LassoParams.png" width="400">

How late on a parameter is introduced in Lasso regression relates to how 'important' this input is with respect to predicting the output. The sooner the parameter appears, the more important it's respective variable is in predicting the output (in this case amount of prostate specific antigen).



