# Linear Regression
## An analysis of shrinkage methods for linear regression (adapted from 'Elements of Statistical Learning')

The classic linear regression problem has a simple cost function
<img src="https://render.githubusercontent.com/render/math?math=min \quad \sum _i (y_i-\beta _0 - \sum _j x_{i,j}\beta _j)^2">
where <img src="https://render.githubusercontent.com/render/math?math=\beta _0"> represents the intercept, <img src="https://render.githubusercontent.com/render/math?math=\beta"> represents the vector of linear coefficients, and <img src="https://render.githubusercontent.com/render/math?math=x,y"> represent the input data matrix and response vector respectively. However how do we balance bias and variance using this approach? 

### Ridge Regression 
By adding a constraint <img src="https://render.githubusercontent.com/render/math?math=\sum _j \beta _j ^2 \leq t">, the parameters <img src="https://render.githubusercontent.com/render/math?math=\beta"> are somewhat constrained. By then varying <img src="https://render.githubusercontent.com/render/math?math=t">, the bias, variance trade-off can be determined and an optimum set of parameters can be found

