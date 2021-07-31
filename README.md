![Linux Build Status (Github Actions)](https://github.com/ayman-albaz/linear-models/actions/workflows/install_and_test.yml/badge.svg) [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


# Linear Models
Linear models is a Nim library for generalized linear models. It is written ontop of [Arraymancer](https://github.com/mratsim/Arraymancer) tensors.


## Compilation flags
This library uses BLAS and LAPACK in its Arraymancer backend. Please use the [appropriate flags when compiling](https://github.com/mratsim/Arraymancer#performance-notice-on-nim-020--compilation-flags). I prefer using the [Arraymancer .cfg file](https://github.com/mratsim/Arraymancer/blob/master/nim.cfg) by dropping the .cfg file into the main directory of my project.


## Supported Linear Models
| Model                                            | Nim Command              |
|--------------------------------------------------|--------------------------|
| Gaussian regression (linear regression)          | `glm(X, y, Gaussian())`  |
| Binomial regression (binary logistic regression) | `glm(X, y, Binomial())`  |
| Poisson regression                               | `glm(X, y, Poisson())`   |

Note: This library also supports `Gamma()` regression, however it is currently unstable for most inputs due to the optimizer implementation.


## Examples

### Binomial Regression
Note: Each observation should be a row, and all inputs (both X and y) should be `float64` (this is because BLAS calls need float64 inputs). X inputs must be 2D and y inputs must be 1D.
```Nim
import arraymancer
import linear_models

  let
    X = [[0.95601119,  0.87647851],
         [-2.20004465, -0.62625987],
         [-1.27545515,  1.32644564],
         [-1.44131698,  0.39791802],
         [-2.1776243 , -0.37052885],
         [-0.29938274,  1.29160856],
         [-2.52902482, -0.40531331],
         [-0.45909187,  1.00496831],
         [-2.77913571,  1.74098504],
         [-0.86087541,  2.6546214 ],
         [-2.85495442,  0.43957948],
         [ 0.33060411,  0.23314301],
         [-0.78649263,  1.38671912],
         [-1.06159023,  0.924985  ]].toTensor().astype(float)
    y = [0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0].toTensor().astype(float)
    
    # GLM fit
    result1 = glm(X, y, Binomial())

    # GLM fit with 95% two-tailed confidence interval
    result1 = glm(X, y, Binomial(), confidenceIntervalAlpha=0.05)

    # GLM fit with intercept
    result2 = glm(X.addConstant(), y, Binomial())

    # GLM fit with polynomial expansion and intercept
    result3 = glm(X.polynomialFeatures(degree=3).addConstant(), y, Binomial())

    # Viewing results - fitSummary
    discard result1.fitSummary.coefficients
    discard result1.fitSummary.covariance
    discard result1.fitSummary.residuals
    discard result1.fitSummary.totalIter
    discard result1.fitSummary.maxIter
    discard result1.fitSummary.dispersion
    discard result1.fitSummary.degreesFreedom

    # Viewing results - statsSummary
    discard result1.statsSummary.hasConverged
    discard result1.statsSummary.startTime
    discard result1.statsSummary.endTime

    # Viewing results - statsTable
    for i in 0..<X.shape[1]:
      discard result1.statsSummary.statsTable[i].coefficient
      discard result1.statsSummary.statsTable[i].standardError
      discard result1.statsSummary.statsTable[i].zScore
      discard result1.statsSummary.statsTable[i].pValue
      discard result1.statsSummary.statsTable[i].confidenceIntervalLower
      discard result1.statsSummary.statsTable[i].confidenceIntervalUpper

```

### Bonus procs
This library includes some linear algebra procs that maybe useful for your other projects. Future work includes integrating these useful procs into the Arraymancer library and removing them from this one.

- `backSolve*[T: float](X, B: Tensor[T]): Tensor[T]` which is similar to R's `backsolve` and uses the `dtrsm` from CBLAS.
- `forwardSolve*[T: float](X, B: Tensor[T]): Tensor[T]` which is similar to R's `forwardsolve` and uses the `dtrsm` from CBLAS.
- `choleskyDecomposition*[T: float](X: Tensor[T], uplo: string = "L"): Tensor[T]` which is similar to R's `cholesky` and uses the `dpotrf` from CLAPACK.
- `pinv*[T: SomeFloat](X: Tensor[T]): Tensor[T]` which is similar to Numpy's `np.linalg.pinv` and uses the `SVD` algorithm from Arraymancer.


## Accuracy
All functions in this library are accurate up-to 14 decimal places (float64).


## Performance
This library was written with accuracy as a top priority as opposed to performance, however almost all implementations here are faster than Statsmodels and R implementations and equal to, slower, or faster than Julia's distributions implementations. 


## TODO
List is organized from most important to least important:
- Change optimization method to a more stable one so things like Gamma regression work properly (help needed here).
- Add multinomial regression (help needed here).
- Add integration with a DataFrames library so that formulas can be used (help needed here).
- Add other GLM families.


Performance, feature, and documentation PR's are always welcome.



## Contact
I can be reached at aymanalbaz98@gmail.com

 