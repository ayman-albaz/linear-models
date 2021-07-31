import 
  arraymancer,
  ../families,
  ../linear_algebra,
  ../utils


type
  FitSummary* = object
    coefficients*, covariance*, residuals*: Tensor[float]
    totalIter*, maxIter*: int
    dispersion*, degreesFreedom*: float

 
proc iterativelyReweightedLeastSquares*[T: float, S: Family](X, y: Tensor[T],
                                                             family: S,
                                                             maxIter: int = 200,
                                                             tolerance: T = 1e-8): FitSummary =
  #[
      Adapted from: https://bwlewis.github.io/GLM/ which is based off of the R implementation
  ]# 

  # Init variables
  let 
    y2D = y.reshape1DColTo2D()
    degreesFreedom = X.shape[0].float - X.shape[1].float

  var
    X_qr = qr(X)
    s = zeros[T](X.shape[1], 1)
    sOld = zeros[T](X.shape[1], 1)
    t = zeros[T](X.shape[0], 1)
    curIter: int
    wMin: T
    g, gPrime, residuals, z, W, C, x: Tensor[T]

  # IRLS algorithm
  for i in 1..maxIter:

    # Applying inverse link, gradient, variance function
    g = family.inverseLink(t)
    gPrime = family.gradientInverseLink(t)
    residuals = (y2D -. g) /. gPrime
    z = reshape1DColTo2D(t +. residuals)
    W = (gPrime ^. 2.T) /. family.variance(g)
    
    # Tiny weights warning
    wMin = min(W)
    if wMin < sqrt(EPS): echo "WARNING: Tiny weights encountered."

    # BLAS / LAPACK stable least squares solving method
    sOld = s
    C = choleskyDecomposition(X_qr.Q.transpose * (W *. X_qr.Q))
    s = forwardSolve(C, X_qr.Q.transpose * (W *. z))
    s = backsolve(C.transpose, s)
    t = X_qr.Q * s

    # Keep track of current iteration
    curIter += 1

    # End early if little learning occurs. Note: R does matmul here.
    if all(abs(s -. sOld) <. tolerance): break

  # Final solve
  x = backsolve(X_qr.R, X_qr.Q.transpose * t)
  
  # Auxiliary stats
  let 
    phi = family.dispersion(residuals, degreesFreedom)  #TODO: Not removing 0's from residuals may cause errors, see: https://github.com/SurajGupta/r-source/blob/master/src/library/stats/R/glm.R#L652
    covariance = phi * pinv(X.transpose * W.reshape2DTo1D().diagonal() * X)

  # Summary object
  result = FitSummary(coefficients: x.transpose,
                      covariance: covariance,
                      residuals: residuals,
                      dispersion: phi,
                      degreesFreedom: degreesFreedom,
                      totalIter: curIter, maxIter: maxIter)

