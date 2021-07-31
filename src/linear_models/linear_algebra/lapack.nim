import
  arraymancer,
  arraymancer/linear_algebra/helpers/triangular,
  nimlapack,
  ../utils/exceptions,
  ../utils/tensor_utils

proc dpotrf[T: float](X: Tensor[T], uplo: string): Tensor[T] =
    
  # Sanity checks
  checkTensor2D(X)
  checkTensorSquare(X)

  # Check if uplo is valid
  if not (uplo in ["U", "L"]): raise newException(ValueError, "`uplo` must be 'U' or 'L'")

  # Get Tensor contiguity
  let XContiguity = getTensorContiguity(X)

  # Variables
  var
    uplo_cp = uplo.cstring
    n = X.shape[0].cint
    a = X.clone(XContiguity).astype(cdouble)
    lda = X.shape[1].cint
    info = 0.cint

  # Reverse `uplo` if C contiguous
  if XContiguity == rowMajor:
    if uplo == "L":
      uplo_cp = "U"
    elif uplo == "U":
      uplo_cp = "L"

  # CLAPACK CALL: http://www.netlib.org/lapack/explore-html/d1/d7a/group__double_p_ocomputational_ga2f55f604a6003d03b5cd4a0adcfb74d6.html
  dpotrf(uplo_cp, n.addr, a.getDataPtr, lda.addr, info.addr)

  # Check if errors
  checkLapackSuccessful(info)

  # Triangle cleanup
  if uplo == "U": a = a.triu().astype(float)
  elif uplo == "L": a = a.tril().astype(float)

  return a

 
proc choleskyDecomposition*[T: float](X: Tensor[T], uplo: string = "L"): Tensor[T] =
  result = dpotrf(X, uplo)

