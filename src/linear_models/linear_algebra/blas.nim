import
  arraymancer,
  nimblas,
  ../utils/tensor_utils


proc dtrsm[T: float](X, B: Tensor[T], uplo: UploType): Tensor[T] =

  # Variable init
  let
    XContiguity = getTensorContiguity(X)
  
  var
    order = XContiguity
    side = right
    transa = conjTranspose
    diag = nonUnit
    m = B.shape[1].cint
    n = B.shape[0].cint
    alpha = 1.0.cdouble
    a = X.clone(XContiguity).astype(cdouble)
    lda = X.shape[0].cint
    b = B.clone(XContiguity).astype(cdouble)
    ldb: cint

  # Contiguity fix
  if order == rowMajor: 
    ldb = B.shape[0].cint
  elif order == colMajor: 
    ldb = B.shape[1].cint

  # CBLAS CALL: http://www.netlib.org/lapack/explore-html/d1/d54/group__double__blas__level3_ga6a0a7704f4a747562c1bd9487e89795c.html
  trsm(order, side, uplo, transa, diag, m, n, alpha, a.getDataPtr, lda , b.getDataPtr, ldb)

  return b


proc backSolve*[T: float](X, B: Tensor[T]): Tensor[T] =
  result = dtrsm(X, B, upper)


proc forwardSolve*[T: float](X, B: Tensor[T]): Tensor[T] =
  result = dtrsm(X, B, lower)
