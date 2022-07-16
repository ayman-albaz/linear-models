import
  arraymancer,
  ../utils/exceptions


proc pinv*[T: SomeFloat](X: Tensor[T]): Tensor[T] = 
  #[
    Uses SVD to find pseudo inverse.
    Based off of SciPy's implementation: https://github.com/scipy/scipy/blob/v1.7.0/scipy/linalg/basic.py#L1241-L1346
  ]#

  # Sanity checks
  checkTensor2D(X)

  # Init variables
  var u, s, vh: Tensor[T]

  # Pseudo inverse
  (u, s, vh) = svd(X)
  u = u /. s.reshape1DRowTo2D()
  result = u * vh