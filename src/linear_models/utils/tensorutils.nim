import 
  arraymancer,
  ../utils/exceptions


proc reshape1DRowTo2D*[T](X: Tensor[T]): Tensor[T] =
  result = X.reshape(1, X.size.int)


proc reshape1DColTo2D*[T](X: Tensor[T]): Tensor[T] =
  result = X.reshape(X.size.int, 1)


proc reshape2DTo1D*[T](X: Tensor[T]): Tensor[T] =
  result = X.reshape(X.size.int)


proc all*(X: Tensor[bool]): bool {.inline.} = 
  for i in X:
    if i == false:
      return false
  return true


proc getTensorContiguity*[T](X: Tensor[T]): OrderType =
  if X.isCContiguous: result = rowMajor
  elif X.isFContiguous: result = colMajor


proc diagonal1D[T](X: Tensor[T]): Tensor[T] =
  result = newTensor[T](X.shape[0], X.shape[0])
  for i, v in enumerate(X): result[i, i] = v


proc diagonal2D[T](X: Tensor[T]): Tensor[T] =

  # Init variables
  let resultLength = min(X.shape[0], X.shape[1])
  result = newTensor[T](resultLength)

  # Get diagonal inputs
  for i in 0..<resultLength:
    result[i] = X[i, i]


proc diagonal*[T](X: Tensor[T]): Tensor[T] =
  if isTensor1D(X): result = diagonal1D(X)
  elif isTensor2D(X): result = diagonal2D(X)
  else: raise newException(TensorShapeError, "`X` must be 1D or 2D.")


proc addConstant1D[T](X: Tensor[T]): Tensor[T] =
  result = concat(ones[T](X.shape[0], 1), X.reshape1DColTo2D(), axis=1)


proc addConstant2D[T](X: Tensor[T]): Tensor[T] =
  result = concat(ones[T](X.shape[0], 1), X, axis=1)


proc addConstant*[T](X: Tensor[T]): Tensor[T] = 
  if isTensor1D(X): result = addConstant1D(X)
  elif isTensor2D(X): result = addConstant2D(X)
  else: raise newException(TensorShapeError, "`X` must be 1D or 2D.")