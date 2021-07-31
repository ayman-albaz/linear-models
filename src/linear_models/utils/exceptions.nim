import arraymancer
import strformat


type
  TensorShapeError* = object of ValueError
  TensorDimensionalityError* = object of ValueError
  TensorContiguityInconsistency* = object of ValueError
  LapackError* = object of ValueError


proc isTensorSquare*[T](X: Tensor[T]): bool = 
  result = X.shape[0] == X.shape[1]


proc checkTensorSquare*[T](X: Tensor[T]) = 
  if not isTensorSquare(X): raise newException(TensorShapeError, "`X` must be a square matrix")


proc isTensor1D*[T](X: Tensor[T]): bool = 
  result = X.rank == 1


proc checkTensor1D*[T](X: Tensor[T]) = 
  if not isTensor1D(X): raise newException(TensorDimensionalityError, "`X` must be 1D")


proc isTensor2D*[T](X: Tensor[T]): bool = 
  result = X.rank == 2


proc checkTensor2D*[T](X: Tensor[T]) = 
  if not isTensor2D(X): raise newException(TensorDimensionalityError, "`X` must be 2D")


proc isTensorContiguityConsistency[T](X1, X2: Tensor[T]): bool =
  result = (X1.isCContiguous and X2.isCContiguous) or (X1.isFContiguous and X2.isFContiguous)


proc checkTensorContiguityConsistency*[T](X1, X2: Tensor[T]) = 
  #[
    Not used yet but maybe useful in the future
  ]#
  if not isTensorContiguityConsistency(X1, X2):
    raise newException(TensorContiguityInconsistency, "Both tensors must have the same contingency")


proc isLapackSuccessful*[T: SomeInteger](info: T): bool = 
  result = info == 0


proc checkLapackSuccessful*[T: SomeInteger](info: T) = 
  if info < 0:
    raise newException(LapackError, fmt"The {-1 * info}-th argument had an illegal value.")
  elif info > 1:
    raise newException(LapackError, fmt"The leading minor of order {info} is not positive definite, and the factorization could not be completed.")
