import
  arraymancer,
  ../utils/exceptions


proc calculateUniqueCombinations(a, b: Natural): int =
  for i in 0..<min(a, b):
    for j in i..<max(a, b):
      result += 1


proc combinationsProduct[T: SomeNumber](X1, X2: Tensor[T], depth: int = 0): Tensor[T] = 
  #[
    Depth parameter ensures uniqueness in degree > 2 rounds
  ]#
  
  # Init variables
  let
    m1 = X1.shape[0].int
    n1 = X1.shape[1].int
    n2 = X2.shape[1].int
    numCombinations = calculateUniqueCombinations(n1, n2) - depth
  result = newTensor[T](m1, numCombinations)

  # Find all unique combinations
  for i in 0..<n1:
    for j in i..<n2 - depth + depth * i:
      result[_, i * (n2 - depth) + j - i] = X1[_, i] *. X2[_, j]


proc polynomialFeatures*[T: SomeNumber](X: Tensor[T], degree: Positive = 2): Tensor[T] = 
  #[
    Expands numeric tensor to include all unique polynomial combinations (upto degree).
  ]#

  # Sanity checks
  checkTensor2D(X)
  if degree == 1: return X

  # Init variables
  result = X.clone()
  var expansion = result

  # Algorithm
  for d in 2..degree:
    expansion = combinationsProduct(X, expansion, d - 2)
    result = concat(result, expansion, axis=1)
