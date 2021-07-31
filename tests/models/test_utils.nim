import
  unittest,
  arraymancer,
  ../../src/linear_models


suite "utils":

  test "polynomialFeatures(degree=2)":
    let 
      X = [[3, 2], [2, 1]].toTensor()
      result = [[3, 2, 9, 6, 4], [2, 1, 4, 2, 1]].toTensor()
    check polynomialFeatures(X, 2) == result

  test "polynomialFeatures(degree=3)":
    let 
      X = [[3, 2], [2, 1]].toTensor()
      result = [[ 3,  2,  9,  6,  4, 27, 18, 12,  8],
       [ 2,  1,  4,  2,  1,  8,  4,  2,  1]].toTensor()
    check polynomialFeatures(X, 3) == result

  test "polynomialFeatures(degree=4)":
    let 
      X = [[3, 2], [2, 1]].toTensor()
      result = [[ 3,  2,  9,  6,  4, 27, 18, 12,  8, 81, 54, 36, 24, 16],
                [ 2,  1,  4,  2,  1,  8,  4,  2,  1, 16,  8,  4,  2, 1]].toTensor()
    check polynomialFeatures(X, 4) == result