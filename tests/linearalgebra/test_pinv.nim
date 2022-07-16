import
  unittest,
  arraymancer,
  sugar,
  ../../src/linear_models

suite "pinv":

  test "2D float":
    let 
      X = [[1, 2], [3, 4]].toTensor.astype(float)
      result = [[-2.0, 1.5], [1.0, -0.5]].toTensor.astype(float)
    check pinv(X).map(x => round(x, 5)) == result.map(x => round(x, 5))
