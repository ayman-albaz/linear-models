import 
  unittest,
  arraymancer,
  sugar,
  ../../src/linear_models

suite "lapack":

  test "choleskyDecomposition symmetric rowMajor lower":
    let 
      X = [[2, -1, 0], [-1, 2, -1], [0, -1, 2]].toTensor.astype(float).asContiguous(rowMajor, force=true)
      result = [[1.414213562373095, 0.0, 0.0],
                  [-0.7071067811865475, 1.224744871391589, 0.0],
                  [0.0, -0.8164965809277261, 1.154700538379251]].toTensor()
    check choleskyDecomposition(X, "L").map(x => round(x, 5)) == result.map(x => round(x, 5))

  test "choleskyDecomposition symmetric rowMajor upper":
    let 
      X = [[2, -1, 0], [-1, 2, -1], [0, -1, 2]].toTensor.astype(float).asContiguous(rowMajor, force=true)
      result = [[1.414213562373095, -0.7071067811865475, 0.0],
                  [0.0, 1.224744871391589, -0.8164965809277261],
                  [0.0, 0.0, 1.154700538379251]].toTensor()
    check choleskyDecomposition(X, "U").map(x => round(x, 5)) == result.map(x => round(x, 5))

  test "choleskyDecomposition symmetric colMajor lower":
    let 
      X = [[2, -1, 0], [-1, 2, -1], [0, -1, 2]].toTensor.astype(float).asContiguous(colMajor, force=true)
      result = [[1.414213562373095, 0.0, 0.0],
                  [-0.7071067811865475, 1.224744871391589, 0.0],
                  [0.0, -0.8164965809277261, 1.154700538379251]].toTensor()
    check choleskyDecomposition(X, "L").map(x => round(x, 5)) == result.map(x => round(x, 5))

  test "choleskyDecomposition symmetric colMajor upper":
    let 
      X = [[2, -1, 0], [-1, 2, -1], [0, -1, 2]].toTensor.astype(float).asContiguous(colMajor, force=true)
      result = [[1.414213562373095, -0.7071067811865475, 0.0],
                  [0.0, 1.224744871391589, -0.8164965809277261],
                  [0.0, 0.0, 1.154700538379251]].toTensor()
    check choleskyDecomposition(X, "U").map(x => round(x, 5)) == result.map(x => round(x, 5))