import 
  unittest,
  arraymancer,
  sugar,
  ../../src/linear_models

suite "blas":

    test "backSolve rowMajor":
      let
        X = [[1, 2, 3], [0, 1, 1], [0, 0, 2]].toTensor().astype(float).asContiguous(rowMajor, force=true)
        B = [[8, 4, 2]].toTensor().astype(float).asContiguous(rowMajor, force=true)
        result = [[-1.0], [3.0], [1.0]].toTensor()
      check backSolve(X, B.transpose).map(x => round(x, 5)) == result.map(x => round(x, 5))

    test "backSolve colMajor":
      let
        X = [[1, 2, 3], [0, 1, 1], [0, 0, 2]].toTensor().astype(float).asContiguous(colMajor, force=true)
        B = [[8, 4, 2]].toTensor().astype(float).asContiguous(colMajor, force=true)
        result = [[-1.0], [3.0], [1.0]].toTensor()
      check backSolve(X, B.transpose).map(x => round(x, 5)) == result.map(x => round(x, 5))

    test "forwardSolve rowMajor":
      let
        X = [[1, 2, 3], [0, 1, 1], [0, 0, 2]].toTensor().astype(float).asContiguous(rowMajor, force=true)
        B = [[8, 4, 2]].toTensor().astype(float).asContiguous(rowMajor, force=true)
        result = [[8.0], [4.0], [1.0]].toTensor()
      check forwardSolve(X, B.transpose).map(x => round(x, 5)) == result.map(x => round(x, 5))

    test "forwardSolve colMajor":
      let
        X = [[1, 2, 3], [0, 1, 1], [0, 0, 2]].toTensor().astype(float).asContiguous(colMajor, force=true)
        B = [[8, 4, 2]].toTensor().astype(float).asContiguous(colMajor, force=true)
        result = [[8.0], [4.0], [1.0]].toTensor()
      check forwardSolve(X, B.transpose).map(x => round(x, 5)) == result.map(x => round(x, 5))

    test "forwardSolve rowMajor edge case":
      let
        X = [[0.39179338, 0.0], [0.08218041, 0.3601183]].toTensor().asContiguous(rowMajor, force=true)
        B = [[1.365386, -1.821262]].toTensor().asContiguous(rowMajor, force=true)
        result = [[3.484964447331907], [-5.852681763512599]].toTensor()
      check forwardSolve(X, B.transpose).map(x => round(x, 5)) == result.map(x => round(x, 5))

    test "backSolve rowMajor edge case":
      let
        X = [[0.3917933847329501, 0.0], [0.08218040550578956, 0.3601183155443933]].toTensor().asContiguous(rowMajor, force=true)
        B = [[3.484964928125221], [-5.852680425936701]].toTensor().asContiguous(rowMajor, force=true)
        result = [[12.30386], [-16.25210]].toTensor()
      check backSolve(X.transpose, B).map(x => round(x, 5)) == result.map(x => round(x, 5))
