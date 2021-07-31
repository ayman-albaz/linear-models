import 
  base,
  arraymancer,
  math_utils


type 
  Gamma* = object of Family


proc link*[T: SomeFloat](self: Gamma, X: Tensor[T]): Tensor[T] = 
  result = 1.T /. X


proc inverseLink*[T: SomeFloat](self: Gamma, X: Tensor[T]): Tensor[T] = 
  result = 1.T /. X


proc gradientInverseLink*[T: SomeFloat](self: Gamma, X: Tensor[T]): Tensor[T] = 
  result = -1.T /. (X ^. 2.T)


proc variance*[T: SomeFloat](self: Gamma, X: Tensor[T]): Tensor[T] = 
  result = X ^. 2


proc dispersion*[T: SomeFloat](self: Gamma, residuals: Tensor[T], degreesFreedom: T): T = 
  result = dispersion(residuals, degreesFreedom)
