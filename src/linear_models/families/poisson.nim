import 
  base,
  arraymancer


type 
  Poisson* = object of Family


proc link*[T: SomeFloat](self: Poisson, X: Tensor[T]): Tensor[T] = 
  result = ln(X)


proc inverseLink*[T: SomeFloat](self: Poisson, X: Tensor[T]): Tensor[T] = 
  result = exp(X)


proc gradientInverseLink*[T: SomeFloat](self: Poisson, X: Tensor[T]): Tensor[T] = 
  result = exp(X)


proc variance*[T: SomeFloat](self: Poisson, X: Tensor[T]): Tensor[T] = 
  result = X


proc dispersion*[T: SomeFloat](self: Poisson, residuals: Tensor[T], degreesFreedom: T): T = 
  result = 1.T
