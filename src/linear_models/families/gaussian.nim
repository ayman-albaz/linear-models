import 
  base,
  arraymancer,
  math_utils


type 
  Gaussian* = object of Family


proc link*[T: SomeFloat](self: Gaussian, X: Tensor[T]): Tensor[T] = 
  result = X


proc inverseLink*[T: SomeFloat](self: Gaussian, X: Tensor[T]): Tensor[T] = 
  result = X


proc gradientInverseLink*[T: SomeFloat](self: Gaussian, X: Tensor[T]): Tensor[T] = 
  result = ones[T](X.shape[0], X.shape[1])


proc variance*[T: SomeFloat](self: Gaussian, X: Tensor[T]): Tensor[T] = 
  result = ones[T](X.shape[0], X.shape[1])


proc dispersion*[T: SomeFloat](self: Gaussian, residuals: Tensor[T], degreesFreedom: T): T = 
  result = dispersion(residuals, degreesFreedom)
