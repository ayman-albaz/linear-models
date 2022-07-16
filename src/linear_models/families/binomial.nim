import 
  base,
  arraymancer,
  mathutils,
  sugar


type 
  Binomial* = object of Family


proc link*[T: SomeFloat](self: Binomial, X: Tensor[T]): Tensor[T] = 
  result = logit(X)


proc inverseLink*[T: SomeFloat](self: Binomial, X: Tensor[T]): Tensor[T] = 
  result = logistic(X)


proc gradientInverseLink*[T: SomeFloat](self: Binomial, X: Tensor[T]): Tensor[T] = 
  result = gradientLogistic(X)


proc variance*[T: SomeFloat](self: Binomial, X: Tensor[T]): Tensor[T] = 
  result = map(X, x => x * (1.T - x))


proc dispersion*[T: SomeFloat](self: Binomial, residuals: Tensor[T], degreesFreedom: T): T = 
  result = 1.T
