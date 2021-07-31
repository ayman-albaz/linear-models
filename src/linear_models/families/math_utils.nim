import 
  arraymancer,
  ../utils/constants,
  sugar


proc logit*[T: SomeFloat](X: Tensor[T]): Tensor[T] {.inline.} = 
  result = ln(X /. (1.T -. X))


proc logistic*[T: SomeFloat](X: Tensor[T]): Tensor[T] {.inline.} = 
  result =  1.T /. (1.T +. exp(-X))


proc gradientLogistic*[T: SomeFloat](X: Tensor[T]): Tensor[T] {.inline.} = 
  result = map(logistic(X) *. (1.T -. logistic(X)), x => max(x, EPS))


proc inverse*[T: SomeFloat](X: Tensor[T]): Tensor[T] {.inline.} = 
  result =  1.T /. X


proc dispersion*[T: SomeFloat](residuals: Tensor[T], degreesFreedom: T): T = 
  result = sum(ones[T](residuals.shape[0], 1) *. residuals ^. 2.0) / degreesFreedom