import
  arraymancer,
  distributions,
  times,
  ../families/base, 
  ../linear_algebra/pinv,
  ../utils/tensor_utils,
  optimizers


type

  StatsRow* = object
    coefficient*, standardError*, zScore*, pValue*, confidenceIntervalLower*, confidenceIntervalUpper*: float

  StatsTable* = object
    statsRows*: seq[StatsRow]

  StatsSummary* = object
    hasConverged*: bool
    startTime*, endTime*: Time
    statsTable*: StatsTable

  GlmSummary* = object
    fitSummary*: FitSummary
    statsSummary*: StatsSummary
    

proc glm*[T: float, S: Family](X, y: Tensor[T],
                               family: S,
                               maxIter: int = 200,
                               tolerance: T = 1e-8,
                               confidenceIntervalAlpha: float = 0.05): GlmSummary = 

  # Init variables
  let 
    t0 = getTime()
    fitSummary = iterativelyReweightedLeastSquares(X, y, family, maxIter, tolerance)
    t1 = getTime()
    normDist = initNormalDistribution(0.0, 1.0)
    zCrit = normDist.ppf(1.0 - confidenceIntervalAlpha / 2.0)  #TODO: Change to TDIST for N<40
  var
    statsRow: StatsRow
    statsTable: StatsTable
    standardErrors: Tensor[T]
    mse: T

  # Get error
  mse = sum(squaredError(family.inverseLink(X * fitSummary.coefficients.transpose), y.reshape1DColTo2D)) / (X.shape[0] - X.shape[1]).float
  standardErrors = sqrt(diagonal(fitSummary.covariance))

  # Create statsTable
  for i, coefficient in enumerate(fitSummary.coefficients):
    statsRow.coefficient = coefficient
    statsRow.standardError = standardErrors[i]
    statsRow.zScore = statsRow.coefficient / statsRow.standardError
    statsRow.pValue = 2.0 * (1.0 - normDist.cdf(abs(statsRow.zScore)))
    statsRow.confidenceIntervalLower = statsRow.coefficient - statsRow.standardError * zCrit
    statsRow.confidenceIntervalUpper = statsRow.coefficient + statsRow.standardError * zCrit
    statsTable.statsRows.add(statsRow)

  # Create statsSummary
  let statsSummary = StatsSummary(hasConverged: fitSummary.totalIter < fitSummary.maxIter,
                                  startTime: t0, endTime: t1,
                                  statsTable: statsTable)

  # Create glmSummary
  result = GlmSummary(fitSummary: fitSummary, statsSummary: statsSummary)
