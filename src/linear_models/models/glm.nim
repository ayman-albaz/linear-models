import
  arraymancer,
  distributions,
  times,
  ../families/base, 
  ../utils/tensorutils,
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
    

proc glm*[T: float, S: Family](
  X, y: Tensor[T],
  family: S,
  maxIter: int = 200,
  tolerance: T = 1e-8,
  confidenceIntervalAlpha: float = 0.05,
  useZ: bool = true
): GlmSummary = 
  ##[
    If useZ, use Z distribution for p-value and confidence interval
    else, use t distribution for p-value and confidence interval.
    Recommendation is to set useZ to `true` when number of rows >= 40,
    otherwise set useZ to `false` 
  ]##

  # Init variables
  let t0 = getTime()
  let fitSummary = iterativelyReweightedLeastSquares(X, y, family, maxIter, tolerance)
  let t1 = getTime()
  var normDist = initNormalDistribution()
  var tDist = initTDistribution(1)
  var crit: float
  case useZ:
    of true: 
      normDist = initNormalDistribution(0.0, 1.0)
      crit = normDist.ppf(1.0 - confidenceIntervalAlpha / 2.0)
    of false:
      tDist = initTDistribution(X.shape[0])
      crit = tDist.ppf(1.0 - confidenceIntervalAlpha / 2.0)
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
    statsRow.pValue = case useZ:
      of true: 2.0 * (1.0 - normDist.cdf(abs(statsRow.zScore)))
      of false: 2.0 * (1.0 - tDist.cdf(abs(statsRow.zScore)))
    statsRow.pValue = 2.0 * (1.0 - normDist.cdf(abs(statsRow.zScore)))
    statsRow.confidenceIntervalLower = statsRow.coefficient - statsRow.standardError * crit
    statsRow.confidenceIntervalUpper = statsRow.coefficient + statsRow.standardError * crit
    statsTable.statsRows.add(statsRow)

  # Create statsSummary
  let statsSummary = StatsSummary(hasConverged: fitSummary.totalIter < fitSummary.maxIter,
                                  startTime: t0, endTime: t1,
                                  statsTable: statsTable)

  # Create glmSummary
  result = GlmSummary(fitSummary: fitSummary, statsSummary: statsSummary)
