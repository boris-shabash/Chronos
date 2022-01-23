# Copyright (c) Boris Shabash

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

monthday_names_ = ["1st", "2nd", "3rd"]
monthday_names_.extend([f"{i}th" for i in range(4, 21)])
monthday_names_.extend(["21st", "22nd", "23rd"])
monthday_names_.extend([f"{i}th" for i in range(24, 31)])
monthday_names_.append("31st")


weekday_names_ = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


Normal_dist_code = "Normal"
StudentT_dist_code = "StudentT"
LogNormal_dist_code = "LogNormal"
Gamma_dist_code = "Gamma"
Poisson_dist_code = "Poisson"
HalfNormal_dist_code = "HalfNormal"
Beta_dist_code = "Beta"

#SUPPORTED_DISTRIBUTIONS = [Normal_dist_code, StudentT_dist_code, Gamma_dist_code]
SIGMOID_CONSTRAINED_DISTRIBUTIONS = (Beta_dist_code,)
POSITIVE_DISTRIBUTIONS = (Gamma_dist_code, LogNormal_dist_code)
UNCONSTRAINED_DISTRIBUTIONS = (Normal_dist_code, StudentT_dist_code)

SUPPORTED_DISTRIBUTIONS = POSITIVE_DISTRIBUTIONS + UNCONSTRAINED_DISTRIBUTIONS + SIGMOID_CONSTRAINED_DISTRIBUTIONS


POINT_ESTIMATE_METHODS = ("MAP", "MLE")
DISTRIBUTION_ESTIMATE_METHODS = ("SVI", "MCMC")
SUPPORTED_METHODS = POINT_ESTIMATE_METHODS + DISTRIBUTION_ESTIMATE_METHODS

PRIOR_BASED_METHODS = list(SUPPORTED_METHODS).copy()
PRIOR_BASED_METHODS.remove("MLE")
PRIOR_BASED_METHODS = tuple(PRIOR_BASED_METHODS)

OPTIMIZATION_BASED_METHODS = list(SUPPORTED_METHODS).copy()
OPTIMIZATION_BASED_METHODS.remove("MCMC")
OPTIMIZATION_BASED_METHODS = tuple(OPTIMIZATION_BASED_METHODS)