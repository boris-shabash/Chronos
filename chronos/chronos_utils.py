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
Gamma_dist_code = "Gamma"

POSITIVE_DISTRIBUTIONS = [Gamma_dist_code]
UNCONSTRAINED_DISTRIBUTIONS = [Normal_dist_code, StudentT_dist_code]

SUPPORTED_DISTRIBUTIONS = POSITIVE_DISTRIBUTIONS + UNCONSTRAINED_DISTRIBUTIONS


POINT_ESTIMATE_METHODS = ["MAP", "MLE"]
DISTRIBUTION_ESTIMATE_METHODS = ["SVI"]
SUPPORTED_METHODS = POINT_ESTIMATE_METHODS + DISTRIBUTION_ESTIMATE_METHODS