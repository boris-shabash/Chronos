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
<<<<<<< HEAD
Poisson_dist_code = "Poisson"
HalfNormal_dist_code = "HalfNormal"
SUPPORTED_DISTRIBUTIONS = [Normal_dist_code, StudentT_dist_code, Gamma_dist_code, Poisson_dist_code, HalfNormal_dist_code]
=======
SUPPORTED_DISTRIBUTIONS = [Normal_dist_code, StudentT_dist_code, Gamma_dist_code]
>>>>>>> cd9d0d9cc86748da578d8e7b46d3667ccd4ea723


SUPPORTED_METHODS = ["MAP", "MLE"]