monthday_names_ = ["1st", "2nd", "3rd"]
monthday_names_.extend([f"{i}th" for i in range(4, 21)])
monthday_names_.extend(["21st", "22nd", "23rd"])
monthday_names_.extend([f"{i}th" for i in range(24, 31)])
monthday_names_.append("31st")


weekday_names_ = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


Normal_dist_code = "Normal"
StudentT_dist_code = "StudentT"
SUPPORTED_DISTRIBUTIONS = [Normal_dist_code, StudentT_dist_code]


SUPPORTED_METHODS = ["MAP", "MLE"]