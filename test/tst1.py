import holidays
us_holidays = holidays.US(years=range(2002, 2019))
clusters["is_holiday"] = pd.to_datetime(clusters.index).isin(us_holidays)
clusters[clusters["day_of_week"] < 5].groupby("cluster")["is_holiday"].mean()