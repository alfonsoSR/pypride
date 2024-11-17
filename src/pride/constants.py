from astropy import time


J2000: time.Time = time.Time("2000-01-01T12:00:00", scale="tt").utc  # type: ignore
