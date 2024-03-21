from datetime import timezone

import pandas as pd

EPOCH_TIME_UTC = pd.Timestamp(year=1970, month=1, day=1, tzinfo=timezone.utc)
EPOCH_TIME_NO_TZ = pd.Timestamp(year=1970, month=1, day=1)

ZERO_DELTA = pd.Timedelta(nanoseconds=0)
ONE_NANOSECOND = pd.Timedelta(nanoseconds=1)
ONE_SECOND = pd.Timedelta(seconds=1)
ONE_DAY = pd.Timedelta(days=1)
