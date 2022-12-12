import datetime

__all__ = ['today', 'tomorrow', 'this_week', 'next_week', 'this_month', 'next_month', 'until', 'time_as_str']


def today(hour=24, minute=0, second=0):
    if hour < 24:
        return datetime.datetime.combine(datetime.date.today(), datetime.time(hour, minute, second))
    else:
        return datetime.datetime.combine(datetime.date.today() + datetime.timedelta(days=1),
                                         datetime.time(hour - 24, minute, second))


def tomorrow(hour=0, minute=0, second=0):
    return datetime.datetime.combine(datetime.date.today() + datetime.timedelta(days=1),
                                     datetime.time(hour, minute, second))


def this_week(day=7, hour=0, minute=0, second=0):
    if day < datetime.date.today().weekday():
        day += 7
    return datetime.datetime.combine(
        datetime.date.today() + datetime.timedelta(days=day - datetime.date.today().weekday()),
        datetime.time(hour, minute, second))


def next_week(day=7, hour=23, minute=59, second=59):
    return this_week(7 + day, hour, minute, second)


def this_month(day=1, hour=0, minute=0, second=0):
    if day < datetime.date.today().day:
        date = (datetime.date.today().replace(day=28) + datetime.timedelta(days=4)).replace(day=day)
    else:
        date = datetime.date.today().replace(day=day)
    return datetime.datetime.combine(date, datetime.time(hour, minute, second))


def next_month(day=None, hour=0, minute=0, second=0):
    date = (datetime.date.today().replace(day=28) + datetime.timedelta(days=4)).replace(day=1)
    if day is not None:
        date = date.replace(day=day)
    return datetime.datetime.combine(date, datetime.time(hour, minute, second))


def until(t: datetime.datetime):
    return (t-datetime.datetime.now()).total_seconds()


def time_as_str(t):
    if t < 60:
        return f"{t:.2f} seconds"
    elif t < 60*60:
        return f"{t/60:.2f} minutes"
    elif t < 60*60*24:
        return f"{t/60/60:.2f} hours"
    else:
        return f"{t/60/60/24:.2f} days"
