import sys

def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '\u2588' * filled_len + ' ' * (bar_len - filled_len)

    sys.stdout.write('[{}] {}{} ...{}\r'.format(bar, percents, '%', status))
    sys.stdout.flush()

import time

for i in range(10):
    progress(i,9,status="wow")
    time.sleep(0.5)
print("\n")