import sys
import math
import time
from tensorflow.keras.callbacks import Callback

"""
This code(AnimatedLogger) is strongly derived from ProgbarLogger in https://github.com/tensorflow/tensorflow/blob/r2.0/tensorflow/python/keras/callbacks.py#L693-L768
"""


class AnimatedLogger(Callback):
    """Callback that prints metrics to stdout.
    Arguments:
        count_mode: One of "steps" or "samples".
            Whether the progress bar should
            count samples seen or steps (batches) seen.
    Raises:
        ValueError: In case of invalid `count_mode`.
    """

    def __init__(self, count_mode='samples'):
        super(AnimatedLogger, self).__init__()
        if count_mode == 'samples':
            self.use_steps = False
        elif count_mode == 'steps':
            self.use_steps = True
        else:
            raise ValueError('Unknown `count_mode`: ' + str(count_mode))
        self.data_dict = {}

    def on_train_begin(self, logs=None):
        self.verbose = self.params['verbose']
        self.epochs = self.params['epochs']

    def on_epoch_begin(self, epoch, logs=None):
        self.seen = 0
        if self.use_steps:
            self.target = self.params['steps']
        else:
            self.target = self.params['samples']

        self.start_time = time.time()

        print('Epoch %d/%d' % (epoch + 1, self.epochs))

    def on_batch_begin(self, batch, logs=None):
        self.log_values = []

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        batch_size = logs.get('size', 0)

        num_steps = logs.get('num_steps', 1)
        if self.use_steps:
            self.seen += num_steps
        else:
            self.seen += batch_size * num_steps

        for k in self.params['metrics']:
            if k in logs:
                self.log_values.append((k, logs[k]))

        steps = "{}/{}".format(self.seen, self.target)
        bar = progress_bar(self.seen/self.target)
        eta = "ETA: {:.2f} secs".format(
            (time.time()-self.start_time)/(self.seen/self.target)+self.start_time-time.time())
        metrics = ""
        for metric in self.log_values:
            metrics += "{}: {:.5f} | ".format(metric[0], metric[1])
        out = f"\r{steps} {bar} {eta} | {metrics}" + 10 * " "
        sys.stdout.write(out)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for k in self.params['metrics']:
            if k in logs:
                self.log_values.append((k, logs[k]))
        for name, value in self.log_values:
            if epoch == 0:
                self.data_dict[name] = [value]
            else:
                if epoch == len(self.data_dict[name]):
                    self.data_dict[name].append(value)
        steps = "{}/{}".format(self.seen, self.target)
        bar = progress_bar(self.seen/self.target)
        eta = "{:.2f} secs ".format(
            (time.time()-self.start_time)/(self.seen/self.target))
        metrics = ""
        for metric in self.log_values:
            if not (metric[0] in metrics):
                metrics += "{}: {:.5f} | ".format(metric[0], metric[1])
        out = f"\r{steps} {bar} {eta} | {metrics}" + 10 * " "
        sys.stdout.write(out)

        print("")
        graph_all_mats(self.data_dict, epoch+1, width=20)

def progress_bar(progress, width=50):
    return "[" + "\u2588" * int(progress*width) + " " * (width-int(progress*width)) + "]"


def bar(progress, width=10):
    progress = min(1, max(0, progress))
    whole = math.floor(progress * width)
    remainder_width = (progress * width) % 1
    pw = math.floor(remainder_width * 8)
    pc = [" ", "▏", "▎", "▍", "▌", "▋", "▊", "▉"][pw]
    if (width - whole - 1) < 0:
        pc = ""
    return "" + "█" * whole + pc + " " * (width - whole - 1)

def graph_mat(title, values, width=10):
    maximum = max(values)
    mat = []
    mat.append(title[:width]+" "*max(0,width-len(title[:width])))
    for value in values:
        mat.append(bar(value/maximum, width=width))
    return mat


def graph_all_mats(data_dict, iters, width=10):
    mats = []
    for key, data in data_dict.items():
        mats.append(graph_mat(key, data, width=width))

    for l in range(iters+1):
        row = ""
        for mat in mats:
            if len(mat) > l:
                row += "|"+mat[l]+"\t"
            else:
                row += "|"+width*" " + "\t"
        print(row)
