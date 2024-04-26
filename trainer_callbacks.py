# MIT License
#
# Copyright (c) 2024 Mohammad Zunaed, mHealth Lab, BUET
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from tqdm import tqdm
import numpy as np
import os
import torch
import random
from sklearn.metrics import roc_auc_score

def set_random_state(seed_value: int):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.random.manual_seed(seed_value)
    
class ProgressBar():
    def __init__(self, steps: int, description:str = None):
        '''initiate the progbar with total steps, optional: set description'''
        self.steps = steps
        self.description = description        
        self.progbar = tqdm(total=self.steps, unit=' steps')
        self.progbar.set_description(description)
    def update(self, increment_count=None, logs_to_display: dict = None):
        '''increment counter, update displayed info'''
        if logs_to_display is not None:
            logs_to_display = {key: '%.06f' % logs_to_display[key] for key in logs_to_display.keys()}
            self.progbar.set_postfix(logs_to_display)
        if increment_count is not None:
            self.progbar.update(increment_count)
    def close(self, logs_to_display: dict = None):
        '''close the progbar, optional: update displayed info'''
        if logs_to_display is not None:
            logs_to_display = {key: '%.06f' % logs_to_display[key] for key in logs_to_display.keys()}
            self.progbar.set_postfix(logs_to_display)
        self.progbar.close()
        
class AverageMeter():
    def __init__(self):
        self.reset()
    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, inp: list):
        score = inp[0]
        n = inp[1]
        self.sum += score * n
        self.count += n
        self.avg = self.sum / self.count
    def feedback(self):
        return self.avg
    
class PrintMeter():
    def __init__(self):
        self.reset()
    def reset(self):
        self.value = 0
    def update(self, inp):
        self.value = inp
    def feedback(self):
        return self.value
    
class MetricStoreBox():
    def __init__(self, metrics:dict):
        self.metrics = metrics
        self.metric_functions = {}
        for key in metrics.keys():
            self.metric_functions.update({key:metrics[key]()})
    def update(self, info_dict:dict):
        for key in info_dict.keys():
            self.metric_functions[key].update(info_dict[key])
    def get_value(self):
        logs = {}
        for key in self.metrics.keys():
            logs.update({key:self.metric_functions[key].feedback()})
        return logs
    
class ExtraMetricMeter():
    def __init__(self):
        self.reset()
    def reset(self):
        self.y_true = []
        self.y_score = []
    def update(self, y_score: np.array, y_true: np.array):
        self.y_true.append(y_true)
        self.y_score.append(y_score)
    def feedback(self):
        y_true = np.concatenate(self.y_true)
        y_score = np.concatenate(self.y_score)       
        return roc_auc_score(y_true, y_score)