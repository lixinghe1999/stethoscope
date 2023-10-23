import wave
import json
# from progressbar import *
import numpy as np

EPS = 1e-12

# from pynvml import *

def get_framesLength(fname):
    with wave.open(fname) as f:
        params = f.getparams()
    return params[3]


def write_json(my_dict, fname):
    json_str = json.dumps(my_dict, indent=4)
    with open(fname, "w") as json_file:
        json_file.write(json_str)


def dict_mean(dict_list):
    ret_val = {}
    for k in dict_list[0].keys():
        ret_val[k] = np.mean([v[k] for v in dict_list])
    return ret_val


def load_json(fname):
    with open(fname, "r") as f:
        data = json.load(f)
        return data


def get_sample_rate(fname):
    with wave.open(fname) as f:
        params = f.getparams()
    return params[2]


def to_log(input):
    return np.log10(input + 1e-12)


def from_log(input):
    input = np.clip(input, min=-np.inf, max=5)
    return 10**input


def write_list(list, fname):
    with open(fname, "w") as f:
        for word in list:
            f.write(word)
            f.write("\n")


def read_list(fname):
    result = []
    with open(fname, "r") as f:
        for each in f.readlines():
            each = each.strip("\n")
            result.append(each)
    return result


def pow_p_norm(signal):
    """Compute 2 Norm"""
    signal = signal.reshape(-1)
    return np.linalg.norm(signal, ord=2, keepdims=True)**2


def energy_unify(estimated, original):
    target = pow_norm(estimated, original) * original
    target /= pow_p_norm(original) + EPS
    return estimated, target


def pow_norm(s1, s2):
    return np.sum(s1 * s2, keepdims=True)