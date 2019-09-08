import numpy as np

alpha2num = lambda c: ord(c) - ord('A') + 1

def read_from_file(path):
    with open(path, 'r') as file:
        data = file.read()

    return data

def split_data(lines):
    lines = lines.splitlines()
    data = np.empty([0, 6])
    for line in lines:
        row = list(line)
        row[5] = alpha2num(row[5])
        data = np.append(data, np.array([row]), axis = 0)
    return data

def split_point(count):
    return count - round(count / 5)

def prepare_data(data, count):
    learn, test = np.split(data, [count], 0)
    learn_x, learn_y = np.split(learn, [5], 1)
    test_x, test_y = np.split(test, [5], 1)
    return learn_x, learn_y, test_x, test_y

def do():
    lines = read_from_file("./rawdata")
    data = split_data(lines)
    learn_count = split_point(data.shape[0])
    return prepare_data(data, learn_count)
