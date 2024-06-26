import numpy as np
import torch
from unidecode import unidecode

#This function returns geometric mean of a list
def g_mean(list_):
    log_list_ = np.log(list_)
    return np.exp(log_list_.mean())


#This function transforms the LIKE patterns to new language
def LIKE_pattern_to_newLanguage(pattern_list, pattern_type):
    transformed_pattern = ''
    for pattern in pattern_list:
        if len(pattern) == 1:
            transformed_pattern += pattern
        else:
            new_pattern = ''
            count = 0
            for char in pattern:
                if count < 1:
                    new_pattern += char
                    count += 1
                else:
                    if (
                        new_pattern[-1] not in ('_', '@')
                        and char not in ('_', '@')
                    ):
                        new_pattern += char + '$'
                    else:
                        new_pattern += char
            transformed_pattern += new_pattern
    if pattern_type == 'prefix':
        transformed_pattern = f'{transformed_pattern[0]}.{transformed_pattern[1:]}'
    elif pattern_type == 'suffix':
        transformed_pattern += '#'
    elif pattern_type == 'end_underscore':
        transformed_pattern = (f'{transformed_pattern[0]}.{transformed_pattern[1:]}')
    elif pattern_type == 'begin_underscore':
        transformed_pattern = (f'{transformed_pattern[0]}{transformed_pattern[1:]}#')
    elif pattern_type == 'prefix_suffix':
        transformed_pattern = (f'{transformed_pattern[0]}.{transformed_pattern[1:]}#')
    return transformed_pattern


#This function computes loss
def binary_crossentropy(preds, targets, mask):
    loss = targets * torch.log(preds + 0.00001) + (1 - targets) * torch.log(1 - (preds - 0.00001))
    if mask is not None:
        loss = mask * loss
    return - torch.sum(loss) / torch.sum(mask)


def name2tensor(name, char2idx):
    tensor = torch.zeros(len(name), len(char2idx))
    for i, char in enumerate(name):
        tensor[i][char2idx[char]] = 1
    return tensor


#This function loads LIKE-patterns with ground truth probabilities
def loadtrainData(filename, char2idx):
    inputs = []
    targets = []
    length = []
    count = 0
    for line in open(filename):
        line_ = line.strip().split(':')
        transformedLikepattern = LIKE_pattern_to_newLanguage(line_[0].split('%'), line_[1])
        transformed_to_tensor = name2tensor(unidecode(transformedLikepattern), char2idx)
        inputs.append(transformed_to_tensor)
        length.append(len(transformed_to_tensor))
        ground_prob_list = [float(element) for element in line_[-1].split(' ')]
        targets.append(ground_prob_list)
        count +=1
        if count == 300000:
            break
    return inputs, targets, max(length)

#This function pads the zero vectors to like-patterns.
def addpaddingTrain(filename, char2idx):
    zeros_vector = [[0] * len(char2idx)]
    padded_inputs = []
    inputs, targets, maxs = loadtrainData(filename,char2idx)
    for i in inputs:
        old_len = len(i)
        for k in range(maxs - len(i)):
            i = torch.cat((i, torch.tensor(zeros_vector)), 0)
        padded_inputs.append((i, [1] * old_len + [0] * (maxs - old_len)))
    targets1 = []
    for i in targets:
        targets1.append(i + (maxs - len(i)) * [0])
    train_dataset = [(torch.tensor(padded_inputs[i][0]), torch.tensor(padded_inputs[i][1]), torch.tensor(targets1[i])) for i in
                     range(len(targets))]
    return train_dataset


#This function takes a file path that contains test LIKE-patterns
def loadtestData(filename, char2idx):
    inputs = []
    length = []
    actual_card = []
    count = 0
    with open(filename, 'r') as file:
        for line in file:
            line_ = line.strip().split(':')
            actual_card.append(float(line_[-1]))
            transformedLikepattern = LIKE_pattern_to_newLanguage(line_[0].replace(' ', '@').split('%'), line_[1])
            transformed_to_tensor = name2tensor(unidecode(transformedLikepattern), char2idx)
            inputs.append(transformed_to_tensor)
            length.append(len(transformed_to_tensor))
            count +=1
            if count ==100000:
                break
    return inputs, max(length), actual_card



def addpaddingTest(filename, char2idx):
    liste = [[0] * len(char2idx)]
    padded_inputs = []
    masks = []
    inputs, maxs, actual_card = loadtestData(filename, char2idx)
    for i in inputs:
        old_len = len(i)
        for k in range(maxs - len(i)):
            i = torch.cat((i, torch.tensor(liste)), 0)
        padded_inputs.append(i)
        masks.append([1] * old_len + [0] * (maxs - old_len))
    test_dataset = [(torch.tensor(padded_inputs[i]), torch.tensor(masks[i]), torch.tensor(actual_card[i])) for i in range(len(masks))]
    return test_dataset


##This function compute and return q-error
def compute_qerrors(actual_card, estimated_card):
    return max(actual_card/estimated_card, estimated_card/actual_card)
