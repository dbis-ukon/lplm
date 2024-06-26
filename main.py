import time
from collections import namedtuple

from pympler import asizeof
from torch.utils.data import DataLoader
import torch
import misc_utils
import selectivity_estimator
import numpy as np


# This function load the saved model
def load_estimation_model(model_file_name, model_card, device):
    model_card.load_state_dict(torch.load(model_file_name))
    return model.to(device)


# This function trains and returns the embedding model
def modelTrain(train_data, model, device, learning_rate, num_epocs, model_save_path):
    model = selectivity_estimator.train_model(train_data, model, device, learning_rate, num_epocs)
    torch.save(model.state_dict(), model_save_path)
    return model


# This function estimate the cardinalities and saves them to a txt file
def estimate_cardinality(test_dataset, model, device, save_file_path, dataset_size):
    write_to_file = open(save_file_path, 'w')
    qerror_list = []
    with torch.no_grad():
        for name, mask, actual_card in test_dataset:
            name = name.to(device)
            output = model(name)
            output = output.to(device)
            mask = mask.to(device)
            output = torch.prod(torch.pow(output, mask)) * dataset_size
            qerror = misc_utils.compute_qerrors(actual_card, output.item())
            qerror_list.append(qerror[0])
            write_to_file.write(str(output.item()) + '\n')

    print(f'G-mean: {np.round(misc_utils.g_mean(qerror_list), 4)}')
    print(f'Mean: {np.round(np.average(qerror_list), 4)}')
    print(f'Median: {np.round(np.percentile(qerror_list, 50), 4)}')
    print(f'90th: {np.round(np.percentile(qerror_list, 90), 4)}')
    print(f'99th: {np.round(np.percentile(qerror_list, 99), 4)}')


def get_vocab(trainfile):

    vocab_dict = {}
    for i in open(trainfile):
        i=i.strip().split(':')[0]
        for k in i:
            if k != '%' and k not in vocab_dict:
                vocab_dict[k] = 0
    vocab = ''
    for token in vocab_dict:
        vocab += token
    return vocab + '$.#'



if __name__ == "__main__":
    vocabulary =get_vocab('author_train.txt')
    trainpath = 'author_train.txt'
    testpath = 'author_test.txt'
    savepath = 'result.txt'
    savemodel= 'model.pth'
    A_NLM_configs = namedtuple('A_NLM_configs', ['vocabulary', 'hidden_size', 'learning_rate', 'batch_size', 'datasize',
                                                 'num_epocs', 'train_data_path', 'test_data_path',
                                                 'save_qerror_file_path', 'device', 'save_path'])

    card_estimator_configs = A_NLM_configs(vocabulary= vocabulary, hidden_size=256,
                                           datasize=450000, learning_rate=0.0001, batch_size=128, num_epocs=64,
                                           train_data_path=trainpath,
                                           test_data_path=testpath,
                                           save_qerror_file_path=savepath,
                                           device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                                           save_path=savemodel)
    char2idx = {letter: i for i, letter in enumerate(card_estimator_configs.vocabulary)}

    model = selectivity_estimator.Cardinality_Estimator(1, card_estimator_configs.hidden_size, card_estimator_configs.device,
                                              len(char2idx))
    train_data = misc_utils.addpaddingTrain(card_estimator_configs.train_data_path, char2idx)
    dataloadertrain = DataLoader(train_data, batch_size=card_estimator_configs.batch_size, shuffle=True)
    trained_model = modelTrain(dataloadertrain, model, card_estimator_configs.device,
                               card_estimator_configs.learning_rate, card_estimator_configs.num_epocs,
                               card_estimator_configs.save_path)
    datasettest = misc_utils.addpaddingTest(card_estimator_configs.test_data_path, char2idx)
    dataloadertest = DataLoader(datasettest, batch_size=1)

    estimate_cardinality(dataloadertest, trained_model, card_estimator_configs.device,
                         card_estimator_configs.save_qerror_file_path, card_estimator_configs.datasize)
