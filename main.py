import time
from collections import namedtuple

from pympler import asizeof
from torch.utils.data import DataLoader
import torch
import misc_utils
import selectivity_estimator
import numpy as np


# This function loads a saved model from a file.
def load_estimation_model(model_file_name, model_card, device):
    model_card.load_state_dict(torch.load(model_file_name))
    return model.to(device)


# This function trains the embedding model with given data and parameters.
def modelTrain(train_data, model, device, learning_rate, num_epocs, model_save_path):
    model = selectivity_estimator.train_model(train_data, model, device, learning_rate, num_epocs)
    torch.save(model.state_dict(), model_save_path)
    return model


# This function estimates the cardinality of queries using the trained model.
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

# This function creates a vocabulary from the training data.
def get_vocab(train_file):

    vocab_dict = {}
    for i in open(train_file):
        i=i.strip().split(':')[0]
        for k in i:
            if k != '%' and k not in vocab_dict:
                vocab_dict[k] = 0
    vocab = ''
    for token in vocab_dict:
        vocab += token
    return vocab + '$.#'



if __name__ == "__main__":
    # Get the vocabulary from the training data.
    vocabulary = get_vocab('train.txt')

    # File paths for training and test data, and where to save results.
    trainpath = 'train.txt'
    testpath = 'test.txt'
    savepath = 'estimated_cardinalities.txt'
    savemodel = 'model.pth'  # Path to save the trained model.

    # Define configuration parameters for the cardinality estimator model.
    A_NLM_configs = namedtuple('A_NLM_configs', ['vocabulary', 'hidden_size', 'learning_rate', 'batch_size', 'datasize',
                                                 'num_epocs', 'train_data_path', 'test_data_path',
                                                 'save_qerror_file_path', 'device', 'save_path'])

    # Create a named tuple with the configuration values.
    card_estimator_configs = A_NLM_configs(vocabulary=vocabulary, hidden_size=256,
                                           datasize=450000, learning_rate=0.0001, batch_size=128, num_epocs=64,
                                           train_data_path=trainpath,
                                           test_data_path=testpath,
                                           save_qerror_file_path=savepath,
                                           device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                                           save_path=savemodel)

    # Map each character in the vocabulary to an index.
    char2idx = {letter: i for i, letter in enumerate(card_estimator_configs.vocabulary)}

    # Initialize the model with the defined parameters.
    model = selectivity_estimator.Cardinality_Estimator(1, card_estimator_configs.hidden_size, card_estimator_configs.device,
                                                        len(char2idx))

    # Prepare the training data by padding sequences and converting them to indices.
    train_data = misc_utils.addpaddingTrain(card_estimator_configs.train_data_path, char2idx)
    dataloadertrain = DataLoader(train_data, batch_size=card_estimator_configs.batch_size, shuffle=True)

    # Train the model with the training data.
    trained_model = modelTrain(dataloadertrain, model, card_estimator_configs.device,
                               card_estimator_configs.learning_rate, card_estimator_configs.num_epocs,
                               card_estimator_configs.save_path)

    # Prepare the test data for estimating cardinality.
    datasettest = misc_utils.addpaddingTest(card_estimator_configs.test_data_path, char2idx)
    dataloadertest = DataLoader(datasettest, batch_size=1)

    # Estimate cardinalities for the test dataset and save the results.
    estimate_cardinality(dataloadertest, trained_model, card_estimator_configs.device,
                         card_estimator_configs.save_qerror_file_path, card_estimator_configs.datasize)
