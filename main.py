
from collections import namedtuple
import torch
from torch.utils.data import DataLoader

import misc_utils
import A_NLM_model
import numpy as np
import os

def load_estimation_model(model_file_name, model, device):
    """
    Load a saved estimation model.

    Args:
        model_file_name (str): Path to the saved model file.
        model (torch.nn.Module): Estimation model.
        device (torch.device): Device to load the model on.

    Returns:
        torch.nn.Module: Loaded estimation model.
    """
    if os.path.exists(model_file_name):
        model.load_state_dict(torch.load(model_file_name))
        return model.to(device)
    else:
        raise FileNotFoundError(f"Model file not found: {model_file_name}")

def model_train(train_data, model, device, learning_rate, num_epochs, model_save_path):
    """
    Train an estimation model and save the trained model.

    Args:
        train_data (torch.utils.data.Dataset): Training dataset.
        model (torch.nn.Module): Estimation model.
        device (torch.device): Device to perform computations on.
        learning_rate (float): Learning rate for training.
        num_epochs (int): Number of training epochs.
        model_save_path (str): Path to save the trained model.

    Returns:
        torch.nn.Module: Trained estimation model.
    """
    model = A_NLM_model.train_model(train_data, model, device, learning_rate, num_epochs)
    torch.save(model.state_dict(), model_save_path)
    return model


def estimate_cardinality(test_dataset, model, device, dataset_size):
    """
    Estimate cardinality and print evaluation metrics.

    Args:
        test_dataset (torch.utils.data.Dataset): Test dataset.
        model (torch.nn.Module): Trained model.
        device (torch.device): Device to perform computations on.
        dataset_size (int): Size of the dataset.

    Returns:
        None
    """
    with torch.no_grad():
        qerror_list = []
        for name, mask, actual_card in test_dataset:
            name, mask, actual_card = name.to(device), mask.to(device), actual_card.to(device)
            output = model(name)
            output = torch.prod(torch.pow(output, mask)) * dataset_size
            qerror = misc_utils.compute_qerrors(actual_card, output.item())
            qerror_list.append(qerror[0].cpu().numpy())

        print(f'G-mean: {np.round(misc_utils.g_mean(qerror_list), 4)}')
        print(f'Mean: {np.round(np.average(qerror_list), 4)}')
        print(f'Median: {np.round(np.percentile(qerror_list, 50), 4)}')
        print(f'90th: {np.round(np.percentile(qerror_list, 90), 4)}')
        print(f'99th: {np.round(np.percentile(qerror_list, 99), 4)}')


def get_vocabulary(filename):
    char_set = set()
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip().split(':')[0]
            char_set.update(char for char in line.strip() if char != '%')
    print(''.join(sorted(char_set)) + '$.#')
    return ''.join(sorted(char_set)) + '$.#'

def main():
    A_NLM_Configs = namedtuple('A_NLM_Configs', ['vocabulary', 'hidden_size', 'learning_rate', 'batch_size', 'datasize',
                                                 'num_epochs', 'train_data_path', 'test_data_path',
                                                 'save_qerror_file_path', 'device', 'save_path'])

    vocab_file_path = 'path_to_file_that_you_get_all_possible_vocabularies'  # vocabulary can be set manually as well or can be taken from train file
    train_data_path = 'file_train_path'
    test_data_path = 'file_test_path'  # Adjust this to the correct file path

    card_estimator_configs = A_NLM_Configs(
        vocabulary=get_vocabulary(vocab_file_path),
        hidden_size=256,
        datasize=450000,
        learning_rate=0.0001,
        batch_size=128,
        num_epochs=64,
        train_data_path=train_data_path,
        test_data_path=test_data_path,
        save_qerror_file_path='save_path_qerrors',
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        save_path='save_model.pth'
    )

    char2idx = {letter: i for i, letter in enumerate(card_estimator_configs.vocabulary)}

    model = A_NLM_model.Cardinality_Estimator(1, card_estimator_configs.hidden_size,
                                              card_estimator_configs.device, len(char2idx))

    train_data = misc_utils.addpaddingTrain(train_data_path, char2idx)
    dataloadertrain = DataLoader(train_data, batch_size=card_estimator_configs.batch_size, shuffle=True)
    trained_model = model_train(dataloadertrain, model, card_estimator_configs.device,
                               card_estimator_configs.learning_rate, card_estimator_configs.num_epochs,
                               card_estimator_configs.save_path)

    datasettest = misc_utils.addpaddingTest(test_data_path, char2idx)  # Assuming you have a similar function for test data
    dataloadertest = DataLoader(datasettest, batch_size=1)
    estimate_cardinality(dataloadertest, trained_model, card_estimator_configs.device, card_estimator_configs.datasize)

if __name__ == "__main__":
    main()
