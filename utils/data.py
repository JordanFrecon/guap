import torch
import random
torch.set_default_tensor_type(torch.DoubleTensor)


class QuickAttackDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        return self.images[item], self.labels[item]


def split_dataset_in_two(raw_dataset, n_class, n_fst_split, n_snd_split, randomize_img=True, randomize_order=True):
    # Split every example according to its label
    example_per_class = {}
    for (x, y) in raw_dataset:
        if torch.is_tensor(y):
            y_id = y.item()
        else:
            y_id = y
        if y_id not in example_per_class.keys():
            example_per_class[y_id] = [[x, torch.tensor(y_id)]]
        else:
            y_list = example_per_class[y_id]
            y_list.append([x, torch.tensor(y_id)])

    # Randomize the examples
    if randomize_img:
        for y in example_per_class.keys():
            random.shuffle(example_per_class[y])

    # Select the right number of examples for Validation and Test
    fst_split = []
    snd_split = []
    for y in example_per_class.keys():
        n_ex_val = n_fst_split // n_class
        n_ex_test = n_snd_split // n_class
        all_y_example = example_per_class[y]

        snd_y_example = all_y_example[:n_ex_test]
        fst_y_example = all_y_example[n_ex_test:n_ex_test + n_ex_val]

        fst_split.extend(fst_y_example)
        snd_split.extend(snd_y_example)

    # Randomize the examples after the splitting : The order is then random
    if randomize_order:
        random.shuffle(fst_split)
        random.shuffle(snd_split)

    return fst_split, snd_split


def split_augmented_dataset_in_two(raw_dataset, n_class, n_fst_split, n_snd_split, randomize_img=True, randomize_order=True):
    # Split every example according to its label
    example_per_class = {}
    for (x, y, sigma) in raw_dataset:
        if torch.is_tensor(y):
            y_id = y.item()
        else:
            y_id = y
        if y_id not in example_per_class.keys():
            example_per_class[y_id] = [[x, torch.tensor(y_id), sigma]]
        else:
            y_list = example_per_class[y_id]
            y_list.append([x, torch.tensor(y_id), sigma])

    # Randomize the examples
    if randomize_img:
        for y in example_per_class.keys():
            random.shuffle(example_per_class[y])

    # Select the right number of examples for Validation and Test
    fst_split = []
    snd_split = []
    for y in example_per_class.keys():
        n_ex_val = n_fst_split // n_class
        n_ex_test = n_snd_split // n_class
        all_y_example = example_per_class[y]

        snd_y_example = all_y_example[:n_ex_test]
        fst_y_example = all_y_example[n_ex_test:n_ex_test + n_ex_val]

        fst_split.extend(fst_y_example)
        snd_split.extend(snd_y_example)

    # Randomize the examples after the splitting : The order is then random
    if randomize_order:
        random.shuffle(fst_split)
        random.shuffle(snd_split)

    return fst_split, snd_split


def prepare_data(raw_data, batch_size, n_class, n_val, n_tst, n_val1, n_val2=None, pin_memory=False, augmented=False):
    n_val2 = n_val-n_val1 if n_val2 is None else n_val2

    if augmented:
        validation_set, test_set = split_augmented_dataset_in_two(raw_dataset=raw_data, n_class=n_class, n_fst_split=n_val, n_snd_split=n_tst)
    else:
        validation_set, test_set = split_dataset_in_two(raw_dataset=raw_data, n_class=n_class, n_fst_split=n_val,
                                                        n_snd_split=n_tst)

    # Split validation set into two parts for (semi)-universal attacks
    if augmented:
        validation_set_p1, validation_set_p2 = split_augmented_dataset_in_two(raw_dataset=validation_set,
                                                                              n_class=n_class,n_fst_split=n_val1,
                                                                              n_snd_split=n_val2)
    else:
        validation_set_p1, validation_set_p2 = split_dataset_in_two(raw_dataset=validation_set, n_class=n_class,
                                                             n_fst_split=n_val1, n_snd_split=n_val2)

    # Data loaders
    validation_p2_loader = torch.utils.data.DataLoader(dataset=validation_set_p2, batch_size=batch_size // 2,
                                                       pin_memory=pin_memory)
    validation_loader = torch.utils.data.DataLoader(dataset=validation_set, batch_size=batch_size, pin_memory=pin_memory)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, pin_memory=pin_memory)

    return validation_loader, test_loader, validation_set_p1, validation_p2_loader


def prepare_data_new(raw_data, batch_size, n_class, n_val, n_tst, pin_memory=False):

    validation_set, test_set = split_dataset_in_two(raw_dataset=raw_data, n_class=n_class, n_fst_split=n_val, n_snd_split=n_tst)

    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, pin_memory=pin_memory)

    return validation_set, test_loader