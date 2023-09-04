import os
import argparse
import pickle5 as pickle
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from sklearn.metrics import confusion_matrix

from utils1 import get_tiny_images
from utils1 import build_vocabulary, get_bags_of_sifts
from utils1 import nearest_neighbor_classify

CAT = ['Kitchen', 'Store', 'Bedroom', 'LivingRoom', 'Office',
       'Industrial', 'Suburb', 'InsideCity', 'TallBuilding', 'Street',
       'Highway', 'OpenCountry', 'Coast', 'Mountain', 'Forest']

CAT2ID = {v: k for k, v in enumerate(CAT)}

ABBR_CAT = ['Kit', 'Sto', 'Bed', 'Liv', 'Off',
            'Ind', 'Sub', 'Cty', 'Bld', 'St',
            'HW', 'OC', 'Cst', 'Mnt', 'For']

NUM_PER_CAT = 100  # number of training or testing examples per category
assert NUM_PER_CAT <= 100


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--feature', help='feature',
                        type=str, default='bag_of_sift')
    parser.add_argument('--classifier', help='classifier',
                        type=str, default='nearest_neighbor')
    parser.add_argument('--dataset_dir', help='dataset directory',
                        type=str, default='../hw2_data/p3_data/')
    args = parser.parse_args()

    print('Loading all data paths and labels...')
    train_img_paths, test_img_paths, train_labels, test_labels = get_img_paths_and_labels(
        'C:/Users/88692/Documents/cv/hw2/hw2_data/p1_data')

    ####################################
    ###### Step 1: Build Features ######
    ####################################
    ##### TODO: get_tiny_images() in utils.py #####
    ##### TODO: build_vocabulary() in utils.py #####
    ##### TODO: get_bags_of_sifts() in utils.py #####
    print(f'Feature: {args.feature}')
    if args.feature == 'tiny_image':
        train_img_feats = get_tiny_images(train_img_paths)
        test_img_feats = get_tiny_images(test_img_paths)

    elif args.feature == 'bag_of_sift':
        # vocab and features is saved to disk to avoid recomputing the vocabulary every time
        if os.path.isfile('vocab.pkl') is False:
            print(
                'No existing visual word vocabulary found. Computing one from training images')
            # vocab_size is up to you, larger values will work better (to a point) but be slower to compute
            vocab_size = 100
            batch_size = 200
            vocab = build_vocabulary(train_img_paths, vocab_size, batch_size)
            with open('vocab.pkl', 'wb') as f:
                pickle.dump(vocab, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open('vocab.pkl', 'rb') as f:
                vocab = pickle.load(f)
        # train_image_feats
        if os.path.isfile('train_image_feats.pkl') is False:
            batch_size = 200
            num_samples = 500
            train_img_feats = get_bags_of_sifts(
                train_img_paths, vocab, batch_size, num_samples)
            with open('train_image_feats.pkl', 'wb') as f:
                pickle.dump(train_img_feats, f,
                            protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open('train_image_feats.pkl', 'rb') as f:
                train_img_feats = pickle.load(f)
        # test_image_feats
        if os.path.isfile('test_image_feats.pkl') is False:
            batch_size = 100
            test_img_feats = get_bags_of_sifts(
                test_img_paths, vocab, batch_size)
            with open('test_image_feats.pkl', 'wb') as f:
                pickle.dump(test_img_feats, f,
                            protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open('test_image_feats.pkl', 'rb') as f:
                test_img_feats = pickle.load(f)

    else:
        raise NameError('Unknown feature type')

    ##########################################
    ###### Step 2: Classify Test Images ######
    ##########################################
    ##### TODO: nearest_neighbor_classify() in utils.py #####
    print(f'Classifier: {args.classifier}')
    if args.classifier == 'nearest_neighbor':
        pred_cats = nearest_neighbor_classify(
            train_img_feats, train_labels, test_img_feats)

    elif args.classifier == 'random_classifier':
        pred_cats = test_labels[:]
        np.random.shuffle(pred_cats)

    else:
        raise NameError('Unknown classifier type')

    # Compute Accuracy (DO NOT MODIFY)
    accuracy = float(len([x for x in zip(test_labels, pred_cats)
                     if x[0] == x[1]])) / float(len(test_labels))
    print(f'Accuracy = {accuracy}\n')
    # Plot Confusion Matrix (You don't need to write this code)
    test_labels_ids = [CAT2ID[x] for x in test_labels]  # (1500)
    pred_cats_ids = [CAT2ID[x] for x in pred_cats]  # (1500)

    if pred_cats_ids:
        plot_confusion_mtx(test_labels_ids, pred_cats_ids, args.feature)


def get_img_paths_and_labels(data_path):

    train_img_paths, test_img_paths = [], []
    train_labels, test_labels = [], []

    for cat in CAT:
        cat_train_img_paths = glob(os.path.join(
            data_path, 'train', cat, '*.jpg'))
        cat_test_img_paths = glob(os.path.join(
            data_path, 'test', cat, '*.jpg'))
        for i in range(NUM_PER_CAT):
            train_img_paths.append(cat_train_img_paths[i])
            test_img_paths.append(cat_test_img_paths[i])
            train_labels.append(cat)
            test_labels.append(cat)

    # (1500), (1500), (1500), (1500)
    return train_img_paths, test_img_paths, train_labels, test_labels


def plot_confusion_mtx(test_labels_ids, pred_cats, feature_type):
    # compute confusion matrix
    cm = confusion_matrix(test_labels_ids, pred_cats)  # (15, 15)
    # normalize the confusion matrix by row (i.e number of samples in each class)
    cm_normalized = cm.astype(np.float32) / np.sum(cm, axis=1)[:, np.newaxis]
    # plot and save the image of confusion matrix
    plt.figure()
    plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Normalized confusion matrix')
    plt.colorbar()
    plt.xticks(np.arange(len(ABBR_CAT)), ABBR_CAT, rotation=45)
    plt.yticks(np.arange(len(ABBR_CAT)), ABBR_CAT)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(f'{feature_type}.png')


if __name__ == '__main__':
    main()
