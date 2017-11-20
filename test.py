# load  data and run basic classifiers

# Plot images
import pickle
import numpy as np
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import cross_val_score


# import cv2


def get_data():
    with open('train_label.pkl', 'rb') as f:
        train_labels_augmented = pickle.load(f)

    print "loading data from bottleneck features file "
    # todo make this filename a param

    # todo move this to a better place
    train_labels_one_hot = to_categorical(train_labels_augmented - 1)
    train_data = np.load(open('bottleneck_features_train_batch_150.npy'))
    print "shape of train data is  " + str(train_data.shape)

    return train_data, train_labels_one_hot


def nb(train_data, train_labels_one_hot, k_fold=10):
    print "training on Naive Bayes "
    from sklearn.naive_bayes import MultinomialNB
    gnb = MultinomialNB()
    accuracies = cross_val_score(gnb, train_data, train_labels_one_hot, cv=k_fold, n_jobs=-1)
    print(accuracies)
    print(np.mean(accuracies))


def main():
    train_data, train_labels_one_hot = get_data()
    k_fold = 10
    nb(train_data, train_labels_one_hot)


main()
