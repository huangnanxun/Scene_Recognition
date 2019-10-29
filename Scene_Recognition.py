#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from scipy import stats
from pathlib import Path, PureWindowsPath
from sklearn import metrics


# In[ ]:


def extract_dataset_info(data_path):
    # extract information from train.txt
    f = open(os.path.join(data_path, "train.txt"), "r")
    contents_train = f.readlines()
    label_classes, label_train_list, img_train_list = [], [], []
    for sample in contents_train:
        sample = sample.split()
        label, img_path = sample[0], sample[1]
        if label not in label_classes:
            label_classes.append(label)
        label_train_list.append(sample[0])
        img_train_list.append(os.path.join(data_path, Path(PureWindowsPath(img_path))))
    print('Classes: {}'.format(label_classes))

    # extract information from test.txt
    f = open(os.path.join(data_path, "test.txt"), "r")
    contents_test = f.readlines()
    label_test_list, img_test_list = [], []
    for sample in contents_test:
        sample = sample.split()
        label, img_path = sample[0], sample[1]
        label_test_list.append(label)
        img_test_list.append(os.path.join(data_path, Path(PureWindowsPath(img_path))))  # you can directly use img_path if you run in Windows

    return label_classes, label_train_list, img_train_list, label_test_list, img_test_list


# In[ ]:


def get_tiny_image(img, output_size = (16,16)):
    tiny_img = np.zeros((output_size[1],output_size[0]))
    w_width = int(np.floor(img.shape[1]/output_size[0]))
    h_width = int(np.floor(img.shape[0]/output_size[1]))
    for i in range(0,output_size[0]):
        for j in range(0,output_size[1]):
            tiny_img[j][i] = int(np.round(np.mean(img[0+j*h_width:h_width-1+j*h_width,0+i*w_width:w_width-1+i*w_width])))
    tiny_mean = np.mean(tiny_img)
    tiny_ave = tiny_img - tiny_mean
    feature = tiny_ave/np.linalg.norm(tiny_ave)
    return feature


# In[ ]:


def predict_knn(feature_train, label_train, feature_test, k):
    neigh = NearestNeighbors(k)
    neigh.fit(feature_train)
    outp = neigh.kneighbors(feature_test, return_distance=False)
    label_test_pred_list = []
    for neighbor_set in outp:
        vote = {}
        for neighbor in neighbor_set:
            nei_class = label_train[neighbor]
            if nei_class in vote:
                vote[nei_class] += 1
            else:
                vote.update({nei_class:1})
        label_test_pred_list.append(max(zip(vote.values(),vote.keys()))[1])
    label_test_pred = label_test_pred_list
    return label_test_pred


# In[ ]:


def classify_knn_tiny(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):
    feature_tr = []
    for image_rt in img_train_list:
        img1 = cv2.imread(image_rt, 0)
        img2 = get_tiny_image(img1,(16,16))
        img2_vec = img2.reshape((1,256))
        feature_tr.append(img2_vec.tolist()[0])
    feature_tr = np.matrix(feature_tr)
    feature_te = []
    for image_rt in img_test_list:
        img1 = cv2.imread(image_rt, 0)
        img2 = get_tiny_image(img1,(16,16))
        img2_vec = img2.reshape((1,256))
        feature_te.append(img2_vec.tolist()[0])
    feature_te = np.matrix(feature_te)
    test_o = predict_knn(feature_tr, label_train_list, feature_te, 5)
    confusion = metrics.confusion_matrix(label_test_list, test_o, labels=label_classes)
    accuracy = metrics.accuracy_score(label_test_list, test_o)
    visualize_confusion_matrix(confusion, accuracy, label_classes)
    return confusion, accuracy


# In[ ]:


def visualize_confusion_matrix(confusion, accuracy, label_classes):
    plt.title("accuracy = {:.3f}".format(accuracy))
    plt.imshow(confusion)
    ax, fig = plt.gca(), plt.gcf()
    plt.xticks(np.arange(len(label_classes)), label_classes)
    plt.yticks(np.arange(len(label_classes)), label_classes)
    # set horizontal alignment mode (left, right or center) and rotation mode(anchor or default)
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="center", rotation_mode="default")
    # avoid top and bottom part of heatmap been cut
    ax.set_xticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    ax.set_yticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    plt.show()


# In[ ]:


def compute_dsift(img, stride = 10, size = 5):
    kpts = []
    j_max = int(np.floor((img.shape[0] - 2 * size)/stride))+1
    i_max = int(np.floor((img.shape[1] - 2 * size)/stride))+1
    for j in range(j_max):
        for i in range(i_max):
            kpts.append(cv2.KeyPoint(size+stride*i, size+stride*j, size))
    sift = cv2.xfeatures2d.SIFT_create()
    dense_feature = sift.compute(img, kpts)[1]
    return dense_feature


# In[ ]:


def build_visual_dictionary(dense_feature_list, dic_size = 50):
    X = np.array(dense_feature_list)
    kmeans = KMeans(n_clusters=dic_size,n_init=3,max_iter=300).fit(X)
    vocab = kmeans.cluster_centers_
    np.savetxt('selected_vocab1.txt', vocab)
    return vocab


# In[ ]:


def compute_bow(feature, vocab):
    tmp_bow = np.repeat(0,vocab.shape[0], axis=0)
    neigh = NearestNeighbors(n_neighbors=1).fit(vocab)
    cor_nei = neigh.kneighbors(np.array(feature), return_distance=False)
    for item in cor_nei:
        tmp_bow[item[0]] = tmp_bow[item[0]] + 1
    bow_feature = tmp_bow / np.linalg.norm(tmp_bow)
    return bow_feature


# In[ ]:


def classify_knn_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):
    Dict_Size = 150
    dense_feature_tr = np.array([]).reshape((0, 128))
    for image_rt in img_train_list:
        img1 = cv2.imread(image_rt, 0)
        img1_den_list = compute_dsift(img1)
        dense_feature_tr = np.concatenate((dense_feature_tr,img1_den_list),axis=0)
    vocab = build_visual_dictionary(dense_feature_tr, dic_size = Dict_Size)
    bow_tr_list = []
    for image_rt in img_train_list:
        img1 = cv2.imread(image_rt, 0)
        img1_bow = compute_bow(compute_dsift(img1), vocab)
        bow_tr_list.append(img1_bow)
    bow_tr_mat = np.matrix(bow_tr_list)
    bow_te_list = []
    for image_rt in img_test_list:
        img1 = cv2.imread(image_rt, 0)
        img1_bow = compute_bow(compute_dsift(img1), vocab)
        bow_te_list.append(img1_bow)
    bow_te_mat = np.matrix(bow_te_list)
    test_o = predict_knn(bow_tr_mat, label_train_list, bow_te_mat, 15)
    confusion = metrics.confusion_matrix(label_test_list, test_o, labels=label_classes)
    accuracy = metrics.accuracy_score(label_test_list, test_o)
    visualize_confusion_matrix(confusion, accuracy, label_classes)
    return confusion, accuracy


# In[ ]:


def predict_svm(feature_train, label_train, feature_test):
    classif = LinearSVC(penalty='l2',loss='squared_hinge',dual=False,tol=0.0001,C=1.0,multi_class='ovr', max_iter=1000)
    classif.fit(feature_train,np.array(label_train))
    test_o = classif.predict(feature_test)
    label_test_pred = test_o
    return label_test_pred


# In[ ]:


def classify_svm_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):
    Dict_Size = 150
    dense_feature_tr = np.array([]).reshape((0, 128))
    for image_rt in img_train_list:
        img1 = cv2.imread(image_rt, 0)
        img1_den_list = compute_dsift(img1)
        dense_feature_tr = np.concatenate((dense_feature_tr,img1_den_list),axis=0)
    vocab = build_visual_dictionary(dense_feature_tr, dic_size = Dict_Size)
    bow_tr_list = []
    for image_rt in img_train_list:
        img1 = cv2.imread(image_rt, 0)
        img1_bow = compute_bow(compute_dsift(img1), vocab)
        bow_tr_list.append(img1_bow)
    bow_tr_mat = np.matrix(bow_tr_list)
    bow_te_list = []
    for image_rt in img_test_list:
        img1 = cv2.imread(image_rt, 0)
        img1_bow = compute_bow(compute_dsift(img1), vocab)
        bow_te_list.append(img1_bow)
    bow_te_mat = np.matrix(bow_te_list)
    test_o = predict_svm(bow_tr_mat, label_train_list, bow_te_mat)
    confusion = metrics.confusion_matrix(label_test_list, test_o, labels=label_classes)
    accuracy = metrics.accuracy_score(label_test_list, test_o)
    visualize_confusion_matrix(confusion, accuracy, label_classes)
    return confusion, accuracy


# In[ ]:


if __name__ == '__main__':
    # To do: replace with your dataset path
    label_classes, label_train_list, img_train_list, label_test_list, img_test_list = extract_dataset_info("./scene_classification_data")
    
    classify_knn_tiny(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)

    classify_knn_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)
    
    classify_svm_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)

