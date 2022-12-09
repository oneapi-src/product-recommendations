'''
   This code base is adopted from the below notebook
   https://www.kaggle.com/shawamar/product-recommendation-system-for-e-commerce
'''
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=W0311,E0401,W0622,W0612,W0105

# Product Recommendation System for e-commerce
import argparse
import time
import sys
import warnings
import logging

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import TruncatedSVD
from joblib import dump, load

import pandas as pd
warnings.filterwarnings("ignore")

DATASET_FILE ="./data/flipkart_com-ecommerce_sample.csv"
TRUE_K =12
def run_hyperparametertuning(X1):
    """run_hyperparametertuning"""
    best_score = 0.001  # Setting the initial Silhoutte Score
    #Parameters for Hyperparameter Tuning
    no_cluster = [5,10, 15,20]  # number of clusters
    max_iter = [400, 450,500,550]  # max iteration
    #Start of Hyperparameter Tuning to save the best model based on the Silhoutte score for the cluster created on Prediction from the models
    # Start hyper parameter tuning time
    sum_time = []
    for i in no_cluster:
        for j in max_iter:
            print('No.cluster', i, '\nMax Iter', j)
            start = time.time()
            model = KMeans(n_clusters=i, init='k-means++', max_iter=j, random_state=0)
            #Train the model
            model.fit(X1[:int(datasize*0.7)])
            #Prediction using the model
            y_means = model.predict(X1[:(int(datasize))])
            sum_time.append(time.time()-start)
            #Sihoutte score calculation from prediction
            score = silhouette_score(X1[:(int(datasize))], y_means)
            print("silhoutte score is :" ,score)
            #Check if the model score is greater than the best score
            if score > best_score:
                best_score=score  # Set the best score to the new score calculated
                print("Saving model!!! Best score is --->",best_score)
                dump(model,"./saved_models/prod_rec.joblib")  #Save the model
                best_params = (i,j)
    logger.info('Total fit and predict time taken during Hyperparameter Tuning in sec: %s', sum(sum_time))  # Calculation on time taken for Hyperparamater tuning
    return best_params

def batch_inference(loaded_model):
    '''Performs batch inference'''
    # Warm up 
    print("warm up in progress........")
    for i in range (5):
        y_means = loaded_model.predict(X1[:(-int(datasize*0.3))])
    # Start Time analysis for Batch inference
    avg = []
    print("Time Analysis for Batch Inference")
    print("dataset size",X1[:(int(datasize))].shape)
    for i in range (0,10):
        start2 = time.time() # Start time
        y_means = loaded_model.predict(X1[:(int(datasize))])  # Prediction
        end_time=time.time()-start2
        avg.append(end_time) #Calculate the time taken for batch inference
        logger.info('Time of Batch time recomendation:%s',end_time)
    logger.info('Average Time of Batch time recomendation:%s', sum(avg)/len(avg))  # Calculate the average time

def real_time_inference(product,model):
        """Perform Real time Inference """
        y_1 = vectorizer.transform([product])
        #svd = TruncatedSVD(n_components=10)
        y_1 = svd.transform(y_1)
        start_time_real = time.time()
        prediction = model.predict(y_1)
        end_time = time.time()-start_time_real
        return end_time

def show_recommendations(product,model):
        """show_recommendations"""
        y_1 = vectorizer.transform([product])
        y_1 = svd.transform(y_1)
        prediction = model.predict(y_1)
        print("Recommendations for : ",product)
        print_cluster(prediction[0])
        
if __name__ == "__main__":
    
    #Arguements
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        '--datasetsize',
                        default=10000,
                        type=int,
                        required=False,
                        help="Size of the dataset"
                        )

    parser.add_argument('-i',
                        '--intel',
                        default=False,
                        help="use intel accelerated technologies")

    parser.add_argument('-l',
                        '--logfile',
                        type=str,
                        default="",
                        help="log file to output benchmarking results to")

    parser.add_argument('-t',
                        '--tuning',
                        required=False,
                        type=str,
                        default=0,
                        help='hyper parameter tuning (0/1)')
                        
    parser.add_argument('-mp',
                        '--modelpath',
                        required=False,
                        type=str,
                        default=None,
                        help='model path')

    FLAGS = parser.parse_args()
    datasize = FLAGS.datasetsize
    if FLAGS.logfile == "":
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(filename=FLAGS.logfile, level=logging.DEBUG)
    logger = logging.getLogger()
   
    if FLAGS.intel:
        "import the intel sklearnex"
        logging.debug("Loading intel libraries..")
        from sklearnex import patch_sklearn
        patch_sklearn()
    else:
        logging.debug("Loading stock libraries..")
    start_time_data_prep = time.time()  # Start of data prep  
    try:
        train_original = pd.read_csv(DATASET_FILE)  #Read data from csv file
    except IOError as e:  # noqa:F841
        print('data not found , please provide the valid path')
    train = train_original
    print(len(train_original))
    while len(train) < datasize:  # Check if the length of csv rows is less than the input data
        train= pd.concat([train,train_original ], ignore_index=True) # Concatenate the original data with the existing data
    train = train.head(datasize)
    print(len(train))
    #import pdb;pdb.set_trace()
    logging.debug(train.shape)
    # Droping missing values
    train = train.dropna()
    logging.debug(train.shape)

    # Converting text data into numeric data using TF-IDF vectorizer.
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(train["description"])
    svd = TruncatedSVD(n_components=10)
    X1 = svd.fit_transform(X)
    logging.debug(X1.shape)

    # Creating function to print clusters.
    def print_cluster(i):
        """print_cluster"""
        print('Cluster %d:' % i),
        for ind in order_centroids[i, :10]:           
            print(' %s' % terms[ind])
    logger.info('Data preparation time:%s', time.time()-start_time_data_prep)

    # Hyperparameter tuning
    if FLAGS.tuning:
        #Start hyperparameter tuning
        (cluster,iter) = run_hyperparametertuning(X1)
        print("Hyperparameter Tuning has been executed successfully!!")
        print("Best parameters=====> n_clusters:",cluster ,"	max_iter :",iter)
        model = KMeans(n_clusters=cluster,max_iter=iter,random_state=0)
        start5 = time.time()  # Start time for Training without hyperparameter tuning
        model.fit(X1[:int(datasize*0.7)])  # Train model using Kmeans
        train_time = time.time()-start5
        logger.info('Kmeans_training_time_with the best params:%s', train_time)
        sys.exit(0)

    # Start Training if inference flag is False    
    if FLAGS.modelpath is None:
    	# Fitting K-Means to the dataset 
        model = KMeans(n_clusters=TRUE_K, random_state=0)
        start4 = time.time()  # Start time for Training without hyperparameter tuning
        model.fit(X1[:int(datasize*0.7)])  # Train model using Kmeans
        train_time = time.time()-start4
        print("Top terms per cluster:")
        original_space_centroids = svd.inverse_transform(model.cluster_centers_)
        order_centroids = original_space_centroids.argsort()[:, ::-1]
        terms = vectorizer.get_feature_names()
        for i in range(TRUE_K):
          print_cluster(i)
        logger.info('Kmeans_training_time_without_Hyperparametertunning:%s', train_time)  # Calculate and print the time taken for hp tuning
        print("Saving model..........")
        dump(model,"./saved_models/prod_rec.joblib")  #Save the model

    # Inferencing
    if FLAGS.modelpath is not None:
        model = load(FLAGS.modelpath)
        original_space_centroids = svd.inverse_transform(model.cluster_centers_)
        order_centroids = original_space_centroids.argsort()[:, ::-1]
        terms = vectorizer.get_feature_names()
        #Calculate batch inference timings
        batch_inference(model)
        # Calculating realtime inference time.
        List = ["cutting tool", "spray paint", "steel drill", "water", "powder"] 
        Avg = []       
        for i in List:
            time_inference_real = real_time_inference(i,model)
            logger.info('time taken for realtime recommendation:%s', time_inference_real)  # Calculate and print the time taken for real time inference
            Avg.append(time_inference_real)
            show_recommendations(i,model) 
        logger.info('Average Time of Real time recomendation:%s', sum(Avg)/len(Avg))  # Calculate average time for real time inference
       