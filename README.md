# Train a TensorFlow model on Kubernetes to recognize art culture based on the collection from the Metropolitan Museum of Art

In this developer journey, we will use Deep Learning to train an image classification model. 
The data comes from the art collection at the New York Metropolitan Museum of Art and the metadata from Google BigQuery.
We will use the Inception model implemented in TensorFlow and we will run the training on a Kubernetes cluster.  
We will save the trained model and load it later to perform inference.  
To use the model, we provide as input a picture of a painting and the model will return the likely culture, for instance Italian Florence art.
The user can choose other attributes to classify the art collection, for instance author, time period, etc.
Depending on the compute resource available, the user can choose the number of images to train, the number of classes to use, etc.
In this journey, we will select a small set of image and a small number of classes to allow the training to complete within a reasonable amount of time.
With a large dataset, the training may take days or weeks.

When the reader has completed this journey, they will understand how to:

* Collect and process the data for Deep Learning in TensorFlow
* Configure Distributed TensorFlow to run on a cluster of servers
* Configure and deploy TensorFlow to run on a Kubernetes cluster
* Train an advanced image classification Neural Network
* Use TensorBoard to visualize and understand the training process

![](doc/source/images/architecture.png)

## Flow

1. Inspect the available attributes in the Google BigQuery database for the Met art collection 
2. Create the labeled dataset using the attribute selected
3. Select a model for image classification from the set of available public models and deploy to IBM Bluemix 
4. Run the training on Kubernetes, optionally using GPU if available
5. Save the trained model and logs
6. Visualize the training with TensorBoard
7. Load the trained model in Kubernetes and run an inference on a new art drawing to see the classification


## Included components

* [TensorFlow](http://www.tensorflow.org): An open-source library for implementing Deep Learning models
* [Image classification models](https://github.com/tensorflow/models/tree/master/slim): an implementation of the Inception neural network for image classification
* [Google metadata for Met Art collection](https://bigquery.cloud.google.com/dataset/bigquery-public-data:the_met?pli=1): a database containing metadata for the art collection at the New York Metropolitan Museum of Art
* [Met Art collection](link): a collection of over 200,000 public art artifacts, including paintings, books, etc.
* [Kubernetes cluster](https://kubernetes.io): an open-source system for orchestrating containers on a cluster of servers 
* [IBM Bluemix Container Service](https://console.ng.bluemix.net/docs/containers/container_index.html?cm_sp=dw-bluemix-_-code-_-devcenter): a public service from IBM that hosts users applications on Docker and Kubernetes 


## Featured technologies

* [TensorFlow](https://www.tensorflow.org): Deep Learning library
* [TensorFlow models](https://github.com/tensorflow/models/tree/master/slim): public models for Deep Learning
* [Kubernetes]():  Container orchestration 

# Watch the Video
[![](http://img.youtube.com/vi/Jxi7U7VOMYg/0.jpg)](https://www.youtube.com/watch?v=Jxi7U7VOMYg)


# Steps
1. [Register for Google BigQuery and set up your environment](#1-set-up-environment)
2. [Create the label for the dataset](#2-create-label)
3. [Download the data](#3-download-data)
4. [Convert the data to TFRecord format](#4-convert-data)
5. [Create a Kubernetes cluster on IBM Bluemix](#5-create-kubernetes-cluster)
6. [Deploy the TensorFlow pods to run the training on Kubernetes](#6-deploy-training)
7. [Save the trained model and logs](#7-save-trained-model)
8. [Visualize the training with TensorBoard](#8-visualize)
9. [Load the trained model in Kubernetes and run an inference on a new art drawing](#9-run-inference)


### 1. Set up environment 

Refer to the [instruction](https://cloud.google.com/bigquery/docs/reference/libraries) to install the client on your laptop to interact with Google BigQuery,   

```
	pip install --upgrade google-cloud-bigquery
```

Install the [Google Cloud SDK](https://cloud.google.com/sdk/docs/) on your laptop.  For example, on the Mac, download:

```
   google-cloud-sdk-168.0.0-darwin-x86_64.tar.gz
```

Unpack and run the command:

```  
	./google-cloud-sdk/bin/gcloud init
```

This will start the browser and request you to log into your gmail account and ask you to choose a project 
in Google cloud.

Authenticate for the client on your laptop by this command:

```
	./google-cloud-sdk/bin/gcloud auth application-default login
```

Your laptop should be ready to interface with Google BigQuery.


### 2. Create label

A labeled dataset is the first requirement for training a model.  Collecting the data and associating label to the data typically 
requires a lot of resources and effort.

Google BigQuery contains a collection of public databases that are useful for various purposes.  For our case, we are interested 
in the data for the [art collection at the Metropolitan Museum](https://bigquery.cloud.google.com/table/bigquery-public-data:the_met.objects?pli=1)
Check this [blog] (https://cloud.google.com/blog/big-data/2017/08/when-art-meets-big-data-analyzing-200000-items-from-the-met-collection-in-bigquery) 
for more details.  Looking at the tables, we see quite a few attributes that can be used to label the art data.  
For this journey, we will select the "culture" attribute, which describes the name of the culture where the art item 
is originated from, for instance "Italian Florence".  Based on the example from this journey, you can choose any other
attribute to label the art images.

The file bigquery.py provides a simple python script that will query the Google BigQuery database.  
To get a list of the unique cultures, the SQL string is:

```
SELECT culture, COUNT(*) c 
        FROM `bigquery-public-data.the_met.objects`
        GROUP BY 1
        ORDER BY c DESC
```

To get a list of all art items labeled with the culture, the SQL string is:

```
SELECT department, culture, link_resource
        FROM `bigquery-public-data.the_met.objects`
        WHERE culture IS NOT NULL
        LIMIT 200
```

You can enter these strings on the Google BigQuery console to see the data.
The journey also provides convenient script to query the attributes. 
First clone the journey git repository:

```
git clone https://github.com/IBM/tensorflow-kubernetes-art-classification.git
```

The script to query Google BigQuery is bigquery.py. 
Edit the script to put the appropriate SQL string and run the script:

```
cd tensorflow-kubernetes-art-classification
python bigquery.py
```

You can redirect the output to save to a file.  For reference, the output of the two queries above are provided in the
following files:

* cultures-all.list
* arts-all.list


### 3. Download data

Although the Google BigQuery holds the attributes, the photos of the art collection are actually kept at a site
from the Metropolitan Museum of Art.  Therefore, to build our labeled dataset, we will need to download the photos 
and associate them with the labels.  Looking at the list of art items, there are some 114,627 items with labels 
that we can use.  There are 4,259 unique labels for these items, although only 540 labels have more than 10 photos and 
would be useful for training a model.  If a particular culture has just a few art images, it's probably not enough
to train the model 

The script download.py is provided to build the raw labeled data.  It will read from the file arts-select.list,
download the image source found in each line and place it in a directory named with the label.
You can copy from the lines from the file `arts-all.list` into the file `arts-select.list` and edit as needed
to create a list of images to download. 

```
python download.py
```



### 4. Convert data

At this point, we will begin to use TensorFlow code to process the data.  
Install TensorFlow on your environment following the [instruction from TensorFlow](https://www.tensorflow.org/install/)

Clone the TensorFlow git repository containing a collection of public models:

```
cd ~
git clone https://github.com/tensorflow/models.git
```

We will use and extend the collection of image classification models in the directory `models/slim`.
The code provided in this directory will allow you to process several different image datasets 
(CIFAR, Flowers, ImageNet) and you can choose from several advanced models to train.  
To extend this code base to process our new dataset of art images, copy the following files into the
directory:

```
cp tensorflow-kubernetes-art-classification/dataset_factory.py models/slim/datasets/dataset_factory.py
cp tensorflow-kubernetes-art-classification/arts.py models/slim/datasets/arts.py 
```

We will convert the raw images into the TFRecord format that the TensorFlow code will use.
To convert the art dataset, put the directories of downloaded pictures in a directory named `met_art`, 
for instance `~/data/met_art`.
Run the script:

```
python3 convert.py --dataset_dir="~/data"
```

The output will be in the directory `~/data`:

```
arts_train_00000-of-00005.tfrecord
arts_train_00001-of-00005.tfrecord
arts_train_00002-of-00005.tfrecord
arts_train_00003-of-00005.tfrecord
arts_train_00004-of-00005.tfrecord
arts_validation_00000-of-00005.tfrecord
arts_validation_00001-of-00005.tfrecord
arts_validation_00002-of-00005.tfrecord
arts_validation_00003-of-00005.tfrecord
arts_validation_00004-of-00005.tfrecord
labels.txt
```

Note that the data has been divided into two sets:  one for training and one for validation.  The portion
of data set aside for validation is 25% and this can be changed in the script convert.py.  The file `labels.txt`
lists all the culture label found in the images directory.

Occasionally, an image file is corrupted and the image processing step in the conversion would fail.  
You can scan the image collection first for corrupted files by running the command:

```
python3 convert.py --dataset_dir="~/data" --check_image=True
```

Then the corrupted images can be removed from the dataset.


### 5. Create Kubernetes cluster

Register for a free account at the [IBM Bluemix Kubernetes service](https://console.bluemix.net). 
Create a Kubernetes cluster following the [instruction](https://console.bluemix.net/docs/containers/container_index.html#clusters)
Select the Lite version and download the Bluemix cli and kubectl package for interfacing with your cluster.

### 6. Deploy training 

Initialize your Bluemix Container Plug-in and set your terminal context to your Kubernetes cluster
To deploy the pod, you will need to create an image containing the TensorFlow code by running the command:

```
docker build -t my_image_name:v1 -f Dockerfile
``` 

Note that we include a small sample copy of the dataset in this image.  The reason is twofold.  First, shared 
filesystem is not available for the free Bluemix account.  In normal practice, the dataset is too large to copy into the image and 
you would keep the dataset in a shared filesystem such as SoftLayer NFS.  When a pod is started, the shared filesystem
would be mounted so that the dataset is available to all the pods.
Second, the computation resource provided with the free Bluemix account is not sufficient to run the training
within a reasonable amount of time.  In practice, you would use a larger dataset and allocate sufficient resources 
such as multiple CPU cores and GPU.  Depending on the amount of computation resources, the training can run for days
or over a week.
 
Create a namespace on Bluemix and [add this image] 
(https://console.bluemix.net/docs/services/Registry/registry_setup_cli_namespace.html#registry_namespace_add)
to the namespace . 

Next, create a [Kubernetes secret]
(https://console.bluemix.net/docs/services/Registry/ registry_tokens.html#registry_tokens)
to store the Bluemix token information.  Add the Kubernetes secret to the template yaml file:

```
apiVersion: v1
kind: Pod
metadata:
name: <pod_name> spec:
containers:
- name: <container_name>
image: registry.<region>.bluemix.net/<my_namespace>/<my_image>:<tag> imagePullSecrets:
- name: <secret_name>
```

Deploy the pod with the command:

```
kubectl create -f <my_yaml>
```

Along with the pod, a local volume will be created and mounted to the pod to hold the output of the training.
This includes the checkpoints, which are used for resuming after a crash and saving a trained model, 
and the event file, which is used for visualization.


### 7. Save trained model

Copy the files from the Kubernetes local volume.

The trained model is the last checkpoint file.


### 8. Visualize

The event file copied from the Kubernetes local volume contains the log data for TensorBoard.
Start the TensorBoard and point to the local directory with the event file:

```
tensorboard --logdir=<path_to_dir>
```

Then open your browser with the link displayed from the command.

### 9. Run inference

Now that you have trained a model to classify art image by culture, you can provide
a new art image to see how it will be classified by the model. 
In the training we have run above, we used a very small dataset for illustration purpose because of the 
very limited resources provided with the Lite version of the Kubernetes cluster.  Therefore
the trained model only cover some 5 culture categories and will not be very accurate.  For this step,
we use a saved model from a previous training that will cover some 600 culture categories.
The model is included in the git repository.  



# License
[Apache 2.0](LICENSE)


# Privacy Notice
If using the `Deploy to Bluemix` button some metrics are tracked, the following
information is sent to a [Deployment Tracker](https://github.com/IBM-Bluemix/cf-deployment-tracker-service) service
on each deployment:

* Node.js package version
* Node.js repository URL
* Application Name (`application_name`)
* Application GUID (`application_id`)
* Application instance index number (`instance_index`)
* Space ID (`space_id`)
* Application Version (`application_version`)
* Application URIs (`application_uris`)
* Labels of bound services
* Number of instances for each bound service and associated plan information

This data is collected from the `package.json` file in the sample application and the `VCAP_APPLICATION` and `VCAP_SERVICES` environment variables in IBM Bluemix and other Cloud Foundry platforms. This data is used by IBM to track metrics around deployments of sample applications to IBM Bluemix to measure the usefulness of our examples, so that we can continuously improve the content we offer to you. Only deployments of sample applications that include code to ping the Deployment Tracker service will be tracked.

## Disabling Deployment Tracking

To disable tracking, simply remove ``require("cf-deployment-tracker-client").track();`` from the ``app.js`` file in the top level directory.
