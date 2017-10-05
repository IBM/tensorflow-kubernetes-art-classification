FROM tensorflow/tensorflow:nightly

WORKDIR /

RUN apt-get update && \
    apt-get install -y git && \
    mkdir /model /data && \
    git clone https://github.com/tensorflow/models.git && \
    cp -r /models/research/slim/* /model/ && \
    rm -rf /models  

COPY dataset_factory.py /model/datasets/.
COPY arts.py /model/datasets/. 
COPY data/*.tfrecord /data/
COPY data/labels.txt /data/.

ENTRYPOINT ["python", "/model/train_image_classifier.py"]
