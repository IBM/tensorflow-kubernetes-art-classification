# 쿠버네티스에서 메트로폴리탄 미술관의 작품을 활용하여 예술 작품을 인식하는 텐서플로우 모델 학습시키기

*Read this in other languages: [English](README.md).*

이 코드 패턴에서는 딥 러닝을 활용하여 이미지 분류 모델을 학습시킵니다. 데이터는 Google BigQuery를 사용하여 뉴욕 메트로폴리탄 미술관에 있는 작품들을 활용합니다.
텐서플로우에 구현 된 Inception 모델을 사용하고 쿠버네티스 클러스터에서 학습 시킨 다음 학습된 모델을 저장하고 나중에 추론을 수행할 때 다시 로드합니다.
모델을 사용하는 방법은, 예술 작품 이미지를 인풋으로 제공하면 가능성이 높은 관련 문화 정보(예를 들면 "이탈리아, 피렌체")를 반환합니다.
사용자는 미술 컬렉션을 분류할 때에 예를 들어 작가나 시기 등의 다른 속성을 선택할 수도 있습니다.
가용한 컴퓨팅 리소스에 따라 사용자는 학습시킬 이미지의 수 및 클래스 수 등을 선택할 수 있습니다.
이 코드 패턴에서는, 합리적인 시간 내에 학습을 마치기 위해 적은 양의 이미지와 클래스 수만을 사용할 것입니다.
데이터가 많으면, 학습 시간이 몇 일 또는 몇 주 까지도 소요될 수 있습니다.

이 코드 패턴을 이해하면 다음을 이해할 수 있습니다.:

* 텐서플로우로 딥러닝을 위한 데이터를 수집하고 처리합니다.
* 서버 클러스터에서 실행하기 위해 분산된 텐서플로우를 구성합니다.
* 쿠버네티스 클러스터에서 실행하기 위해 텐서플로우를 구성하고 배포합니다.
* 이미지 분류 신경망을 학습시킵니다. 
* TensorBoard를 활용하여 학습 과정을 시각화 하고 이해합니다.

![](doc/source/images/architecture.png)


## Flow

1. 메트로폴리탄 예술품을 수집하기 위해 구글 BigQuery에서 사용 가능한 속성을 검사합니다.
2. 선택한 속성을 사용하여 레이블이 지정된 데이터셋을 생성합니다.
3. 사용 가능한 공개 모델 중 이미지 분류 모델을 선택하고 IBM Cloud로 배포합니다.
4. 쿠버네티스에서 기계 학습을 수행합니다. 가능하다면 선택적으로 GPU를 사용합니다.
5. 학습된 모델과 로그를 저장합니다.
6. TensorBoard로 학습 과정을 시각화합니다.
7. 쿠버네티스에서 학습된 모델을 로드하여 새로운 미술 작품이 어떻게 분류될지 추론합니다.


## 포함된 구성 요소

* [TensorFlow](http://www.tensorflow.org): 딥러닝 모델 구현을 위한 오픈 소스 라이브러리
* [Image classification models](https://github.com/tensorflow/models/tree/master/research/slim): 텐서플로우 Slim API를 사용하여 구현된 이미지 분류를 위한 모델 목록
* [Google metadata for Met Art collection](https://bigquery.cloud.google.com/dataset/bigquery-public-data:the_met?pli=1): 뉴욕 메트로폴리탄 미술관에 있는 예술 작품 컬렉션의 200,000개 이상의 항목에 대한 메타 데이터가 저장된 데이터베이스
* [Met Art collection](https://metmuseum.org): 이 미술관에는 회화, 서적 등을 포함하는 45만개 이상의 공공 미술품이 있습니다.
* [Kubernetes cluster](https://kubernetes.io): 서버 클러스터에서 컨테이너를 관리하는 오픈 소스 시스템
* [IBM Cloud Container Service](https://cloud.ibm.com/docs/containers/container_index.html?cm_sp=dw-bluemix-_-code-_-devcenter): Docker와 Kubernetes에서 사용자 애플리케이션을을 호스팅하는 IBM의 퍼블릭 클라우드 서비스


## 주요 기술

* [TensorFlow](https://www.tensorflow.org): 딥러닝 라이브러리
* [TensorFlow models](https://github.com/tensorflow/models/tree/master/research/slim): 딥러닝을 위한 퍼블릭 모델 목록
* [Kubernetes](https://kubernetes.io): 컨테이너 오케스트레이션


# 영상으로 보기
[![](http://img.youtube.com/vi/I-8xmMxo-RQ/0.jpg)](https://www.youtube.com/watch?v=I-8xmMxo-RQ)


# 사전 준비 사항

[TensorFlow](https://www.tensorflow.org/install)를 설치합니다.

다음 중 한가지 방법으로 쿠버네티스 클러스터를 생성합니다.:
* 보유중인 서버를 활용하여 로컬에서 테스트 : [Minikube](https://kubernetes.io/docs/getting-started-guides/minikube)
* 클라우드에 배포 : [IBM Cloud Container Service](https://github.com/IBM/container-journey-template)
* 로컬 또는 클라우드에서 선택적으로 사용 : [IBM Cloud Private](https://www.ibm.com/cloud-computing/products/ibm-cloud-private/)

이 코든는 [IBM Cloud Container Service의 쿠버네티스 클러스터](https://cloud.ibm.com/docs/containers/cs_ov.html#cs_ov)에서 테스트 했습니다.

# Steps
1. [구글 BigQuery 등록 및 개발 환경 셋업](#1-개발-환경-셋업)
2. [데이터셋에 레이블 생성](#2-레이블-생성)
3. [데이터 다운로드](#3-데이터-다운로드)
4. [데이터를 TFRecord 포맷으로 변환](#4-데이터-변환)
5. [TensorFlow 컨테이너 이미지 생성](#5-이미지-생성)
6. [쿠버네티스에서 트레이닝을 실행하기 위해 TensorFlow pod 배포](#6-트레이닝-배포)
7. [학습된 모델의 정확도를 검증](#7-모델-평가)
8. [학습된 모델 및 로그 저장](#8-트레이닝된-모델-저장)
9. [텐서보드로 딥러닝 학습을 시각화](#9-시각화)
10. [학습된 모델을 쿠버네티스에 올리고 새 미술품의 문화권을 추론](#10-이미지-추론)


### 1. 개발 환경 셋업

[설명](https://cloud.google.com/bigquery/docs/reference/libraries)을 참조하여 Google BigQuery를 사용하도록 클라이언트를 설치합니다.:

```
$ pip install --upgrade google-cloud-bigquery
```

[Google Cloud SDK](https://cloud.google.com/sdk/docs/)를 설치합니다.

> 예를 들어 맥에서는 ` google-cloud-sdk-168.0.0-darwin-x86_64.tar.gz` 파일을 다운로드 하고 압축을 풉니다.

다음 명령을 실행합니다.:

```
$ ./google-cloud-sdk/bin/gcloud init
```

브라우저가 열리면 Gmail 계정으로 로그인 한 후 구글 클라우드에서 프로젝트를 선택합니다. 이 단계에서의 프로젝트 ID를 기록해 두십시오. 이 ID는 나중에 쿼리 스크립트에서 사용됩니다.

다음 명령으로 클라이언트에 인증합니다.:

```
$ ./google-cloud-sdk/bin/gcloud auth application-default login
```

Google BigQuery에 접속하도록 준비되어 있어야 합니다.

### 2. 레이블 생성

레이블된 데이터셋은 모델을 학습시키기 위한 첫번째 필요 사항입니다. 데이터를 수집하고 레이블을 지정하는 것은 일반적으로 많은 리소스와 노력을 필요로 합니다. 

Google BigQuery에는 다양한 목적에 유용한 공개 데이터베이스들이 포함되어 있습니다.
이 패턴에서는 [메트로폴리탄 미술관에 있는 예술 작품 모음](https://bigquery.cloud.google.com/table/bigquery-public-data:the_met.objects?pli=1) 데이터를 활용합니다.
자세한 내용은 이 [블로그(영문)](https://cloud.google.com/blog/big-data/2017/08/when-art-meets-big-data-analyzing-200000-items-from-the-met-collection-in-bigquery)를 확인하십시오.
테이블을 보면 미술품 데이터에 레이블을 지정하기 위해 사용될 수 있는 몇몇 속성이 있습니다.
이 코드 패턴에서는 예를 들어 "이탈리아, 플로렌스"와 같이 미술품이 기원한 문화권의 이름을 나타내는 "문화(culture)" 속성을 선택합니다. 이 코드 패턴의 예제를 기반으로 다른 속성을 선택하여 레이블을 지정할 수도 있습니다.
bigquery.py 파일은 Google BigQuery 데이터베이스를 조회하는 간단한 파이썬 스크립트입니다.
고유 문화권 목록을 얻는 SQL 문은 다음과 같습니다.:

```sql
SELECT culture, COUNT(*) c
        FROM `bigquery-public-data.the_met.objects`
        GROUP BY 1
        ORDER BY c DESC
```

문화권이 레이블된 모든 미술품 목록을 얻는 SQL문은 다음과 같습니다.:

```sql
SELECT department, culture, link_resource
        FROM `bigquery-public-data.the_met.objects`
        WHERE culture IS NOT NULL
        LIMIT 200
```

이러한 쿼리 스트링을 Google BigQuery 콘솔에서 데이터를 확인하는데에 사용할 수 있습니다. 또한 이 코드 패턴에서는 속성을 쿼리하기 위해 유용한 스크립트를 제공합니다. 먼저 이 git 레파지토리를 clone하십시오.:

```
$ cd /your_home_directory
$ git clone https://github.com/IBM/tensorflow-kubernetes-art-classification.git
```

Google BigQuery에 쿼리하기 위한 스크립트는 bigquery.py 입니다. 이 스크립트를 수정하여 적절한 SQL문을 넣으십시오. 이 파일에 있는 다음 구문을 이전 단계에서 노트해놓은 프로젝트 ID로 업데이트 하십시오.:

```
client = bigquery.Client(project="change-to-your-project-id")
```

수정한 스크립트를 실행합니다.:

```
$ cd tensorflow-kubernetes-art-classification
$ python bigquery.py
```

쿼리 결과를 파일로 저장할 수 있습니다. 참조를 위해, 위에 제시한 두 쿼리의 결과가 다음 파일로 저장되어 있습니다.:

* cultures-all.list
* arts-all.list


### 3. 데이터 다운로드

Google BigQuery가 속성 정보를 저장하고 있기는 하지만, 실제 이미지는 메트로폴리탄 미술관 사이트에 있습니다. 그래서 레이블이 지정된 데이터셋을 만들기 위해서는 그 이미지들을 다운로드 하여 레이블과 연관시킬 필요가 있습니다. 미술품의 목록을 보면 114,627개의 레이블이 있는 미술품을 사용할 수 있습니다. 이러한 미술품에 4,259개의 레이블이 있습니다. 하지만 540개의 레이블만이 10개 이상의 예시 이미지를 갖고 있고 모델을 학습하기에 적합합니다. 어떤 특정 문화권에 대해 단 몇개의 이미지만 있다면 모델을 학습시키기에 충분하지 않습니다.

download.py 스크립트는 정제되지 않은 레이블 데이터를 빌드하기 위해 사용됩니다. 이 스크립트는 arts-select.list 파일을 읽어 각 라인에 있는 이미지를 다운로드 하고 해당 레이블의 이름을 가진 폴더에 저장합니다. `arts-all.list` 파일에서 필요한 라인을 참조 또는 복사하여 `arts-select.list` 파일에서 다운로드할 이미지의 목록을 수정할 수 있습니다.

```
$ python download.py
```

> Note: 디스크 용량이 걱정되거나 IBM Cloud Kubernetice Service(Lite)를 사용하는 경우 그냥 sample-dataset.tar.gz 파일의 압축을 풀고 다운로드한 데이터로 사용해도 됩니다.


### 4. 데이터 변환

이 단계에서는 데이터를 처리하기 위해 텐서플로우 코드를 활용합니다. [텐서플로우 가이드](https://www.tensorflow.org/install/)를 따라 텐서플로우를 설치하십시오.

퍼블릭 모델 컬렉션을 갖고 있는 텐서플로우 git 레파지토리를 clone 합니다.:

```
$ cd /your_home_directory
$ git clone https://github.com/tensorflow/models.git
```

`models/slim` 폴더에 있는 여러 이미지 분류 모델을 사용하거나 확장할 것입니다.
이 디렉토리에 제공된 코드로 다른 이미지 데이터셋을(CIFAR, Flowers, ImageNet) 처리하거나 여러 학습 모델 중에 선택할 수 있습니다.
이 코드 패턴의 미술품 이미지 데이터 셋을 처리하기 위해 이 코드를 기반으로 확장하겠습니다. 다음 파일을 이 디렉토리로 복사하십시오.:

```
$ cp tensorflow-kubernetes-art-classification/dataset_factory.py models/research/slim/datasets/dataset_factory.py
$ cp tensorflow-kubernetes-art-classification/arts.py models/research/slim/datasets/arts.py
```

비정제된 이미지를 텐서플로우가 사용하는 TFRecord 포맷으로 변환하겠습니다. 데이터 셋을 변환하기 위해 `met_art` 폴더에 다운로드한 이미지 폴더를 넣습니다. 예를들면 `/your_home_directory/data/met_art` 폴더입니다.
다음 스크립트를 실행합니다.:

```
$ cp tensorflow-kubernetes-art-classification/convert.py models/research/slim/convert.py
$ cd models/research/slim
$ python convert.py --dataset_dir="/your_home_directory/data"
```

결과는 `/your_home_directory/data` 폴더에 나타납니다.:

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

데이터가 두 집합으로 나뉘는 것을 확인하십시오. : 하나는 학습을 위한, 다른 하나는 검증을 위한 데이터입니다. 검증을 위한 데이터 셋의 비율은 25%입니다. 이 비율은 convert.py 스크립트에서 변경할 수 있습니다. `labels.txt` 파일에 이미지 폴더에서 찾아볼 수 있는 모든 문화권 레이블 목록이 있습니다.

때로는 이미지 파일이 손상되어 변환 과정에서 이미지 처리 단계가 실패하는 경우가 있습니다.
다음 명령을 실행하여 손상된 파일이 있는지 먼저 이미지 컬렉션을 검사 할 수 있습니다.:

```sh
$ python convert.py --dataset_dir="/your_home_directory/data" --check_image=True
```

이 명령을 실행하면 손상된 이미지들이 데이터셋에서 제거됩니다.

### 5. 이미지 생성

pod를 배포하려면 먼저 다음 명령을 실행하여  텐서플로우 코드를 포함하는 이미지를 생성해야 합니다.:

```
$ cd /your_home_directory/tensorflow-kubernetes-art-classification
$ mkdir data
$ cp /your_home_directory/data/*.tfrecord data/.
$ cp /your_home_directory/data/labels.txt data/.
$ docker build -t your_image_name:v1 -f Dockerfile .
```

우리가 이 닥커 이미지에 적은 양의 데이터 셋만을 사용한다는 점을 기억하십시오. 이렇게 하는 이유는 두가지 측면인데 첫번째는 IBM Lite 계정으로는 공유 파일 시스템을 사용할 수 없습니다. 일반적으로 데이터셋은 닥커 이미지로 복사하기에는 너무 큽니다. 그렇기 때문에 데이터셋을 소프트레이어 NFS와 같은 공유 파일 시스템에 두어야 합니다. Pod가 시작될 때, 공유 파일시스템을 마운트 해서 데이터셋이 모든 pod에서 사용될 수 있도록 합니다. 두번째 측면은 IBM Cloud Lite 계정에 제공되는 컴퓨팅 리소스가 합리적인 시간 내에 트레이닝을 실행하기에 충분하지 않습니다. 실제 상황에서는 더 큰 데이터셋을 사용하고 여러 CPU 코어 및 GPU와 같이 충분한 리소스를 할당해야 합니다. 컴퓨팅 리소스의 양에 따라 트레이닝은 몇일이 걸릴 수도 몇주가 걸릴 수도 있습니다.

다음 단계로 이 [설명](https://cloud.ibm.com/docs/containers/cs_cluster.html#bx_registry_other)을 따르십시오.
  1. IBM Cloud Container Registry에 네임스페이스를 생성하고 이미지를 이 네임스페이스로 업로드 합니다.
  2. 만료되지 않는 레지스트리 토큰을 생성합니다.
  3. IBM Cloud 토큰 정보를 저장하기 위해 쿠버네티스 시크릿을 생성합니다.


### 6. 트레이닝 배포

`train-model.yaml` 파일을 이미지 이름과 시크릿 이름 부분을 수정합니다.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: met-art
spec:
  containers:
  - name: tensorflow
    image: registry.ng.bluemix.net/tf_ns/met-art:v1
    volumeMounts:
    - name: model-logs
      mountPath: /logs
    ports:
    - containerPort: 5000
    command:
    - "/usr/bin/python"
    - "/model/train_image_classifier.py"
    args:
    - "--train_dir=/logs"
    - "--dataset_name=arts"
    - "--dataset_split_name=train"
    - "--dataset_dir=/data"
    - "--model_name=inception_v3"
    - "--clone_on_cpu=True"
    - "--max_number_of_steps=100"
    - "--batch_size=10"
  volumes:
  - name: model-logs
    persistentVolumeClaim:
      claimName: met-art-logs
  imagePullSecrets:
  - name: bluemix-secret
  restartPolicy: Never
```

```sh
# Mac OS 사용 중이라면
$ sed -i '.original' 's/registry.ng.bluemix.net\/tf_ns\/met-art:v1/registry.<region>.bluemix.net\/<your_namespace>\/<your_image>:<tag>/' train-model.yaml
$ sed -i '.original' 's/bluemix-secret/<your_token>/' train-model.yaml

# 다른 Linux 플랫폼을 사용 중이라면
$ sed -i 's/registry.ng.bluemix.net\/tf_ns\/met-art:v1/registry.<region>.bluemix.net\/<your_namespace>\/<your_image>:<tag>/' train-model.yaml
$ sed -i 's/bluemix-secret/<your_token>/' train-model.yaml
```

다음 명령으로 pod를 배포합니다.:

```
$ kubectl create -f train-model.yaml
```

다음 명령으로 트레이닝의 상태를 확인할 수 있습니다.:

```
$ kubectl logs train-met-art-model
```
pod와 함께 로컬 볼륨이 생성되고 pod에 마운트되어 트레이닝의 결과를 저장합니다.이 결과물은 Crash가 났을 경우 재개하는데에 쓰이는 체크포인트와 트레이닝된 모델 및 시각화에 쓰이는 이벤트 파일을 포함하고 있습니다. pod의 재시작 정책은 트레이닝을 마치면 더이상 pod를 재시작할 필요가 없기 때문에 "Never"로 설정되어 있습니다.

### 7. 모델 평가

위의 트레이닝 단계에서 얻은 최신 체크포인트로부터 모델을 평가합니다.:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: eval-met-art-model
spec:
  containers:
  - name: tensorflow
    image: registry.ng.bluemix.net/tf_ns/met-art:v1
    volumeMounts:
    - name: model-logs
      mountPath: /logs
    ports:
    - containerPort: 5000
    command:
    - "/usr/bin/python"
    - "/model/eval_image_classifier.py"
    args:
    - "--alsologtostderr"
    - "--checkpoint_path=/logs/model.ckpt-100"
    - "--eval_dir=/logs"
    - "--dataset_dir=/data"
    - "--dataset_name=arts"
    - "--dataset_split_name=validation"
    - "--model_name=inception_v3"
    - "--clone_on_cpu=True"
    - "--batch_size=10"
  volumes:
  - name: model-logs
    persistentVolumeClaim:
      claimName: met-art-logs
  imagePullSecrets:
  - name: bluemix-secret
  restartPolicy: Never
```

`eval-model.yaml` 파일의 이미지 이름과 시크릿 이름을 스텝 6에서 했던 방식 처럼 각자의 것으로 수정합니다.

다음 명령으로 pod를 배포합니다.:

```
$ kubectl create -f eval-model.yaml
```

다음 명령으로 평가 과정의 상태를 체크합니다.:

```
$ kubectl logs eval-met-art-model
```


### 8. 트레이닝된 모델 저장

쿠버네티스 영구 볼륨의 모든 로그 파일을 로컬 호스트로 복사합니다.:

```
$ kubectl create -f access-model-logs.yaml
$ kubectl cp access-model-logs:/logs <path_to_local_dir>
```

디스크 공간이 문제가 되면 마지막 체크포인트 파일들인 트레이닝된 모델만 복사합니다. 그리고 다음 단계인 시각화 단계를 진행하기 위해 이벤트 파일들도 복사하십시오.


### 9. 시각화

쿠버네티스 영구 볼륨에서 복사한 이벤트 파일은 TensorBoard가 사용할 로그 데이터를 포함하고 있습니다. TensorBoard를 시작하고 이벤트 파일이 있는 로컬 디렉토리를 지정하십시오:

```
$ tensorboard --logdir=<path_to_dir>
```

커맨드를 통해 보여진 링크를 브라우저로 열어 확인하십시오.

### 10. 이미지 추론

이미지를 문화권으로 분류하는 모델을 트레이닝 했으므로 새로운 미술품 이미지를 인풋으로 제공하여 이 이미지가 모델에 의해 어떻게 분류되는지 확인하십시오.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: infer-met-art-model
spec:
  containers:
  - name: tensorflow
    image: registry.ng.bluemix.net/tf_ns/met-art:v1
    volumeMounts:
    - name: model-logs
      mountPath: /logs
    ports:
    - containerPort: 5000
    command:
    - "/usr/bin/python"
    - "/model/model/classify.py"
    args:
    - "--alsologtostderr"
    - "--checkpoint_path=/logs/model.ckpt-100"
    - "--eval_dir=/logs"
    - "--dataset_dir=/data"
    - "--dataset_name=arts"
    - "--dataset_split_name=validation"
    - "--model_name=inception_v3"
    - "--image_url=https://images.metmuseum.org/CRDImages/dp/original/DP800938.jpg"
  volumes:
  - name: model-logs
    persistentVolumeClaim:
      claimName: met-art-logs
  imagePullSecrets:
  - name: bluemix-secret
  restartPolicy: Never
```

스텝 6에서 했던 것 처럼 `infer-model.yaml` 파일의 Docker 이미지 이름과 시크릿 이름을 각자의 것으로 수정합니다. 그리고 이미지를 선택하여 image_url을 선택한 이미지로 변경합니다.

다음 명령으로 Pod를 배포하세요.:

```
$ kubectl create -f infer-model.yaml
```

추론이 진행되면 다음 명령으로 상태를 체크합니다. :

```
$ kubectl logs infer-met-art-model
```

위에서 실행한 트레이닝 과정에서는 매우 적은 양의 데이터만 사용했습니다. 쿠버네티스 클러스터의 Lite 버전이 아주 제한된 리소스만을 제공하기 때문입니다. 그래서 학습된 모델은 5개의 문화권 카테고리만을 다루고 아주 정확하지도 않습니다. 이 단계에서 600개의 문화권 카테고리를 커버하는 이 전에 학습시켰던 것의 [checkpoint](https://ibm.box.com/s/wyzl1k2tz1nosrf44mj20cmlruy7gsut)를 사용하십시오. 이 checkpoint의 정확도는 66% 입니다. 추론을 실행하기 위해 우리가 트레이닝 해놓은 체크포인트를 사용하려면 위의 파일을 다운로드 하고 이를 쿠버네티스 퍼시스턴트 볼륨으로 복사하십시오.:

```sh
$ kubectl delete -f access-model-logs.yaml # in case the access pod already exists
$ kubectl create -f access-model-logs.yaml
$ kubectl cp inception-v3-2k-metart-images.tar.gz access-model-logs:/logs/.
$ kubectl exec access-model-logs -ti /bin/bash
$ cd /logs
$ tar xvfz inception-v3-2k-metart-images.tar.gz
$ exit
```

다음으로 infer-model.yaml 파일을 이 체크포인트로 수정하십시오.

```yaml
command:
- "/usr/bin/python"
- "/model/model/classify.py"
args:
- "--alsologtostderr"
- "--checkpoint_path=/logs/inception-v3-2k-metart-images/model.ckpt-15000"
- "--eval_dir=/logs"
- "--dataset_dir=/logs/inception-v3-2k-metart-images"
- "--dataset_name=arts"
- "--dataset_split_name=validation"
- "--model_name=inception_v3"
- "--image_url=https://images.metmuseum.org/CRDImages/dp/original/DP800938.jpg"
```

마지막으로 추론을 다시 실행합니다.:

```sh
$ kubectl delete -f infer-model.yaml # in case the infer pod already exists
$ kubectl create -f infer-model.yaml
$ kubectl logs infer-met-art-model
```

# 더 학습 하려면...(영문)

* **인공 지능 코드 패턴**: 이 패턴이 유용하셨나요? 다른 [AI 코드 패턴들](https://developer.ibm.com/code/technologies/artificial-intelligence/)도 확인하십시오.
* **인공 지능 및 데이터 코드 패턴 플레이리스트**: 코드 패턴 비디오를 모아놓은 [플레이리스트](https://www.youtube.com/playlist?list=PLzUbsvIyrNfknNewObx5N7uGZ5FKH0Fde)를 즐겨찾기 하십시오.
* **PowerAI**: AI를 위한 엔터프라이즈 플랫폼에서 머신러닝을 시작하고 확장하고, 더 빠르게 하십시오.: [IBM Power Systems](https://www.ibm.com/ms-en/marketplace/deep-learning-platform)
* **IBM Cloud의 쿠버네티스**: [IBM Cloud에서 쿠버네티스와 Docker](https://www.ibm.com/cloud-computing/bluemix/containers)의 힘을 활용하여 애플리케이션을 배포하십시오.

# 링크 (영문)
* [IBM Cloud 컨테이너 서비스](https://cloud.ibm.com/docs/containers/container_index.html?cm_sp=dw-bluemix-_-code-_-devcenter): 사용자 애플리케이션을 Docker와 쿠버네티스에서 호스팅할 수 있는 IBM의 Public 서비스
* [텐서플로우](http://www.tensorflow.org): 딥러닝 모델을 구현한 오픈 소스 라이브러리
* [쿠버네티스 클러스터](https://kubernetes.io): 서버 클러스터에서 컨테이너를 관리하기 위한 오픈 소스 시스템
* [뉴욕 메트로폴리탄 미술관](https://metmuseum.org): 이 미술관은 그림, 책 등 450,000개 이상의 공공 예술품을 다양하게 전시하고 있습니다.
* [메트로 폴리탄의 소장품에 대한 구글 메타데이터](https://bigquery.cloud.google.com/dataset/bigquery-public-data:the_met?pli=1): 뉴욕 메트로폴리탄 미술관에 있는 200,000개 이상의 품목에 대해 메타데이터를 저장한 데이타베이스
* [구글 BigQuery](https://bigquery.cloud.google.com): 거대한 데이터셋에 대해 상호작용 가능한 분석을 제공하는 웹서비스
* [이미지 분류 모델](https://github.com/tensorflow/models/tree/master/research/slim): 텐서플로우 Slim API를 사용하여 구현된 이미지 분류 모델들

# 라이센스
[Apache 2.0](LICENSE)