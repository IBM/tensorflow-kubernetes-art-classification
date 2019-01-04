# 利用大都会艺术博物馆的藏品作为数据集，在 Kubernetes 上训练 TensorFlow 模型来识别艺术品
*阅读本文的其他语言版本：[English](README.md)。*

在此 Code Pattern 中，我们将使用深度学习来训练图像分类模型。
数据来自纽约大都会艺术博物馆的艺术藏品和 Google BigQuery 的元数据。
我们将使用 TensorFlow 中实现的 Inception 模型，并且将在 Kubernetes 集群上进行训练。
我们将保存训练的模型，之后加载该模型进行推理。
为使用此模型，我们提供油画图片作为输入，模型将返回可能的文化信息，例如，“Italian, Florence”艺术。
用户可以选择使用其他属性对艺术藏品进行分类，例如，按作者、时期等。
根据可用计算资源，用户可以选择要用于训练的图片数量、要使用的类别数量等。
在此 Code Pattern 中，我们将选择少量图像和少量类别，以便训练在合理的时间段内完成。
对于较大的数据集，训练可能需要耗时数日甚至数周时间才能完成。

读者完成此 Code Pattern 后，将会掌握如何：

* 在 TensorFlow 中收集和处理用于深度学习的数据
* 配置分布式 TensorFlow 以在服务器集群上运行
* 配置和部署 TensorFlow 以在 Kubernetes 集群上运行
* 训练高级图像分类神经网络
* 使用 TensorBoard 来可视化并理解训练流程

![](doc/source/images/architecture.png)


## 操作流程

1.检查 Google BigQuery 数据库中可供大都会艺术博物馆的艺术藏品使用的属性。

2.使用所选属性创建带标签的数据集

3.从一组可用公共模型中选择图像分类模型，并部署至 IBM Cloud

4.在 Kubernetes 上运行训练过程，如果 GPU 可用，可选择使用

5.保存训练的模型和日志

6.通过 TensorBoard 来可视化训练情况

7.在 Kubernetes 中加载训练模型，对新艺术画作运行推理以查看分类


## 包含的组件

* [TensorFlow](http://www.tensorflow.org)：用于实现深度学习模型的开源库
* [图像分类模型](https://github.com/tensorflow/models/tree/master/research/slim)：使用 TensorFlow Slim 高级 API 实现的一组图像分类模型。
* [用于大都会艺术博物馆藏品的 Google 元数据](https://bigquery.cloud.google.com/dataset/bigquery-public-data:the_met?pli=1)：包含纽约大都会艺术博物馆超过 200,000 件艺术藏品的元数据的数据库
* [大都会艺术博物馆藏品](https://metmuseum.org)：该博物馆收藏有超过 450,000 件公共艺术品，包括油画、书籍等。
* [Kubernetes 集群](https://kubernetes.io)：用于在服务器集群上编排容器的开源系统
* [IBM Cloud Container Service](https://cloud.ibm.com/docs/containers/container_index.html?cm_sp=dw-bluemix-_-code-_-devcenter)：来自 IBM 的公共服务，可在 Docker 和 Kubernetes 上托管用户应用程序


## 特色技术

* [TensorFlow](https://www.tensorflow.org)：深度学习库
* [TensorFlow 模型](https://github.com/tensorflow/models/tree/master/research/slim)：用于深度学习的公共模型
* [Kubernetes](https://kubernetes.io)：容器编排


# 观看视频
[![](http://img.youtube.com/vi/I-8xmMxo-RQ/0.jpg)](https://www.youtube.com/watch?v=I-8xmMxo-RQ)


# 前提条件

在工作站上安装 [TensorFlow](https://www.tensorflow.org/install)。

通过以下一种方法创建 Kubernetes 集群：
* [Minikube](https://kubernetes.io/docs/getting-started-guides/minikube)：用于使用您自己的服务器进行本地测试。
* [IBM Cloud Container Service](https://github.com/IBM/container-journey-template)：用于在云端部署。
* [IBM Cloud Private](https://www.ibm.com/cloud-computing/products/ibm-cloud-private/)：用于以上任一场景。

这里的代码针对 [来自 IBM Cloud Container Service 的 Kubernetes 集群](https://cloud.ibm.com/docs/containers/cs_ov.html#cs_ov) 进行了测试。


# 步骤
1.[注册 Google BigQuery 并设置环境](#1-set-up-environment)

2.[为数据集创建标签](#2-create-label)

3.[下载数据](#3-download-data)

4.[将数据转换为 TFRecord 格式](#4-convert-data)

5.[创建 TensorFlow 容器图像](#5-create-image)

6.[部署 TensorFlow pod 以在 Kubernetes 上运行训练过程](#6-deploy-training)

7.[评估训练模型的准确性](#7-evaluate-model)

8.[保存训练的模型和日志](#8-save-trained-model)

9.[通过 TensorBoard 来可视化训练情况](#9-visualize)

10.[在 Kubernetes 中加载训练模型，对新艺术画作运行推理](#10-run-inference)


### 1.设置环境

请参阅[指示信息](https://cloud.google.com/bigquery/docs/reference/libraries)，在您的笔记本电脑上安装
客户端，以便与 Google BigQuery 进行交互：

```
$ pip install --upgrade google-cloud-bigquery
```

在您的笔记本电脑上安装 [Google Cloud SDK](https://cloud.google.com/sdk/docs/)。

> 例如，在 Mac 上，下载并解压缩 ` google-cloud-sdk-168.0.0-darwin-x86_64.tar.gz`。

运行命令：

```
$ ./google-cloud-sdk/bin/gcloud init
```

这将启动浏览器并要求您登录 Gmail 帐户，然后会要求您在 Google Cloud 中
选择一个项目。  请记录此步骤中的项目 ID，以供稍后在查询脚本中使用。

在笔记本电脑上使用以下命令对客户端进行认证：

```
$ ./google-cloud-sdk/bin/gcloud auth application-default login
```

您的笔记本电脑应已准备好与 Google BigQuery 进行连接。


### 2.创建标签

带标签的数据集是对模型进行训练的首要要求。  通常情况下，收集数据并将标签与数据关联
需要大量的资源和工作量。

Google BigQuery 包含适合各种用途的公共数据库集合。  对于我们的案例，我们对
[大都会艺术博物馆的艺术藏品](https://bigquery.cloud.google.com/table/bigquery-public-data:the_met.objects?pli=1) 的相关数据感兴趣
请查看此[博客](https://cloud.google.com/blog/big-data/2017/08/when-art-meets-big-data-analyzing-200000-items-from-the-met-collection-in-bigquery)
获取更多详细信息。  通过查看表格，我们看到很多可用于作为艺术品数据标签的属性。
对于此 Code Pattern，我们将选择“culture”属性，它描述了艺术品来源的文化名称，
例如，“Italian, Florence”。  根据此 Code Pattern 的示例，您可以选择任何其他
属性作为艺术图像的标签。

bigquery.py 文件提供了一个简单的 Python 脚本，用于查询 Google BigQuery 数据库。
为获取不重复的文化列表，SQL 字符串为：

```sql
SELECT culture, COUNT(*) c
        FROM `bigquery-public-data.the_met.objects`
        GROUP BY 1
        ORDER BY c DESC
```

为获取以此文化作为标签的所有艺术品的列表，SQL 字符串为：

```sql
SELECT department, culture, link_resource
	FROM `bigquery-public-data.the_met.objects`
	WHERE culture IS NOT NULL
	LIMIT 200
```

您可以在 Google BigQuery 控制台上输入这些字符串以查看数据。该 Code Pattern 还提供了便利的脚本
以查询属性。首先，克隆以下 Git 存储库：

```
$ cd /your_home_directory
$ git clone https://github.com/IBM/tensorflow-kubernetes-art-classification.git
```

用于查询 Google BigQuery 的脚本为 bigquery.py。编辑此脚本来插入上述合适的 SQL 字符串。  使用来自先前步骤的 ID 更新此项目。

```
client = bigquery.Client(project="change-to-your-project-id")
```

运行该脚本：

```
$ cd tensorflow-kubernetes-art-classification
$ python bigquery.py
```

您可以重定向输出以将其保存至文件。  作为参考，以下文件中提供了
以上两项查询的输出：

* cultures-all.list
* arts-all.list


### 3.下载数据

虽然 Google BigQuery 存有艺术藏品的属性，但藏品的照片实际上保存在
大都会艺术博物馆的站点上。因此，要构建带标签的数据集，我们就需要下载照片，
并将其与标签关联。  查看艺术品列表，其中有约 114,627 件艺术品带有标签，
可供我们使用。  这些艺术品共有 4,259 个唯一标签，但仅 540 个标签含 10 张以上的照片，并且
可用于训练模型。  如果某个特定文化只有少量艺术图像，可能不足以
用来训练模型。

提供的脚本 download.py 用于构建原始带标签的数据。  它将从 arts-select.list 文件读取，
下载每行中找到的图像源，并将其置于以该标签命名的目录中。
您可以将 `arts-all.list` 文件中的行复制到 `arts-select.list` 文件中，并根据需要进行编辑，
以创建要下载的图像列表。

```
$ python download.py
```

> 备注：如果磁盘空间可能不足，或者您要使用 IBM Cloud Kubernetes Service(Lite)，
只需解压 sample-dataset.tar.gz 并将其用作为下载的数据即可。


### 4.转换数据

此时，我们将开始使用 TensorFlow 代码来处理数据。
遵循 [TensorFlow 的指示信息](https://www.tensorflow.org/install/)，在您的环境上安装 TensorFlow。

克隆包含公共模型集合的 TensorFlow Git 存储库：

```
$ cd /your_home_directory
$ git clone https://github.com/tensorflow/models.git
```

我们将使用并扩展 `models/slim` 目录中的图像分类模型的集合。
此目录中提供的代码将支持您处理多个不同的图像数据集
（CIFAR、Flowers、ImageNet），并且您可以从几个高级模型中选择要训练的模型。
要扩展此代码库以处理新的艺术图像数据集，请将以下文件复制到
目录中：

```
$ cp tensorflow-kubernetes-art-classification/dataset_factory.py models/research/slim/datasets/dataset_factory.py
$ cp tensorflow-kubernetes-art-classification/arts.py models/research/slim/datasets/arts.py
```

我们要将原始图像转换为 TensorFlow 代码将使用的 TFRecord 格式。
要转换艺术品数据集，请将下载图片的目录放置在名为 `met_art` 的目录中，
例如，`/your_home_directory/data/met_art`。
运行该脚本：

```
$ cp tensorflow-kubernetes-art-classification/convert.py models/research/slim/convert.py
$ cd models/research/slim
$ python convert.py --dataset_dir="/your_home_directory/data"
```

输出将位于目录 `/your_home_directory/data` 中：

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

请注意，此数据已被分为两个数据集：  一个用于训练，另一个用于验证。  为验证留出
的数据集部分为 25%，这可在 convert.py 脚本中进行更改。  `labels.txt` 文件
列出了图像目录中找到的所有文化标签。

有时，图像文件损坏，导致转换中的图像处理步骤失败。
您可以首先运行以下命令来扫描图像集合，找出损坏的文件：

```sh
$ python convert.py --dataset_dir="/your_home_directory/data" --check_image=True
```

然后，可从数据集中移除损坏的图像。


### 5.创建图像

要部署 pod，您将需要运行以下命令来创建包含 TensorFlow 代码的图像：

```
$ cd /your_home_directory/tensorflow-kubernetes-art-classification
$ mkdir data
$ cp /your_home_directory/data/*.tfrecord data/.
$ cp /your_home_directory/data/labels.txt data/.
$ docker build -t your_image_name:v1 -f Dockerfile .
```

请注意，此图像中包含一个小型数据集样本副本。  原因有两方面。首先，
免费的 IBM Cloud 帐户不可使用共享文件系统。  在正常做法中，数据集太大，以致于无法复制
到图像中，您会将数据集保留在共享文件系统中，例如，SoftLayer NFS。启动 pod 后，
将安装此共享文件系统，以便该数据集可供所有 pod 使用。其次，
随免费 IBM Cloud 帐户提供的计算资源不足以在合理的时间内
运行训练过程。在实际操作中，您将使用更大的数据集，并分配足够的资源，例如，多个 CPU 核心和
GPU。根据计算资源的数量，训练过程可能会运行几天，甚至会超过一周时间。

接下来，遵循这些[指示信息](https://cloud.ibm.com/docs/containers/cs_cluster.html#bx_registry_other) 来完成以下操作：

	1.在 IBM Cloud Container Registry 中创建一个名字空间，并将图像上传至此名称空间
  
	2.创建不会过期的注册表令牌
	
	3.创建 Kubernetes 密钥来存储 IBM Cloud 令牌信息
	


### 6.部署训练过程

使用您的图像名称和密钥名称来更新 `train-model.yaml` 文件：

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
  volumes:
  - name: model-logs
    persistentVolumeClaim:
      claimName: met-art-logs
  imagePullSecrets:
  - name: bluemix-secret
  restartPolicy: Never
```

```sh
# 对于 Mac OS
$ sed -i '.original' 's/registry.ng.bluemix.net\/tf_ns\/met-art:v1/registry.<region>.bluemix.net\/<your_namespace>\/<your_image>:<tag>/' train-model.yaml
$ sed -i '.original' 's/bluemix-secret/<your_token>/' train-model.yaml

# 对于所有其他 Linux 平台
$ sed -i 's/registry.ng.bluemix.net\/tf_ns\/met-art:v1/registry.<region>.bluemix.net\/<your_namespace>\/<your_image>:<tag>/' train-model.yaml
$ sed -i 's/bluemix-secret/<your_token>/' train-model.yaml
```

使用以下命令部署 pod：

```
$ kubectl create -f train-model.yaml
```

使用以下命令检查训练状态：

```
$ kubectl logs train-met-art-model
```

随 pod 一起将创建一个本地卷，并将该卷安装到 pod 上，用于保存训练输出。
这包括用于在崩溃后复原和保存训练模型的检查点，以及
用于实现可视化的事件文件。此外，将 pod 的重新启动策略设置为“Never”，因为
一旦训练完成，就无需再重新启动 pod。


### 7.评估模型

对以上训练步骤中最后一个检查点中的模型进行评估：

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
效仿步骤 6 中的操作，使用您的图像名称和密钥名称来更新 `eval-model.yaml` 文件。

使用以下命令部署 pod：

```
$ kubectl create -f eval-model.yaml
```

使用以下命令检查评估状态：

```
$ kubectl logs eval-met-art-model
```


### 8.保存训练的模型

将 Kubernetes 持久卷上的所有日志文件都复制到本地主机。

```
$ kubectl create -f access-model-logs.yaml
$ kubectl cp access-model-logs:/logs <path_to_local_dir>
```

如果磁盘空间可能不足，那么仅复制作为最后检查点文件的训练模型。
此外，复制事件文件用于下面的“可视化”步骤。


### 9.可视化

从 Kubernetes 持久卷复制的事件文件包含 TensorBoard 的日志数据。
启动 TensorBoard 并指向含事件文件的本地目录：

```
$ tensorboard --logdir=<path_to_dir>
```

然后，使用浏览器打开命令行中显示的链接。


### 10.运行推理

现在，您完成按文化信息对艺术图像进行分类的模型的训练，您可以提供
新的艺术图像来查看该模型将如何对其进行分类。

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

效仿步骤 6 中的操作，使用您的 Docker 图像名称和密钥名称来更新 `infer-model.yaml` 文件。
此外，将 image_url 替换为您所选的艺术图像。

使用以下命令部署 pod：

```
$ kubectl create -f infer-model.yaml
```

使用以下命令检查推理状态：

```
$ kubectl logs infer-met-art-model
```

在以上运行的训练中，我们使用了很小的数据集来进行说明，因为
Kubernetes 集群的 Lite 版本提供的资源非常有限。因此，
训练的模型仅涵盖了 5 个文化类别，准确性不高。对于此步骤，
您可以使用我们先前训练的[检查点](https://ibm.box.com/s/wyzl1k2tz1nosrf44mj20cmlruy7gsut)，其中涵盖了
600 个文化类别。该检查点的准确率为 66%。
如果要使用我们的检查点来运行推理，请从以上链接下载该检查点，
然后将其复制到 Kubernetes 持久卷：

```sh
$ kubectl delete -f access-model-logs.yaml # in case the access pod already exists
$ kubectl create -f access-model-logs.yaml
$ kubectl cp inception-v3-2k-metart-images.tar.gz access-model-logs:/logs/.
$ kubectl exec access-model-logs -ti /bin/bash
$ cd /logs
$ tar xvfz inception-v3-2k-metart-images.tar.gz
$ exit
```

接下来，使用该检查点更新 infer-model.yaml：

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

最后，运行推理：

```sh
$ kubectl delete -f infer-model.yaml # in case the infer pod already exists
$ kubectl create -f infer-model.yaml
$ kubectl logs infer-met-art-model
```
# 了解更多信息

* **人工智能 Code Pattern**：喜欢此 Code Pattern？了解我们的其他 [AI Code Pattern](https://developer.ibm.com/code/technologies/artificial-intelligence/)。
* **关于 AI 和数据的 Code Pattern 播放清单**：收藏包含我们所有 Code Pattern 视频的[播放清单](https://www.youtube.com/playlist?list=PLzUbsvIyrNfknNewObx5N7uGZ5FKH0Fde)
* **PowerAI**：通过一个在 Enterprise Platform for AI 上运行的用于机器学习的软件发行版，更快地开始开发或扩展：[IBM Power Systems](https://www.ibm.com/ms-en/marketplace/deep-learning-platform)
* **IBM Cloud 上的 Kubernetes**：为您的应用程序带来 [IBM Cloud 上的 Kubernetes 和 Docker](https://www.ibm.com/cloud-computing/bluemix/containers) 的组合力量

# 链接
* [IBM Cloud Container Service](https://cloud.ibm.com/docs/containers/container_index.html?cm_sp=dw-bluemix-_-code-_-devcenter)：来自 IBM 的公共服务，可在 Docker 和 Kubernetes 上托管用户应用程序
* [TensorFlow](http://www.tensorflow.org)：用于实现深度学习模型的开源库
* [Kubernetes 集群](https://kubernetes.io)：用于在服务器集群上编排容器的开源系统
* [纽约大都会艺术博物馆](https://metmuseum.org)：该博物馆收藏有超过 450,000 件公共艺术品，包括油画、书籍等。
* [用于大都会艺术博物馆藏品的 Google 元数据](https://bigquery.cloud.google.com/dataset/bigquery-public-data:the_met?pli=1)：包含纽约大都会艺术博物馆超过 200,000 件艺术藏品的元数据的数据库
* [Google BigQuery](https://bigquery.cloud.google.com)：提供大规模数据集的交互式分析的 Web 服务
* [图像分类模型](https://github.com/tensorflow/models/tree/master/research/slim)：使用 TensorFlow Slim 高级 API 实现的一组图像分类模型。

# 许可
[Apache 2.0](LICENSE)
