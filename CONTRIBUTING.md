## Contributing In General

Our project welcomes external contributions! If you have an itch, please
feel free to scratch it.

To contribute code or documentation, please submit a pull request to the [GitHub
repository](https://github.com/IBM/tensorflow-kubernetes-art-classification).

A good way to familiarize yourself with the codebase and contribution process is
to look for and tackle low-hanging fruit in the [issue
tracker](https://github.com/IBM/tensorflow-kubernetes-art-classification/issues). Before embarking on
a more ambitious contribution, please quickly [get in touch](#communication)
with us.

**We appreciate your effort, and want to avoid a situation where a contribution
requires extensive rework (by you or by us), sits in the queue for a long time,
or cannot be accepted at all!**

### Proposing new features

If you would like to implement a new feature, please [raise an
issue](https://github.com/IBM/tensorflow-art-journey/issues) before sending a pull
request so the feature can be discussed. This is to avoid you spending your
valuable time working on a feature that the project developers are not willing
to accept into the code base.

### Fixing bugs

If you would like to fix a bug, please [raise an
issue](https://github.com/IBM/tensorflow-kubernetes-art-classification/issues) before sending a pull
request so it can be discussed. If the fix is trivial or non controversial then
this is not usually necessary.

### Merge approval

The project maintainers use LGTM (Looks Good To Me) in comments on the code
review to indicate acceptance. A change requires LGTMs from two of the
maintainers of each component affected.

For more details, see the [MAINTAINERS](MAINTAINERS.md) page.

## Communication

Please feel free to connect with us: [here](https://github.com/IBM/tensorflow-kubernetes-art-classification/issues)

## Setup

You will need to a Google account to access the Google BigQuery service.

You will also need your own Kubernetes cluster and this can be created with one of the following methods:
* [Minikube](https://kubernetes.io/docs/getting-started-guides/minikube) for local testing using your own servers.
* [IBM Bluemix Container Service](https://github.com/IBM/container-journey-template) to deploy in cloud.
* [IBM Cloud Private](https://www.ibm.com/cloud-computing/products/ibm-cloud-private/) for either scenario above.

The code here is tested against [Kubernetes Cluster from Bluemix Container Service](https://console.ng.bluemix.net/docs/containers/cs_ov.html#cs_ov).


## Testing

To test your change, you can rerun the relevant [steps in the README file](README.md).


