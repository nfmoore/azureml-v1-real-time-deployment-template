# AML Real Time Deployment Template

Machine Learning Operations (MLOps) is based on DevOps principles and practices that increase the efficiency of Machine Learning workflows. It aims to facilitate faster experimentation, development and production deployment of Machine Learning models while ensuring high quality standards. A standard end-to-end MLOps workflow will consist of model training, registration, deployment and monitoring.

![ML lifecycle](/docs/images/ml-lifecycle.png)

This deployment template uses [Azure Machine Learning](https://docs.microsoft.com/en-us/azure/machine-learning/overview-what-is-azure-ml) and [Azure Pipelines](https://docs.microsoft.com/en-us/azure/devops/pipelines/get-started/what-is-azure-pipelines) (part of Azure DevOps). The template contains code and DevOps pipeline definitions to automate an end-to-end deployment of a real time Machine Learning solution using MLOps principles and practices. The template includes: unit tests and code coverage, model training and registration, controlled deployments (via approvals) to ACI in a test environment and AKS in a production environment with model monitoring.

## References

- [Azure Machine Learning documentation](https://docs.microsoft.com/en-us/azure/machine-learning/)
- [Azure Pipelines documentation](https://docs.microsoft.com/en-us/azure/devops/pipelines/)
- [Azure Machine Learning Python SDK](https://docs.microsoft.com/en-us/python/api/overview/azure/ml/?view=azure-ml-py)
- [Azure Machine Learning CLI](https://docs.microsoft.com/en-us/azure/machine-learning/reference-azure-machine-learning-cli)
