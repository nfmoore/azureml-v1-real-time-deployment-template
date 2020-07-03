# Getting Started

## Setup and Deploy Platform

### 1. Create a new GitHub repository

1. Login to your GitHub account, navigate to the [AML Real-Time Deployment Template repository](https://github.com/nfmoore/aml-real-time-deployment-template) and create a new repository from this template. Use [these](https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/creating-a-repository-from-a-template) instructions for more details about creating a repository from a template.

### 2. Create and configure a new Azure DevOps Project

1. Create a new project in Azure DevOps. Use [these](https://docs.microsoft.com/en-us/azure/devops/organizations/projects/create-project?view=azure-devops&tabs=preview-page#create-a-project) instructions for more details about creating a project in Azure DevOps.

2. Create an Azure Resource Manager Service connection. This is needed so Azure Devops can connect to your subscription and create/manage resources.

   1. Click on `project settings` (found at the bottom-left of the portal).
   2. Click on `Service Connections` (found on the sidebar).
   3. Click `New Service Connection` (found at the top-right of the portal).
   4. Select `Azure Resource Manager` and click `Next`.
   5. Select `Service principal (automatic)` and click `Next`.
   6. Leave `Subscription` as the scope level and select your `subscription` from the dropdown.
   7. In the `Service connection name` textbox enter `azure-service-connection`. Note: you can enter another name for your service connection but this will require editing the [`variables.yml`](../.pipelines/templates/variables.yml) file.
   8. Leave the `Allow all pipelines to use this connection` checkbox selected and click `Next`.

   The above steps assume you have `Contributor` or `Owner` access to the subscription or resource group.

   See [these](https://docs.microsoft.com/en-us/azure/devops/pipelines/library/service-endpoints#create-a-service-connection) instructions for more details about creating an Azure Resource Manager Service connection.

3. Create a variable group in Azure DevOps to store values that are reused across multiple pipelines or pipeline stages.

   1. Select the `Library` tab from the `Pipelines` section (found on the sidebar).
   2. Click `+ Variable group` and create a variable group named `aml-deployment-templates`. Note: you can enter another name for your variable group but this will require editing the [`variables.yml`](../.pipelines/templates/variables.yml) file.
   3. The variable group should contain the following required variables:

      | Variable Name          | Suggested Value            |
      | ---------------------- | -------------------------- |
      | `aks_cluster_name`     | `aks-cluster`              |
      | `aml_sku`              | `enterprise`               |
      | `compute_cluster_name` | `cpu-cluster`              |
      | `environment`          | `development`              |
      | `location`             | `australiaeast`            |
      | `namespace`            | `amlplatform`              |
      | `service_connection`   | `azure-service-connection` |

See [these](https://docs.microsoft.com/en-us/azure/devops/pipelines/library/variable-groups?view=azure-devops&tabs=classic#use-a-variable-group) instructions for more details about creating a variable group in Azure DevOps.

### 3. Setup the environment

1. The easiest way to setup the environment is by using the [AML Platform Deployment Template](https://github.com/nfmoore/aml-platform-deployment-template).
2. Alternatively, you can provision the infrastructure manually by:

   1. Creating an Azure Machine Learning Workspace
   2. Creating a compute cluster with the same name as the `compute_cluster_name` variable in the variable group in Azure DevOps.
   3. Creating an inference cluster with the same name as the `aks_cluster_name` variable in the variable group in Azure DevOps. This will create an AKS cluster in your resource group.
   4. Registering a dataset with the name `cardiovascular_disease_dataset` using the [Cardiovascular Disease dataset](https://www.kaggle.com/sulianova/cardiovascular-disease-dataset) (which has been adapted from Kaggle) and is available [here](https://github.com/nfmoore/aml-platform-deployment-template/blob/master/data/cardiovascular-disease.csv).

   The above steps assume you have `Contributor` or `Owner` access to the subscription or resource group.

### 4. Deploy the solution

1. In your GitHub repository, edit the any variables in the [`variables.yml`](../.pipelines/templates/variables.yml) you wish to change for your deployment. Remember to change the values for `service_connection` or `resource_group_name` if you have selected something different in section 2 step 2 above.

2. In your Azure DevOps project, create a pipeline from your repository.

   1. Select the `Pipelines` tab from the `Pipelines` section (found on the sidebar).
   2. Click on `New Pipeline` (found at the top-right of the portal).
   3. Select `GitHub` (authenticate if necessary).
   4. Select your GitHub repository from the list of repositories (authenticate if necessary).
   5. Select `Existing Azure Pipelines YAML File`.
   6. Select the `master` branch and enter `/.pipeline/main.yml` as the path or select it from the dropdown and click `Continue`.
   7. Select `Run` from the top-right of the protal to execute the pipeline and deploy the platform.

   See [these](https://docs.microsoft.com/en-us/azure/devops/pipelines/create-first-pipeline) instructions for more details about creating a pipeline in Azure DevOps.
