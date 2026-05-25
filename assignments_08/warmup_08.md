# Cloud Concepts

These questions are based on the Cloud Overview lesson.

### Cloud Concepts Question 1

_What is the core economic model of cloud computing, and how does it differ from owning your own servers?_

The core economic model of cloud computing is pay-as-you-go model. It differs from owning your own servers because it removes the need to buy your own computing resources and maintaining it. It is also easier to scale up and scale down and you only pay for what you use.

### Cloud Concepts Question 2

_What is the difference between vertical scaling and horizontal scaling? Give a concrete example of when you might choose each._

Vertical scaling is upgrading the machine itself (more CPU, RAM, and bigger GPU) and horizantal scaling is adding more machines and splitting the work across them.

Horizantal scaling would be useful when there are traffic spikes where the cloud adds additional instances automatically and then scales back once that spike ends. An example would be a web service. As traffic grows the number of machines running the same thing can be added.

Vertical scaling would be useful for when it's hard to be split over multiple machines or keeping things simple. An example would be a single database that grows as traffic grows which means queries get slower. Upgrading the hardware would speed things up.

_Then, for the three scenarios below, write one sentence saying which type of scaling applies and why._

    A web app that normally handles 1,000 users per day suddenly needs to handle 100,000 after a viral product launch.
    A data scientist's model training job is running too slowly, and they want a machine with a faster GPU and more RAM.
    A data pipeline that processes 10 files per run now needs to process 10,000 files per run, and the work can be split across machines.

- A web app that has their traffic spike from 1,000 to 100,000 would use horizontal scaling because multiple machines can all run the same website.

- A data scientist's model training job running slow would use vertical scaling because they want to upgrade their machine with a faster GPU and more RAM.

- A data pipeline that goes from 10 files per run to processing 10,000 files per run where the work can be split across machines would be horizontal scaling.

### Cloud Concepts Question 3

_Before writing your definitions, classify each item in the list below as IaaS, PaaS, or SaaS. One sentence of reasoning is enough for each._

    Gmail
    Azure Virtual Machines
    Azure App Service
    AWS S3 (Simple Storage Service)
    GitHub Codespaces
    Snowflake

- Gmail - SaaS - This is software as a service because you just open a browser and use it.
- Azure Virtual Machines - IaaS - This is infrastructure as a service because you get a virtual machine, storage and network connection but you set everything else up yourself.
- Azure App Service - PaaS - This is platform as a service because the provider gives the infrastucture but you deploy the application and the platform handles running it, scaling it, and keeping the machine healthy.
- AWS S3 (Simple Storage Service) - PaaS - This is a platform as a service because AWS fully manages the infrastructure and objects/files are just stored and retrieved from it.
- GitHub Codespaces - SaaS - This is software as a service because Github manages everything, the software environment is used to code.
- Snowflake - SaaS - This is a software as a service because customers use it as a fully managed software pattern over the internet.

_Now describe IaaS, PaaS, and SaaS in your own words. For each, give one example (from the lesson or the list above) and describe what you, as the developer, are responsible for managing._

IaaS is infrastructure as a service. This is where a customer gets a virtual machine, storage, and network connection and they set everything up the way they want. So you can choose the operating system, configure the enviroment, and manage security. It requires the most work/setup but it is the most flexible. An example of this is AWS EC2.

PaaS is platform as a service. This is where the provider manages the infrastructure but you provide the code. The platform handles running the application (or script) you deploy as well as scaling it and keeping the underlying machine healthy. An example of this is Azure App Service.

SaaS is software as a service. This is where the the customer uses an application that someone else built, runs, and maintains. There is no need to think about servers or scaling. And example of this would be Gmail or Dropbox.

### Cloud Concepts Question 4

_What is a managed data platform like Databricks or Snowflake, and how does it differ from using a cloud provider like Azure directly? What do you gain, and what do you give up?_

Managed data platforms are built ontop of cloud infrastructure. Databricks and Snowflake set up the software specifically optimized for data and analytics workloads. It's a layer that manages cloud resources on the consumers behalf. This allows for quicker set up for large-scale data processing or machine learning. It does lose out on flexibility as well as can have a higher cost.

### Cloud Concepts Question 5

_The lesson names two situations where the cloud is probably not the right choice. What are they?_

1. If the dataset fits comfortably on a single machine and does not require massive compute demands, local processing is faster and cheaper. This is good for setting up the initial prototype.
2. When a project is small or simple enough that the complexity, setup time and the potential costs overshadow the benefits.

# Azure Basics

These questions are based on the Getting Started with Azure lesson.

### Azure Basics Question 1

_What is the difference between an Azure subscription and a resource group? Which one is yours alone, and which one does CTD share?_

The difference between an Azure subscription and resource group is that a subscription is the billing account that owns all the resources in an organization, like CTD has a subscription. A resource group is a sandbox that bundles all related cloud resources together like a project directory. Each student gets their own resource group.

### Azure Basics Question 2

_Azure Cloud Shell is ephemeral by default. What does that mean in practice, and what does your course setup use to make it persistent?_

Azure Cloud Shell being ephemeral by default in practice means that once the shell is closed, all the files and directories created will be deleted. To help make it persistent would be to connect Cloud Shell to a file share, this is a named storage folder in Azure similar to a network drive. Once conneceted the entire home directory persists between sessions.

### Azure Basics Question 3

_What is the difference between your SSH private key and your SSH public key? Which one gets uploaded to the remote systems you want to connect to, and why is that safe?_

The SSH private key stays on the machine and should never be shared. The SSH public key is uploaded to the systems that need to be accesses. When connecting, SSH verifies that the two match which confirms the user without a password.

### Azure Basics Question 4

_Run the following command in Cloud Shell without the --output table flag:_

    az account show

_Paste the output into your answer. Then describe in one sentence what changes when you add --output table._

```
{
  "environmentName": "AzureCloud",
  "homeTenantId": "0f040ddd-301f-4665-8677-7b21f129d605",
  "id": "4e07c58c-751e-4765-b40c-632b9ee6fe6e",
  "isDefault": true,
  "managedByTenants": [],
  "name": "CTD Nonprofit Sponsorship",
  "state": "Enabled",
  "tenantId": "0f040ddd-301f-4665-8677-7b21f129d605",
  "user": {
    "cloudShellID": true,
    "name": "live.com#tiffanychung85@gmail.com",
    "type": "user"
  }
}
```

When --output table is added, the selected details are formatted into a table that makes it more easily read. The selected details are EnvironmentName, HomeTenantId, IsDefault, Name, State, TenantId.
