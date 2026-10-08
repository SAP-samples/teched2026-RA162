# Getting Started on Tabular AI

To access our tabular foundation models (SAP-RPT models, TabPFN models), there are 2 possible modes of consumption:
1. Via direct API call to each model (Exercises 1 and 2)
2. Via API call to the Tabular AI orchestration workflow (Exercise 3)

You can head directly to [Exercise 1 - SAP-RPT-1.6](../ex1/README.md) to get started with our exercises.

If you would like to replicate the exercises on your own setup, do note that there are additional steps to set up AI Core, AI Launchpad and HANA Cloud. You can see the instructions below. Do note that you do not need to go through the following steps during the on-site session.

## Familiarise yourself with SAP generative AI hub

Follow the steps in the [Initial Setup](https://help.sap.com/docs/sap-ai-core/sap-ai-core-service-guide/initial-setup?locale=en-US&ai=true) guide to provision an AI Core instance.

For easier access of AI Core features, we would recommend that you provision AI Launchpad as well. 

### Accessing AI Launchpad
If this is your first time setting up AI Launchpad, you will need to go through the following steps:
1. Click **Add** to create a new AI API connection. 

<img src="./images/AILP_Create AI API connection_1.png" width="500" alt="Create new AI API connection on AI Launchpad">

2. Provide a name for the AI API connection.
3. Upload the service key JSON file and the rest of the fields should be auto-populated. You can follow the steps [here](https://help.sap.com/docs/sap-ai-core/sap-ai-core-service-guide/create-service-key?locale=en-US&ai=true) to create the service key from your provisioned AI Core instance.

<img src="./images/AILP_Create AI API connection_2.png" width="500" alt="Upload AI Core service key to auto-populate the remaining fields">

4. Click **Create** to create the AI API connection.
5. Once created, the connection should come with the **default** resource group, which should also be auto-selected. You can go ahead to create new resource group if you require additional partitions.

<img src="./images/AILP_Dashboard.png" width="1000" alt="AI Launchpad dashboard">

### Deploying the models
> Note: For the on-site session, we have already deployed the models on your behalf. You can follow the steps in Exercises 1 and 2. 

If you would like to repeat this tutorial on your own, you will need to deploy the models on your own AI Core instance. Here are the steps to do so.
1. Head to **Model Library** under **Generative AI Hub** via the left navigation bar.

<img src="./images/AIC_Model library.png" width="1000" alt="Head to Model Library on AI Launchpad">

2. Within the **Model Library**, search for SAP RPT 1.6.

<img src="./images/AIC_Model library_SAP RPT.png" width="1000" alt="Search for SAP RPT 1.6 in the Model Library">

3. Click on the SAP RPT 1.6 model. 
4. Click Deploy.

<img src="./images/AIC_Deploy SAP RPT 1.6.png" width="1000" alt="Click Deploy to deploy the SAP RPT 1.6 model">

5. Repeat steps 2 to 4 to deploy the TabPFN-3.5 Plus model.

<img src="./images/AIC_Model library_TabPFN.png" width="1000" alt="Search for TabPFN-3.5 Plus in the Model Library">
<img src="./images/AIC_Deploy TabPFN.png" width="1000" alt="Click Deploy to deploy the TabPFN-3.5 Plus model">

6. Head to **Deployments** under **ML Operations** via the left navigation bar.

<img src="./images/AIC_Deployments.png" width="1000" alt="View all Deployments">

7. You should see the 2 new model deployments that you have created.
8. Once deployment is completed, you should see that the status indicates **RUNNING**.

### Getting the deployment URL for the models
1. Click into each deployment and you should see a deployment URL.
2. Copy this deployment URL.

<img src="./images/AIC_SAP RPT 1.6 deployment.png" width="1000" alt="View model deployment for SAP RPT 1.6">
<img src="./images/AIC_TabPFN deployment.png" width="1000" alt="View model deployment for TabPFN-3.5 Plus">

3. Add `/predict` to the end of the URL when making direct API inference requests to the model.

For API specifications of SAP-RPT-1.6, you can read more on [Help Portal](https://help.sap.com/docs/sap-ai-core/generative-ai/sap-rpt-1-6?locale=en-US&ai=true).
For API specifications of TabPFN-3.5 Plus, you can read more on [Help Portal](https://help.sap.com/docs/sap-ai-core/generative-ai/tabpfn-3-5-plus?locale=en-US&ai=true).

## Preparing HANA Data Lake File System (HDLFS)

### Provisioning SAP HANA Cloud
In your BTP subaccount, you should provision the **relational-data-lake** plan for SAP HANA Cloud.

<img src="./images/HDLFS_Instantiate_1.png" width="500" alt="Provision a new instance of SAP HANA Cloud">
<img src="./images/HDLFS_Instantiate_2.png" width="500" alt="Insert a system password when provisioning SAP HANA Cloud">

We would also recommend that you provision the **tools** plan for SAP HANA Cloud to better navigate its features. 

<img src="./images/HDLFS_Instantiate_3.png" width="500" alt="Provision the UI for SAP HANA Cloud">

### Accessing the SAP HANA Cloud UI
You should be able to see 1 instance in the SAP HANA Cloud dashboard once everything has been properly setup.

<img src="./images/HDLFS_Dashboard.png" width="1000" alt="SAP HANA Cloud dashboard">


Select the instance and click **Manage Configuration**.

<img src="./images/HDLFS_Instance.png" width="1000" alt="Click Manage Configuration to edit some settings on the HDLFS instance">

Make sure to select "Allow all IP addresses" under the **Connections** section.

<img src="./images/HDLFS_Allow all.png" width="1000" alt="Select 'Allow all IP addresses'">

You can refer to Steps 3 and 4 on how to generate the certificate and update the trust configuration in this [developer tutorial](https://developers.sap.com/tutorials/data-lake-file-containers-hdlfscli/) for HANA Data Lake File System. After following with these steps, you should have added the generated certificate under the **Trusts** section as well as a new authoriation with a binding to the certificate under the **Authorizations** section.

<img src="./images/HDLFS_Add certificate.png" width="1000" alt="Add generated certificate to the settings">

Make sure to save the new settings.

### Getting the HDLFS endpoint for registering it as a Data Destination
You will require the HDLFS endpoint when registering your HDLFS instance as the data destination to run the Tabular AI orchestration workflow. Head to **Data Lake Files** and you should see a line item under **File Containers**. Use this endpoint for registrating the data destination.

<img src="./images/HDLFS_Endpoint.png" width="1000" alt="You should 3 line items under 'Authorizations'">

### Adding subject patterns to your HDLFS instance after registering it as a Data Destination

When you have created a new data destination, you should receive 2 patterns in the response payload. Add these 2 patterns to the settings in your HDLFS instance. All in all, you should have 3 items under **Authorizations**.

<img src="./images/HDLFS_Trust list.png" width="1000" alt="You should 3 line items under 'Authorizations'">

Make sure to save the new settings. 

## API Collection
We have additionally provided a set of API calls, available as a Bruno collection or Postman collection. You can find these files in the **api** folder. You can download your preferred file and get started with using SAP-RPT-1.6, TabPFN-3.5 Plus and Tabular AI orchestration workflow via APIs.

We have created environment variables. You can fill it in with the values from your own setup before running the APIs.