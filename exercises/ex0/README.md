# Getting Started

...

## Familiarise yourself with AI Launchpad

### Accessing AI Launchpad
After completing these steps, you should arrive at the AI Core Launchpad.

1. Access AI Launchpad [here](https://ra162-ztwbn9xc.ai-launchpad.prod.eu-central-1.aws.ai-prod.cloud.sap/aic/index.html#/workspaces). 
2. Log in with the username and password provided with each device.
3. You should now see the SAP AI Launchpad.
4. Select **ra162** from list of AI API connections.
5. Select the **default** resource group.

For the hands-on session, we have already deployed the models. You will need to retrieve the deployment URLs for each model before proceed with Exercises 1 and 2.
1. Head to **Deployments** under **ML Operations** via the left navigation bar.
2. You should see 2 new model deployments as follows.
3. Click into each deployment and you should use a deployment URL.
4. For Exercise 1, you will need to copy the deployment URL for the SAP-RPT-1.6 model. Likewise for Exercise 2, but with the deployment URL for the TabPFN-3.5 Plus model.

If you would like to repeat this tutorial on your own, you will need to deploy the models on your own. Here are the steps to do so.

### Deploying the models
1. Head to **Model Library** under **Generative AI Hub** via the left navigation bar.
2. Within the **Model Library**, search for SAP RPT 1.6.
3. Click on the SAP RPT 1.6 model. 
4. Click Deploy.
5. Repeat steps 2 to 4 to deploy the TabPFN-3.5 Plus model.
6. Head to **Deployments** under **ML Operations** via the left navigation bar.
7. You should see the 2 new model deployments that you have created.
8. Once deployment is completed, you should see that the status indicates **RUNNING**.

## Summary
Now, you should have noted down the model deployment URLs. Head to [Exercise 1](../ex1/README.md) to get started with the SAP-RPT-1.6 model.
