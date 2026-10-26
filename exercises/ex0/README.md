# Getting Started on Tabular AI

To access our tabular foundation models (SAP-RPT models, TabPFN models), there are 2 possible modes of consumption:
1. Via direct API call to each model (Exercises 1 and 2)
2. Via API call to the Tabular AI orchestration workflow (Exercise 3)

Before starting on the exercises, you should first set up a virtual environment. Activate the virtual environment and then download all required libraries. Then, prepare the `.env` file with all required credentials before p

You can follow the setup steps below.

## Setting up a Virtual Environment for the exercises

Open the terminal in Visual Studio Code. 

Run the following command to create the local environment.
```
Windows: py -m venv venv
Mac: python3 -m venv venv
```

You should see that a `venv` folder is created in the root directory. 

Run the following command to activate the virtual environment.
```
Windows: venv\Scripts\activate
Mac: source venv/bin/activate
```

Once the virtual environment is up, install the required libraries using the provided `requirements.txt` file.
```
cd exercises
pip install -r requirements.txt
```

## Preparing the .env file

You should see a `.env.example` file in the `exercises` folder. Rename the file to `.env` and open this file. You would see that we have already prepared some environment variables in the file:
```
AICORE_AUTH_URL=<url in RA162-SK.json file>/oauth/token
AICORE_CLIENT_ID=<clientid in RA162-SK.json file>
AICORE_CLIENT_SECRET=<clientsecret in RA162-SK.json file>
AICORE_RESOURCE_GROUP=default
AICORE_BASE_URL=<AI_API_URL in RA162-SK.json file>
BASE_DEPLOYMENT_URL=<AI_API_URL in RA162-SK.json file>/v2/inference/deployments
RPT_DEPLOYMENT_ID=<sap-rpt-1.6 in RA162-SK.json file>
RPT_LARGE_DEPLOYMENT_ID=<sap-rpt-1.6-large in RA162-SK.json file>
TABPFN_DEPLOYMENT_ID=<tabpfn-3.5-plus in RA162-SK.json file>
TABORC_DEPLOYMENT_ID=<tabular-orchestration in RA162-SK.json file>
TABORC_HDLFS_HOST=<hdlfs-host in RA162-SK.json file>
TABORC_HDLFS_URL=<hdlfs-url in RA162-SK.json file>
```

Unzip `credentials.zip` file and open the `RA162-SK.json` file. Copy the required values from this JSON file into your `.env` file.

## Conclusion
Now, you should be ready to proceed to [Exercise 1 - SAP-RPT-1.6](../ex1/README.md) to get started with our exercises. 

## Additional Notes
If you would like to replicate the exercises on your own setup, do note that there are additional steps to set up AI Core, AI Launchpad and HANA Cloud. You can see the instructions in the [Additional_Instructions.md](/exercises/ex0/Additional_Instructions.md).