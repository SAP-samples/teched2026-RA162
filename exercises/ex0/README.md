# Getting Started on Tabular AI

To access our tabular foundation models (SAP-RPT models, TabPFN models), there are 2 possible modes of consumption:
1. Via direct API call to each model (Exercises 1 and 2)
2. Via API call to the Tabular AI orchestration workflow (Exercise 3)

Before starting on the exercises, you should first set up a virtual environment. Activate the virtual environment and then download all required libraries. Then, prepare the `.env` file with all required credentials before p

You can follow the setup steps below.

## Prepare Credentials
You should see the `credentials.zip` file in the `exercises` folders.

Unzip it with the provided password and you should see the `.env` file. It contains all the credentials required for the exercises. 

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

## Conclusion
Now, you should be ready to proceed to [Exercise 1 - SAP-RPT-1.6](../ex1/README.md) to get started with our exercises. 

## Additional Notes
If you would like to replicate the exercises on your own setup, do note that there are additional steps to set up AI Core, AI Launchpad and HANA Cloud. You can see the instructions in the [Additional_Instructions.md](/exercises/ex0/Additional_Instructions.md).