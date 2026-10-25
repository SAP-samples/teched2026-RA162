# Exercise 3 - Tabular AI Orchestration

In this exercise, you will use the provided Python notebook to learn how to set up the Tabular AI orchestration workflow to run predictions with the SAP-RPT-1.6 model at scale. We will be reusing the same dataset from Exercise 1. 

Tabular AI Orchestration has three components:

```
Your App
   |
   v
Tabular AI Orchestration  <-- single endpoint
   |              |
   v              v
Context        SAP-RPT model
Selector       (model of your choice)
   |
Context Registry
(your registered dataset)
```

## Exercise 3.1 - Set up Tabular AI Orchestration (Design Time Steps)

After completing these steps, you will have uploaded the dataset to HANA Data Lake File system and also created:
1. A data destination
2. A tabular artifact
3. A scenario configuration

Now, you are ready to try running the workflow and obtain predictions from the SAP-RPT model.

## Exercise 3.2 - Live Predictions with Tabular AI Orchestration (Run Time Steps)

Note: Tabular AI orchestration has already been deployed for you. Simply use the provided deployment ID to start running predictions during the session. If you would like to repeat this on your own setup, we have provided a code block that you can use (currently greyed out).

After completing these steps, you will have successfully obtained model predictions from SAP-RPT-1.6 via the Tabular AI orchestration workflow

## Summary

Great, you have now completed all the exercises for Tabular AI.
