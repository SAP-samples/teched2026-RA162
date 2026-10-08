# Exercise 1

In this exercise, we will create run SAP-RPT-1.6 with some sample data. You will learn to process tabular data from [used_car_sales](used_car_sales.csv) and to use the RPT API to prexict the price of a car. 

### sample data

| Car Brand | Model               | Price  | Model Year | Location  | Fuel   | Driven (Kms) | Gear      | Ownership | EMI (monthly)  |
|-----------|---------------------|--------|------------|-----------|--------|--------------|-----------|-----------|----------------|
| Hyundai   | EonERA PLUS         | 330399 | 2016       | Hyderabad | Petrol | 10674        | Manual    | 2         | 7350           |
| Maruti    | Wagon R 1.0LXI      | 350199 | 2011       | Hyderabad | Petrol | 20979        | Manual    | 1         | 7790           |
| Maruti    | Alto K10LXI         | 229199 | 2011       | Hyderabad | Petrol | 47330        | Manual    | 2         | 5098           |
| Maruti    | RitzVXI BS IV       | 306399 | 2011       | Hyderabad | Petrol | 19662        | Manual    | 1         | 6816           |
| Tata      | NanoTWIST XTA       | 208699 | 2015       | Hyderabad | Petrol | 11256        | Automatic | 1         | 4642           |
| Maruti    | AltoLXI             | 249699 | 2012       | Hyderabad | Petrol | 28434        | Manual    | 1         | 5554           |
| Maruti    | AltoLXI             | 240599 | 2011       | Hyderabad | Petrol | 31119        | Manual    | 1         | 5352           |
| Maruti    | Alto K10LXI         | 191999 | 2010       | Hyderabad | Petrol | 10910        | Manual    | 1         | 4271           |
| Honda     | Brio1.2 S MT I VTEC | 362299 | 2013       | Hyderabad | Petrol | 40362        | Manual    | 2         | 8059           |


## Requirements

Access credentials to AI Core is required. In particular, the following credentials are needed.
```
AICORE_AUTH_URL
AICORE_CLIENT_ID
AICORE_CLIENT_SECRET

BASE_DEPLOYMENT_URL
RPT1.6_DEPLOYMENT_ID
```

Once you have all creditals set up, you can start in the [Jupyter Notebook part (1)](RPT_notebook_TechEd2026_part1_prediction.ipynb) which focusses on the core functionalities of the RPT model. Afterwards, you may start the [Jupyter Notebook party (2)](RPT_notebook_TechEd2026_part2_explainability.ipynb) which focusses on the explanibility feature.

