# Exercise 3 - Tabular AI Orchestration

In this exercise, you will learn how to use our Tabular AI Orchestration capabilities to connect your data source directly to the SAP-RPT model. There are 3 sub-exercises to follow along. 

## Exercise 3.1 Set up your data destination

After completing these steps you will have created...

1. Click here.
<br>![]()

2.	Insert this line of code.
```abap
response->set_text( |Hello ABAP World! | ). 
```



## Exercise 3.2 Set up the configuration for Tabular Orchestration

After completing these steps you will have...

1.	Enter this code.
```abap
DATA(lt_params) = request->get_form_fields(  ).
READ TABLE lt_params REFERENCE INTO DATA(lr_params) WITH KEY name = 'cmd'.
  IF sy-subrc = 0.
    response->set_status( i_code = 200
                     i_reason = 'Everything is fine').
    RETURN.
  ENDIF.

```

2.	Click here.
<br>![]()




## Exercise 3.3 Get model predictions with Tabular Orchestration

After completing these steps you will have...

1.	Enter this code.
```abap
DATA(lt_params) = request->get_form_fields(  ).
READ TABLE lt_params REFERENCE INTO DATA(lr_params) WITH KEY name = 'cmd'.
  IF sy-subrc = 0.
    response->set_status( i_code = 200
                     i_reason = 'Everything is fine').
    RETURN.
  ENDIF.

```

2.	Click here.
<br>![]()

## Summary

You've now ...

