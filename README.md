# Data Competition Instructions

Contact: Ben Taylor, @bentaylordata, www.linkedin.com/in/bentaylordata/

Files:

  * train_X.csv
  * train_Y.csv
  * val_X.csv
  * val_Y.csv

**train_X.csv:**
This file contains a binary matrix that is 2,000 rows by 10,000 columns. This matrix represents where each wafer ran moving through the fab during the past week. The 2,000 rows represent 2,000 unique contexts where the wafer could have run. These contests can represent tools, chambers, reticles, or any categorical input that could vary between wafers.

**train_Y.csv:**
This file contains the synthetic yield output generated by the wafers. The maximum possible die for theses wafers is 600.

##Predict the next point:
Because there is such a strong time factor the best metric that could be used to judge whether a model is working would be predicting the next wafer to yield. This would not be fair in a competition setting because a single observation prediction could be driven by luck. So to be fair, since this is a synthetic dataset, we can generate 10,000 potential inputs and yields that could have occurred for the next wafer.

**val_X.csv**
This file represents 10,000 different potential inputs for the very next wafer in the series. Run them through your model that has been trained using train_X.csv and train_Y.csv and compare them to val_Y.csv to score. For the final scoring a new dataset, which will be similar, will be given, where val_Y.csv is not provided but is used to score the entries.

**val_Y.csv**
This file represents the yield on the 10,000 potential wafers that could have run. This is the file you will use to compare your prediction against to score the value of your model.

#**!!! IMPORTANT CUSTOM ERROR FUNCTION <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< THIS IS WHAT YOU ARE BEING SCORED ON**
This problem has a custom error function. This forces you to look at type-I and type-II errors in your modeling. So to do this any under prediction residuals get a 10x multiplier penalty added to them. Here is a Matlab example:

```
%Special cost function, under prediction is 10x penalty
err_LSR=yp_LSR-sim_yield_val_die_loss;              %Calculate errors
err_WLSR=yp_WLSR-sim_yield_val_die_loss;

%Custom residual sum
err_underpredict_LSR=abs(err_LSR(err_LSR<0))*10;        %10x penalty
err_overpredict_LSR=abs(err_LSR(err_LSR>=0));
err_underpredict_WLSR=abs(err_WLSR(err_WLSR<0))*10;        %10x penalty
err_overpredict_WLSR=abs(err_WLSR(err_WLSR>=0));

overall_score_LSR=(sum(err_underpredict_LSR)+sum(err_overpredict_LSR))/wafer_count
overall_score_WLSR=(sum(err_underpredict_WLSR)+sum(err_overpredict_WLSR))/wafer_count
```