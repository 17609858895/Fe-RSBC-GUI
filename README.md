# Fe@RSBC-β-CD TC adsorption predictor

This Streamlit application deploys the final Extra Trees (ET) regression model
used in the revised manuscript. The model consumes five predictors in the fixed
order `C0`, `Time`, `pH`, `Dosage`, and `Temp` and returns the predicted TC
adsorption capacity.

The model bundle contains its feature order, experimental-domain limits, model
metadata, and the fitted estimator. No imputer, power transform, or scaler is
applied because the 49-row source matrix is complete and the tree ensemble does
not require feature scaling.

Validation is reported separately from deployment. Model selection used 20
group-aware 80:20 random splits and 10 repeats of nested five-fold group
cross-validation, with identical experimental conditions retained in the same
group. The interface therefore describes the model as an interpolation tool
within the experimental domain, not as a validated extrapolation model.
