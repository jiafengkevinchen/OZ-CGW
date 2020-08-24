library(plm)
library(lmtest)
library(multiwayvcov)
library(car)
library(lfe)

fit_did_no_pretest <- function(fmla, data, index = c("tract", "year")) {
  model <- plm(formula = fmla, data = data, model = "within", index = index)
  vcv <- vcovHC(model, type = "sss", cluster = "group")
  coeftest_model <- coeftest(model, vcov = vcv)
  return(list(
    model = model,
    vcv = vcv, 
    coeftest_model = coeftest_model
  ))
}


fit_did <- function(fmla, pretest_fmla, pretest_cols,
                    data, index = c("tract", "year")) {
  model_pretest <- plm(
    formula = pretest_fmla,
    data = data, model = "within",
    index = index
  )

  vcv_pretest <- vcovHC(model_pretest, type = "sss", cluster = "group")
  coeftest_pretest <- coeftest(model_pretest, vcv_pretest)
  lh_pretest <- linearHypothesis(model_pretest, pretest_cols, vcov. = vcv_pretest)

  model <- plm(formula = fmla, data = data, model = "within", index = index)
  vcv <- vcovHC(model, type = "sss", cluster = "group")
  coeftest_model <- coeftest(model, vcov = vcv)

  return(list(
    model_pretest = model_pretest,
    vcv_pretest = vcv_pretest,
    coeftest_pretest = coeftest_pretest,
    lh_pretest = lh_pretest,
    model = model,
    vcv = vcv,
    coeftest_model = coeftest_model
  ))
}



fit_did_lfe <- function(fmla, pretest_fmla, pretest_cols, data) { 
	model_pretest <- felm(pretest_fmla, data=data)
	pretest <- waldtest(model_pretest, pretest_cols, type="cluster")
	model <- felm(fmla, data=data)
	return(list(
		    model_pretest = model_pretest,
		    lh_pretest = pretest,
		    model = model)
	)
}

