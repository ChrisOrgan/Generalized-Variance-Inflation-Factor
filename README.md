# Generalized-Variance-Inflation-Factor GVIF)
- Variance inflation factors are not fully applicable to models that include a set of one-hot encoded regressors (i.e. recoded multinomial categorical variables), or polynomial regressors.
- Generalized VIF (GVIF): Fox and Monette 1992; is a solution
    - GVIF = det(R11) * det(R22) / det(R)
        where:
            R11 is the correlation matrix for X1
            R22 is the correlation matrix for X2 (rest of independent variables)
            R   is the correlation matrix for all variables in the whole design matrix X, excluding the constant (b0)
    - GVIF ** (1 / (2 * Df)) ** 2 is used as the usual VIF rule of thumb
        - E.g., [GVIF ** (1 / (2 * Df)) ** 2] < 5 is equivalent to VIF < 5 for the continuous (i.e. non-categorical) variables
