import matplotlib.pyplot as plt
from lifelines import LogNormalFitter, LogLogisticFitter, WeibullFitter, ExponentialFitter
import pandas as pd
data = pd.read_csv('Dataset/telco_customer.csv')
data['tenure'] = pd.to_numeric(data['tenure'])
data = data[data['tenure'] > 0]
data['Churn'] = data['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
T = data['tenure']
E = data['Churn']
fig, axes = plt.subplots(2, 2, figsize=(16, 12))


lnf = LogNormalFitter()  # instantiate the class to create an object of lognormal model


groups = data['Partner']
k1 = (groups == 'No')  # group i1 , having the pandas series for the 1st cohort
k2 = (groups == 'Yes')  # group i2 , having the pandas series for the 2nd cohort


# fit the model for 1st cohort
lnf.fit(T[k1], E[k1], label='Do not have a partner')
a1 = lnf.plot(ax=axes[0][0])

# fit the model for 2nd cohort
lnf.fit(T[k2], E[k2], label='Have a partner')
lnf.plot(ax=axes[0][0])

# ------------------------------------------------------------------------


wbf = WeibullFitter()  # instantiate the class to create an object of Weibull model


groups = data['Partner']
k1 = (groups == 'No')  # group i1 , having the pandas series for the 1st cohort
k2 = (groups == 'Yes')  # group i2 , having the pandas series for the 2nd cohort


# fit the model for 1st cohort
wbf.fit(T[k1], E[k1], label='Do not have a partner')
a1 = wbf.plot(ax=axes[0][1])

# fit the model for 2nd cohort
wbf.fit(T[k2], E[k2], label='Have a partner')
wbf.plot(ax=axes[0][1])

# ------------------------------------------------------------------------


# instantiate the class to create an object of loglogistic model
llf = LogLogisticFitter()


groups = data['Partner']
k1 = (groups == 'No')  # group i1 , having the pandas series for the 1st cohort
k2 = (groups == 'Yes')  # group i2 , having the pandas series for the 2nd cohort


# fit the model for 1st cohort
llf.fit(T[k1], E[k1], label='Do not have a partner')
a1 = llf.plot(ax=axes[1][0])

# fit the model for 2nd cohort
llf.fit(T[k2], E[k2], label='Have a partner')
llf.plot(ax=axes[1][0])

# ------------------------------------------------------------------------


# instantiate the class to create an object of exponential model
ef = ExponentialFitter()


groups = data['Partner']
k1 = (groups == 'No')  # group i1 , having the pandas series for the 1st cohort
k2 = (groups == 'Yes')  # group i2 , having the pandas series for the 2nd cohort


# fit the model for 1st cohort
ef.fit(T[k1], E[k1], label='Do not have a partner')
a1 = ef.plot(ax=axes[1][1])

# fit the model for 2nd cohort
ef.fit(T[k2], E[k2], label='Have a partner')
ef.plot(ax=axes[1][1])

# ------------------------------------------------------------------------

axes[0, 0].set_title("LogNatural")
axes[0, 1].set_title("Weibull")
axes[1, 0].set_title("LogLogistic")
axes[1, 1].set_title("Exponential")


plt.suptitle(
    'Using different parametric models to see the churn probability on the basis if the user has a partner or not')

fig.text(0.5, 0.04, 'Timeline', ha='center')
fig.text(0.04, 0.5, 'Probability', va='center', rotation='vertical')
plt.savefig('Images/Churn_by_Partner.jpeg')
plt.show()
