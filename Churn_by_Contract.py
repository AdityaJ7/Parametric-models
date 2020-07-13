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

lnf = LogNormalFitter()


groups = data['Contract']
x1 = (groups == 'Month-to-month')
x2 = (groups == 'Two year')
x3 = (groups == 'One year')

lnf.fit(T[x1], E[x1], label='Month-to-month')
ax = lnf.plot(ax=axes[0][0])


lnf.fit(T[x2], E[x2], label='Two year')
ax1 = lnf.plot(ax=axes[0][0])
ac1 = lnf.plot


lnf.fit(T[x3], E[x3], label='One year')
lnf.plot(ax=axes[0][0])


wbf = WeibullFitter()


groups = data['Contract']
x1 = (groups == 'Month-to-month')
x2 = (groups == 'Two year')
x3 = (groups == 'One year')

wbf.fit(T[x1], E[x1], label='Month-to-month')
ax = wbf.plot(ax=axes[0][1])


wbf.fit(T[x2], E[x2], label='Two year')
ax1 = wbf.plot(ax=axes[0][1])


wbf.fit(T[x3], E[x3], label='One year')
wbf.plot(ax=axes[0][1])

llf = LogLogisticFitter()


groups = data['Contract']
x1 = (groups == 'Month-to-month')
x2 = (groups == 'Two year')
x3 = (groups == 'One year')

llf.fit(T[x1], E[x1], label='Month-to-month')
ax = llf.plot(ax=axes[1][0])


llf.fit(T[x2], E[x2], label='Two year')
ax1 = llf.plot(ax=axes[1][0])


llf.fit(T[x3], E[x3], label='One year')
llf.plot(ax=axes[1][0])

ef = ExponentialFitter()


groups = data['Contract']
x1 = (groups == 'Month-to-month')
x2 = (groups == 'Two year')
x3 = (groups == 'One year')

ef.fit(T[x1], E[x1], label='Month-to-month')
ax = ef.plot(ax=axes[1][1])


ef.fit(T[x2], E[x2], label='Two year')
ax1 = ef.plot(ax=axes[1][1])


ef.fit(T[x3], E[x3], label='One year')
ef.plot(ax=axes[1][1])

axes[0, 0].set_title("LogNatural")
axes[0, 1].set_title("Weibull")
axes[1, 0].set_title("LogLogistic")
axes[1, 1].set_title("Exponential")

plt.suptitle(
    'Using different parametric models to see the churn probability on the basis of User Contract with the company')

fig.text(0.5, 0.04, 'Timeline', ha='center')
fig.text(0.04, 0.5, 'Probability', va='center', rotation='vertical')
plt.savefig('Images/Churn_by_Contract.jpeg')
plt.show()
