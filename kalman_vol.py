# load packages

import matplotlib
import pandas as pd
import pandas_datareader as pdr
from IPython.display import set_matplotlib_formats
from matplotlib import pyplot as plt
from sklearn import linear_model
import statsmodels.api as sm
import numpy as np
from pandas_datareader import DataReader
from pykalman import KalmanFilter

np.random.seed(7557)  # Random number generator

# ploting setup
plt.style.use(['seaborn-white', 'seaborn-paper'])
matplotlib.rc('font', family='Times New Roman', size=15)
set_matplotlib_formats('png', 'png', quality=80)
plt.rcParams['savefig.dpi'] = 100
plt.rcParams['figure.autolayout'] = False
plt.rcParams['figure.figsize'] = 10, 6
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['font.size'] = 12
plt.rcParams['lines.linewidth'] = 1.0
plt.rcParams['lines.markersize'] = 8
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.serif'] = 'cm'
plt.rcParams['axes.grid'] = True
kw_save = dict(bbox_iches='tight', transparent=True)


# useful functions
# ================True
def total_return(prices):
    """Returns the return between the first and last value of the DataFrame.

    Parameters
    ----------
    prices : pandas.Series or pandas.DataFrame

    Returns
    -------
    total_return : float or pandas.Series
        Depending on the input passed returns a float or a pandas.Series.
    """
    return prices.iloc[-1] / prices.iloc[0] - 1


# Getting data
# ===========
end_date = '20191227'  # end date year month day
tckrs = ['MSFT', 'AAPL', 'BAC', 'XOM', 'F', 'HAL']
for tckr in tckrs:
    # download data
    data = pdr.get_data_yahoo(tckr, 1990, end_date)  # start year
    data = data.asfreq('B')  # add frequency needed for some pandas functionalities related with offsets
    data.columns = data.columns.map(lambda col: col.lower())
    # # what about NaNs
    data.isnull().sum()  # will give column-wise sum of missing values
    data.ffill(inplace=True)  # to avoid problems with NaNs.
    # using close prices
    data = data.rename(columns={"adj close": "adjclose"})

    prices = data.adjclose.copy()
    # convert to DataFrame to make easy store more series.
    results_storage = prices.to_frame().copy()

    target = results_storage.asfreq('B') \
        .adjclose \
        # need log return attribute for target df
    target = target.to_frame()
    logdf = np.log(target['adjclose']) - np.log(target['adjclose'].iloc[0])
    y = logdf.rename(columns={"adjclose": "logreturns"})  # need concatenate with target

    df = results_storage.asfreq('B') \
        .adjclose \
        .pct_change() \
        .rolling(10) \
        .std(ddof=0)

    df = df.shift(-1)

    data = pd.concat([y, df], axis=1)
    data.columns = ['logreturns', 'volatility']


    for col in data.columns:
        print(col)

    # fit a regression model using SKLearn. define X and y
    X = df.values.reshape(-1, 1)
   # y = y.reshape(-1, 1)



    X = X[11:-1]
    y = y[11:-1]
    data = data[11:-1]



    # fit a model:
    lm = linear_model.LinearRegression()
    model = lm.fit(X, y)

    predictions = lm.predict(X)

    print(lm.score(X, y), "r-square")
    print(lm.coef_, "coefficient")

    df_pred = pd.DataFrame(predictions, columns=['yhat'])
    # df_yobs = pd.DataFrame(y,columns=['yobs'])
    df_pred['yobs'] = y

    # bigdata = df_pred.concat(df_yobs, axis=1)

    df_pred.plot(kind='line')
    plt.show()

    X = sm.add_constant(X)  # Add a column of ones for constant

    model = sm.OLS(y, X)
    results = model.fit()
    print(results.summary())
    print("Regression for: " + tckr + "completed successfully")

    # Start Kalman Setion

    #pd.set_option('display.mpl_style',  'default')
    matplotlib.rcParams['figure.figsize'] = 8, 6

    # get adjusted close prices from Yahoo
    # secs = ['EWA', 'EWC']
    #     # data = DataReader(secs, 'yahoo', '2010-1-1', '2014-8-1')['Adj Close']

    # visualize the correlation between assest prices over time
    cm = plt.cm.get_cmap('jet')
    dates = [str(p.date()) for p in data[::len(data)//10].index]
    colors = np.linspace(0.1, 1, len(data))
    sc = plt.scatter(data[data.columns[0]], data[data.columns[1]], s=30, c=colors, cmap=cm, edgecolor='k', alpha=0.7)
    cb = plt.colorbar(sc)
    cb.ax.set_yticklabels([str(p.date()) for p in data[::len(data)//9].index]);
    plt.xlabel(data.columns[0])
    plt.ylabel(data.columns[1])
    plt.show()
    # /plt.savefig('price_corr.png')

    delta = 1e-5
    trans_cov = delta / (1 - delta) * np.eye(2)
    obs_mat = np.vstack([data.logreturns, np.ones(data.volatility.shape)]).T[:, np.newaxis]
    kf = KalmanFilter(n_dim_obs=1, n_dim_state=2,
                      initial_state_mean=np.zeros(2),
                      initial_state_covariance=np.ones((2, 2)),
                      transition_matrices=np.eye(2),
                      observation_matrices=obs_mat,
                      observation_covariance=1.0,
                      transition_covariance=trans_cov)

    state_means, state_covs = kf.filter(data.volatility.values)

    pd.DataFrame(dict(slope=state_means[:, 0], intercept=state_means[:, 1]), index=data.index).plot(subplots=True, title=tckr)
    plt.tight_layout()
    plt.show()
    # plt.savefig('slope_intercept.png')

    # # visualize the correlation between assest prices over time
    # cm = plt.cm.get_cmap('jet')
    # dates = [str(p.date()) for p in data[::len(data)/10].index]
    # colors = np.linspace(0.1, 1, len(data))
    # sc = plt.scatter(data[data.columns[0]], data[data.columns[1]], s=50, c=colors, cmap=cm, edgecolor='k', alpha=0.7)
    # cb = plt.colorbar(sc)
    # cb.ax.set_yticklabels([str(p.date()) for p in data[::len(data)//9].index]);
    # plt.xlabel(data.columns[0])
    # plt.ylabel(data.columns[1])
    #
    # # add regression lines
    # step = 5
    # xi = np.linspace(data[data.columns[0]].min(), data[data.columns[0]].max(), 2)
    # colors_l = np.linspace(0.1, 1, len(state_means[::step]))
    # for i, beta in enumerate(state_means[::step]):
    #     plt.plot(xi, beta[0] * xi + beta[1], alpha=.2, lw=1, c=cm(colors_l[i]))

    plt.savefig('price_corr_regress.png')

    print("Kalman for: " + tckr + "completed successfully")

    mod = KalmanFilter(y)
    res = mod.fit()
    print(res.summary())