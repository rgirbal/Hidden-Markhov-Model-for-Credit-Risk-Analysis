import numpy as np
import pandas as pd
import matplotlib.pyplot as pltype('float'))))


default_data = pd.ExcelFile('defaultdata.xlsx').parse()
default_data = default_data.groupby(['date'])['date'].count().to_frame('num_defaults')
default_data.index = pd.to_datetime(default_data.index)

#default data set to have same index as risk factors with non-default dates labelled as 0's
ix = pd.DatetimeIndex(start=pd.datetime.date(pd.to_datetime("1970-01-01")), end=pd.datetime.date(pd.to_datetime("2012-12-31")), freq="D")
default_data = default_data.reindex(ix, fill_value=0)
default_data["cum_num_defaults"] = default_data.cumsum() #adds cumulative defaults as a column, helps calculate T_n

default_date_array = default_data.values.reshape((15706,2))

default_value_PLACEHOLDER_array = np.ones(default_date_array.shape)

np.argwhere(default_date_array> 1998)[0][0]


def get_dataframe_from_file(file_name):
    df = pd.ExcelFile(file_name).parse().set_index(['date']).dropna()
    df.index = pd.to_datetime(df.index)
    return df


def interpolate_by_date(risk_factor):
    return risk_factor.reindex(pd.date_range(start=risk_factor.index.min(),
                                             end=risk_factor.index.max(),
                                             freq='1D')).interpolate(method='linear')


def preprocess_risk_factors():
    ip = get_dataframe_from_file('INDPRO.xls')
    gdp = get_dataframe_from_file('A191RL1Q225SBEA.xls')
    level = get_dataframe_from_file('DTB3.xls')
    ten_y = get_dataframe_from_file('DGS10.xls')
    slope = get_dataframe_from_file('DGS10Y1Y.xls')
    aaa = get_dataframe_from_file('WAAA.xls')
    cred = get_dataframe_from_file('WCRED.xls')
    rec = get_dataframe_from_file('USREC.xls')

    gspc = get_dataframe_from_file('^GSPC.xls')['Adj Close']
    ret = gspc.pct_change(periods=250).dropna()
    vol = gspc.pct_change().dropna().rolling(250).std().dropna()
    ret.name, vol.name = 'RET', 'VOL'

    risk_factors = [ip, gdp, level, ten_y, slope, aaa, cred, rec, ret, vol]
    for i, rf in enumerate(risk_factors):
        risk_factors[i] = interpolate_by_date(rf)

    risk_factors = pd.concat(risk_factors, axis=1)
    return risk_factors.dropna()

df_risk_factors = np.array(preprocess_risk_factors().values)

d = df_risk_factors.shape[1]
a = np.ones((d+1, ))

# Initialize parameters for model

NUM_RISK_FACTORS = df_risk_factors.shape[1] #known as d in the paper

OBSERVABLES_COEFFECIENTS = np.ones((NUM_RISK_FACTORS + 1, )) #known as a = (a_0, ..., a_d) in the paper

LATENT_VAR_REVERSION_COEFFECIENT = 1 #known as k in the paper

CONTAGION_VAR_REVERSION_COEFFECIENT = 1 #known as kappa in the paper

SIGMA = 1 #also known unintentionally as c in the paper but thats ok

CONTAGION_COEFFECIENT = 1 #known as b in the paper

U = default_value_PLACEHOLDER_array


# DERIVED PARAMETERS

def zeta(LATENT_VAR_REVERSION_COEFFECIENT, SIGMA):  # defined when approximating lieklihood function where c = SIGMA
    return np.sqrt(np.square(LATENT_VAR_REVERSION_COEFFECIENT) + 2 * np.square(SIGMA))


# defined when approximating lieklihood function where c = SIGMA
# Gives order that we use for the Bessel Function
def order(LATENT_VAR_REVERSION_COEFFECIENT, SIGMA, z):
    return (2 * LATENT_VAR_REVERSION_COEFFECIENT * z) / (np.square(SIGMA)) - 1


def contagionY(time):
    SUM = 0
    for day in range(time):
        # U_n here is simply assumed to be 1.
        SUM += np.exp(-CONTAGION_VAR_REVERSION_COEFFECIENT * (time - day)) * default_date_array[0][day]
    return CONTAGION_COEFFECIENT * SUM


def big_PHI(z1, t1, z2, t2, z):
    ZETA = zeta(LATENT_VAR_REVERSION_COEFFECIENT, SIGMA)
    DELTA = t2 - t1
    q = order(LATENT_VAR_REVERSION_COEFFECIENT, SIGMA, z)
    K = LATENT_VAR_REVERSION_COEFFECIENT

    numerator1 = sp.special.iv(q, np.sqrt(z1 * z2) * (4 * ZETA * np.exp(-0.5 * ZETA * DELTA)) / (
                1 - np.exp(-ZETA * DELTA)))
    denominator1 = sp.special.iv(q, np.sqrt(z1 * z2) * (4 * K * np.exp(-0.5 * K * DELTA)) / (1 - np.exp(-K * DELTA)))

    numerator2 = ZETA * np.exp(-0.5 * (ZETA - K) * DELTA) * (1 - np.exp(-K * DELTA))
    denominator2 = K * (1 - np.exp(-ZETA * DELTA))

    def expFRAC(var):
        return (var * (1 + np.exp(-var * DELTA))) / (1 - np.exp(-var * DELTA))

    exponent = np.exp((z1 + z2) * (expFRAC(K) - expFRAC(ZETA)))

    return (numerator1 / denominator1) * (numerator2 / denominator2) * exponent


def phi_integral(t1, t2):
    SUM = 0
    for day in range(t1, t2 + 1):  # Should delta t be mesured yearly, monthly or daily??? Currently daily so = 1
        SUM += np.exp(OBSERVABLES_COEFFECIENTS[0]
                      + OBSERVABLES_COEFFECIENTS[1:].dot(df_risk_factors_array[day])) + contagionY(day)

    return SUM


def little_PHI(z1, t1, z2, t2):
    return big_PHI(z1, t1, z2, t2, z) * np.exp(-phi_integral(t1, t2))


def F_M_tau(n, k, l, z_state):
    part1 = np.exp(np.argwhere(default_date_array <= n)[0][0] - np.argwhere(default_date_array <= n - 1)[0][0])
    part2 = np.exp(OBSERVABLES_COEFFECIENTS[0]
                   + OBSERVABLES_COEFFECIENTS[1:].dot(
        df_risk_factors_array[np.argwhere(default_date_array < n)[-1][0]]))
    + contagionY(np.argwhere(default_date_array < n)[-1][0]) + SIGMA * z_state[k]


part3 = little_PHI(z_state[l], np.argwhere(default_date_array <= n - 1)[0][0], z_state[k],
                   np.argwhere(default_date_array <= n)[0][0])

return part1 * part2 * part3