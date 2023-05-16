import numpy as np
import pandas as pd
import math
from math import sqrt
import scipy.special as spec
import warnings
from scipy.stats import norm

sample_data = [
    50.7914,
    45.5242,
    48.7432,
    49.7271,
    49.5492,
    50.2307,
    47.3780,
    50.9415,
    49.3025,
    50.2963,
    49.7309,
    48.4764,
    49.6597,
    49.3094,
    49.2337,
    49.9493,
    50.7786,
    49.0395,
    49.6319,
    51.8398,
    50.9974,
    47.4333,
    47.4708,
    50.6008,
    49.2797,
    51.3829,
    50.0659,
    51.8788]

sample_data_subgroups = [    	
(1, 48.7062), 
(1, 52.8217), 
(1, 50.7195), 
(1, 51.6153), 
(1, 45.9505), 
(1, 47.2057), 
(1, 50.1438), 
(2, 49.4001), 
(2, 50.3203), 
(2, 48.9165), 
(2, 53.5732), 
(2, 49.6896), 
(2, 50.1535), 
(2, 52.1338), 
(3, 52.6277), 
(3, 49.2152), 
(3, 51.5872), 
(3, 49.7418), 
(3, 47.2276), 
(3, 50.6780), 
(3, 45.4440), 
(4, 50.4205), 
(4, 49.8256), 
(4, 49.9478), 
(4, 49.6219), 
(4, 46.7129), 
(4, 52.1411), 
(4, 50.8748), 
(5, 49.1578), 
(5, 50.2328), 
(5, 54.0862), 
(5, 51.0644), 
(5, 51.5309), 
(5, 48.5375), 
(5, 49.7478), 
(6, 48.2702), 
(6, 51.8346), 
(6, 50.6189), 
(6, 47.9971), 
(6, 49.6932), 
(6, 50.7387), 
(6, 47.8524), 
(7, 49.3663), 
(7, 49.0862), 
(7, 48.9394), 
(7, 47.5773), 
(7, 51.6820), 
(7, 50.9390), 
(7, 50.5057), 
(8, 49.8240), 
(8, 52.6363), 
(8, 50.0565), 
(8, 51.6997), 
(8, 47.1930), 
(8, 47.1411), 
(8, 51.0579), 
(9, 46.8016), 
(9, 49.5326), 
(9, 49.2139), 
(9, 52.0699), 
(9, 47.8713), 
(9, 52.8251), 
(9, 48.5065), 
(10, 49.6042), 
(10, 49.7772), 
(10, 47.9431), 
(10, 50.8975), 
(10, 49.1747), 
(10, 48.7357), 
(10, 48.3579), 
(11, 48.2800), 
(11, 48.6296), 
(11, 49.9935), 
(11, 48.9257), 
(11, 49.0928), 
(11, 51.1121), 
(11, 47.5623), 
(12, 47.6895), 
(12, 48.9562), 
(12, 51.8386), 
(12, 52.8005), 
(12, 50.4726), 
(12, 50.2589), 
(12, 53.8216), 
(13, 49.8145), 
(13, 50.9040), 
(13, 47.5791), 
(13, 49.4016), 
(13, 50.9916), 
(13, 47.6175), 
(13, 50.5437), 
(14, 53.0310), 
(14, 52.7705), 
(14, 48.5141), 
(14, 48.1268), 
(14, 50.1865), 
(14, 48.9621), 
(14, 47.7550), 
(15, 49.7535), 
(15, 52.8379), 
(15, 50.2787), 
(15, 50.6172), 
(15, 52.6317), 
(15, 50.7091), 
(15, 49.7502), 
(16, 50.5373), 
(16, 50.3804), 
(16, 50.4036), 
(16, 45.1273), 
(16, 51.8971), 
(16, 50.4648), 
(16, 50.3262), 
(17, 51.6779), 
(17, 53.5320), 
(17, 50.0889), 
(17, 52.5288), 
(17, 47.9699), 
(17, 50.5063), 
(17, 54.5354), 
(18, 48.4554), 
(18, 48.1973), 
(18, 49.8013), 
(18, 50.2697), 
(18, 50.9793), 
(18, 50.5161), 
(18, 51.2717), 
(19, 47.7204), 
(19, 50.1442), 
(19, 49.4636), 
(19, 47.0847), 
(19, 52.8940), 
(19, 50.5330), 
(19, 51.9745), 
(20, 48.8230), 
(20, 50.3212), 
(20, 49.2627), 
(20, 54.9971), 
(20, 49.3569), 
(20, 52.8028), 
(20, 48.9181), 
(21, 50.1945), 
(21, 44.4833), 
(21, 50.0107), 
(21, 51.4338), 
(21, 51.6983), 
(21, 53.2910), 
(21, 49.2356), 
(22, 52.9344), 
(22, 49.4353), 
(22, 48.2737), 
(22, 48.6704), 
(22, 47.5469), 
(22, 47.9728), 
(22, 48.6714), 
(23, 48.9329), 
(23, 49.5115), 
(23, 50.2384), 
(23, 49.5886), 
(23, 53.1541), 
(23, 50.8033), 
(23, 51.5840), 
(24, 47.8494), 
(24, 50.3497), 
(24, 52.0804), 
(24, 51.9442), 
(24, 49.0161), 
(24, 49.0522), 
(24, 46.8291), 
(25, 54.0810), 
(25, 49.5773), 
(25, 48.9657), 
(25, 51.2629), 
(25, 47.4584), 
(25, 50.7807), 
(25, 51.6742), 
(26, 49.3694), 
(26, 48.2803), 
(26, 47.6397), 
(26, 50.5831), 
(26, 45.2746), 
(26, 49.4577), 
(26, 48.8624), 
(27, 50.2737), 
(27, 48.1289), 
(27, 47.7931), 
(27, 48.6272), 
(27, 51.6873), 
(27, 49.3127), 
(27, 53.0916), 
(28, 48.4617), 
(28, 50.7600), 
(28, 52.9256), 
(28, 50.3355), 
(28, 52.4177), 
(28, 48.0196), 
(28, 49.8996)
]


def d2(n):
    """
    Args:
        n: int
            The number of samples. Must be  n >= 2. D2 is only recommended for n < 100

    Returns:
        float:
            The d2 value for n samples
    """
    if n < 2:
        raise Exception("Invalid n value")

    if n > 100:
        warnings.warn("Warning D2 is not recommended for n values beyond 100.")

    d2_constant = [
        None,
        None,
        1.128,
        1.693,
        2.059,
        2.326,
        2.534,
        2.704,
        2.847,
        2.970,
        3.078,
        3.173,
        3.258,
        3.336,
        3.407,
        3.472,
        3.532,
        3.588,
        3.640,
        3.689,
        3.735,
        3.778,
        3.819,
        3.858,
        3.895,
        3.931,
        3.964,
        3.997,
        4.027,
        4.057,
        4.086,
        4.113,
        4.139,
        4.165,
        4.189,
        4.213,
        4.236,
        4.259,
        4.280,
        4.301,
        4.322,
        4.341,
        4.361,
        4.379,
        4.398,
        4.415,
        4.433,
        4.450,
        4.466,
        4.482,
        4.498,
    ]
    # print(len(d2_constant))

    if n < len(d2_constant):
        return d2_constant[n]

    return 3.4873 + 0.0250141 * n - 0.00009823 * n ** 2


def d3(n):
    """
    Args:
        n: int
            The number of samples. D3 is only recommended for n < 100

    Returns:
        float: The d3 constant for n samples
    """

    d3_constant = [
        None,
        None,
        0.8525,
        0.8884,
        0.8798,
        0.8641,
        0.8480,
        0.8332,
        0.8198,
        0.8078,
        0.7971,
        0.7873,
        0.7785,
        0.7704,
        0.7630,
        0.7562,
        0.7499,
        0.7441,
        0.7386,
        0.7335,
        0.7287,
        0.7242,
        0.7199,
        0.7159,
        0.7121,
        0.7084,
    ]
    print(len(d3_constant))

    if n < 2:
        raise Exception("Invalid n value")
    if n > 100:
        warnings.warn("Warning D3 is not recommended for n values beyond 100.")

    if n < len(d3_constant):
        return d3_constant[n]


    return 0.80818 - 0.0051871 * n + 0.00_005098 * n ** 2 - 0.000_00019 * n ** 3


def d4(n):
    """

    Args:
        n: The number of samples.

    Returns:
        float: The d4 constant for n samples
    """
    d4_constant = [
        None,
        None,
        0.954,
        1.588,
        1.978,
        2.257,
        2.472,
        2.645,
        2.791,
        2.915,
        3.024,
        3.121,
        3.207,
        3.285,
        3.356,
        3.422,
        3.482,
        3.538,
        3.591,
        3.640,
        3.686,
        3.730,
        3.771,
        3.811,
        3.847,
        3.883
    ]

    if n < 2:
        raise Exception("Invalid n value")

    if n > 100:
        warnings.warn("Warning D4 is not recommended for n values beyond 100.")

    if n < len(d4_constant):
        return d4_constant[n]

    return 2.88606 + 0.051313 * n - 0.00049243 * n ** 2 + 0.000_0019 * n ** 3


def c4(n):
    """Factor `c4` for unbiased estimation of the standard deviation.

    For a finite sample, the sample standard deviation tends to
    underestimate the population standard deviation. See, e.g.,
    https://en.wikipedia.org/wiki/Unbiased_estimation_of_standard_deviation
    for details. Dividing the sample standard deviation by the correction
    factor `c4` gives an unbiased estimator of the population standard deviation.

    Parameters
    ----------
    n : numeric or array
        The number of samples.

    Returns
    -------
    numeric or array
        The correction factor, usually written `c4` or `b(n)`.


    """
    return np.sqrt(2 / (n - 1)) * spec.gamma(n / 2) / spec.gamma((n - 1) / 2)


def c5(n):
    """
        I Have absolutely no fucking idea what this is or what this means.
    Args:
        n:

    Returns:

    """

    return sqrt(1 - c4(n) ** 2)


# TODO.: Expand for subgroups and add docstring. 
def stdev_within(data,w=2):
    # TODO.. Add Docstring
    df = pd.DataFrame(data)

    print(len(df))


    n = len(data)
    r = np.abs(np.diff(data))
    rbar = r.sum() / (n - w + 1)
    sigma_xbar = rbar / d2(w)
    return sigma_xbar


# TODO. Expand for subgroups and add docstring.
def stdev_overall(data: np.array):
    df = pd.DataFrame(data)
    
    
    """
    Calculates the Overall Standard Deviation. 
    
    Args:
        data (np.array): _description_

    Returns:
        float: Overall Standard Deviation
    """
    
    
    return np.std(data, ddof=1)


# TODO. Expand for subgroups and add docstring.
def cp(data, usl, lsl, sigma_tol):
    return (usl - lsl) / (sigma_tol * stdev_within(data))


# TODO. Expand for subgroups and add docstring.
def cpk(data, usl, lsl, sigma_tol):
    data = np.array(data)
    cpl = (data.mean() - lsl) / ((sigma_tol / 2) * stdev_within(data))
    cpu = (usl - data.mean()) / ((sigma_tol / 2) * stdev_within(data))
    return min(cpl, cpu)


# TODO. Expand for subgroups and add docstring.
def pp(data, usl, lsl, sigma_tol):
    return (usl - lsl) / (sigma_tol * stdev_overall(data))


# TODO. Expand for subgroups and add docstring.
def ppk(data, usl, lsl, sigma_tol):
    data = np.array(data)
    ppl = (data.mean() - lsl) / ((sigma_tol / 2) * stdev_overall(data))
    ppu = (usl - data.mean()) / ((sigma_tol / 2) * stdev_overall(data))
    return min(ppl, ppu)


# TODO.. CPM:
def cpm():
    pass


# TODO.. Explain further
def phi(x):
    # 'Cumulative distribution function for the standard normal distribution'
    # https://stackoverflow.com/questions/809362/how-to-calculate-cumulative-normal-distribution
    return (1.0 + math.erf(x / sqrt(2.0))) / 2.0


# TODO.. Explain further
def ppm(data, lsl, usl):
    x_bar = np.mean(data)
    s = stdev_overall(data)
    z_lsl = (x_bar - lsl) / s
    z_usl = (usl - x_bar) / s
    ppm_lsl = 1_000_000 * (1 - norm.cdf(z_lsl))
    ppm_usl = 1_000_000 * (1 - norm.cdf(z_usl))
    ppm = ppm_lsl + ppm_usl
    return ppm
