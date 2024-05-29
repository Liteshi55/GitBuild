# htrfnwe_optimized.pyx

cimport numpy as np
import numpy as np
cimport cython
from cython.parallel import prange
from libc.math cimport exp, fabs, isnan
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def ewma(np.ndarray[np.float32_t, ndim=1] series, double alpha):
    cdef int n = series.shape[0]
    cdef np.ndarray[np.float32_t, ndim=1] ewma_series = np.empty(n, dtype=np.float32)
    cdef int i

    ewma_series[0] = series[0]
    for i in range(1, n):
        ewma_series[i] = alpha * series[i] + (1 - alpha) * ewma_series[i - 1]

    return ewma_series

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def ema2(np.ndarray[np.float32_t, ndim=1] source, double length):
    cdef double alpha = 2.0 / (max(1.0, length) + 1.0)
    return ewma(source, alpha)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def tema3(np.ndarray[np.float32_t, ndim=1] source, double length):
    cdef np.ndarray[np.float32_t, ndim=1] ema1, ema2_, ema3
    ema1 = ema2(source, length)
    ema2_ = ema2(ema1, length)
    ema3 = ema2(ema2_, length)
    return 3 * (ema1 - ema2_) + ema3

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def halftrend(np.ndarray[np.float32_t, ndim=1] high, 
              np.ndarray[np.float32_t, ndim=1] low, 
              np.ndarray[np.float32_t, ndim=1] close, 
              np.ndarray[np.float32_t, ndim=1] tr, 
              int amplitude=2, 
              double channel_deviation=2.0):
    cdef int n = high.shape[0]
    cdef np.ndarray[np.float32_t, ndim=1] atr2 = tr * 0.5
    cdef np.ndarray[np.float32_t, ndim=1] dev = channel_deviation * atr2
    cdef np.ndarray[np.float32_t, ndim=1] high_price = np.empty_like(high)
    cdef np.ndarray[np.float32_t, ndim=1] low_price = np.empty_like(low)
    cdef int i

    for i in range(amplitude):
        high_price[i] = np.max(high[:i + 1])
        low_price[i] = np.min(low[:i + 1])
    for i in range(amplitude, n):
        high_price[i] = np.max(high[i - amplitude + 1:i + 1])
        low_price[i] = np.min(low[i - amplitude + 1:i + 1])

    cdef np.ndarray[np.float32_t, ndim=1] highma = tema3(high, amplitude * 2)
    cdef np.ndarray[np.float32_t, ndim=1] lowma = tema3(low, amplitude * 2)

    cdef np.ndarray[np.int32_t, ndim=1] trend = np.zeros(n, dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1] next_trend = np.zeros(n, dtype=np.int32)
    cdef np.ndarray[np.float32_t, ndim=1] max_low_price = np.zeros(n, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] min_high_price = np.zeros(n, dtype=np.float32)

    if n > 0:
        max_low_price[0] = low[0]
        min_high_price[0] = high[0]

    for i in range(1, n):
        if next_trend[i - 1] == 1:
            max_low_price[i] = max(low_price[i - 1], max_low_price[i - 1])
            if highma[i] < max_low_price[i] and close[i] < low[i - 1]:
                trend[i] = 1
                next_trend[i] = 0
                min_high_price[i] = high_price[i]
            else:
                trend[i] = trend[i - 1]
                next_trend[i] = next_trend[i - 1]
                min_high_price[i] = min_high_price[i - 1]
        else:
            min_high_price[i] = min(high_price[i - 1], min_high_price[i - 1])
            if lowma[i] > min_high_price[i] and close[i] > high[i - 1]:
                trend[i] = 0
                next_trend[i] = 1
                max_low_price[i] = low_price[i]
            else:
                trend[i] = trend[i - 1]
                next_trend[i] = next_trend[i - 1]
                max_low_price[i] = max_low_price[i - 1]

    cdef np.ndarray[np.float32_t, ndim=1] up = np.zeros(n, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] down = np.zeros(n, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] atr_high = np.zeros(n, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] atr_low = np.zeros(n, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] arrow_up = np.zeros(n, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] arrow_down = np.zeros(n, dtype=np.float32)

    if n > 0:
        up[0] = max_low_price[0]
        down[0] = min_high_price[0]
        atr_high[0] = tr[0]
        atr_low[0] = tr[0]

    for i in range(1, n):
        if trend[i] == 0:
            if trend[i - 1] != 0:
                up[i] = down[i - 1]
                arrow_up[i] = up[i] - atr2[i]
            else:
                up[i] = max(max_low_price[i - 1], up[i - 1])

            atr_high[i] = up[i] + dev[i]
            atr_low[i] = up[i] - dev[i]

        else:
            if trend[i - 1] != 1:
                down[i] = up[i - 1]
                arrow_down[i] = down[i] + atr2[i]
            else:
                down[i] = min(min_high_price[i - 1], down[i - 1])

            atr_high[i] = down[i] + dev[i]
            atr_low[i] = down[i] - dev[i]

    halftrend = np.where(trend == 0, up, down)
    buy = np.where((trend == 0) & (np.roll(trend, 1) == 1), 1, 0)
    sell = np.where((trend == 1) & (np.roll(trend, 1) == 0), 1, 0)

    htdf = {
        'halftrend': halftrend,
        'atrHigh': atr_high,
        'atrLow': atr_low,
        'arrowUp': arrow_up,
        'arrowDown': arrow_down,
        'buy': buy,
        'sell': sell
    }

    return htdf

cdef double best_alpha

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def ema(np.ndarray[double, ndim=1] x, int period):
    cdef int n = x.shape[0]
    cdef np.ndarray[double, ndim=1] weights = np.ones(period, dtype=np.float64) / period
    cdef np.ndarray[double, ndim=1] result = np.empty(n - period + 1, dtype=np.float64)
    cdef int i
    for i in range(n - period + 1):
        result[i] = np.dot(x[i:i + period], weights)
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def range_size(np.ndarray[double, ndim=1] arr, int range_period, double range_multiplier):
    cdef int n = arr.shape[0]
    cdef np.ndarray[double, ndim=1] arr_diff_abs = np.empty(n, dtype=np.float64)
    cdef np.ndarray[double, ndim=1] first_ema, second_ema, result
    cdef int i, pad_length

    arr_diff_abs[0] = 0
    for i in prange(1, n, nogil=True):
        arr_diff_abs[i] = fabs(arr[i] - arr[i - 1])

    first_ema = ema(arr_diff_abs, range_period)
    second_ema = ema(first_ema, (range_period * 2) - 1)

    pad_length = n - second_ema.shape[0]
    result = np.empty(n, dtype=np.float64)
    result[:pad_length] = np.nan
    result[pad_length:] = second_ema * range_multiplier

    return result

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double nz(double x):
    return 0.0 if isnan(x) else x

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def range_filter(np.ndarray[double, ndim=1] x, np.ndarray[double, ndim=1] r):
    cdef int n = x.shape[0]
    cdef np.ndarray[double, ndim=1] range_filt = np.zeros(n, dtype=np.float64)
    cdef np.ndarray[double, ndim=1] hi_band = np.zeros(n, dtype=np.float64)
    cdef np.ndarray[double, ndim=1] lo_band = np.zeros(n, dtype=np.float64)
    cdef int i

    range_filt[0] = x[0]

    for i in prange(1, n, nogil=True):
        with gil:
            range_filt[i] = nz(range_filt[i - 1])

        if not isnan(r[i]):
            if x[i] > range_filt[i - 1]:
                range_filt[i] = max(range_filt[i - 1], x[i] - r[i])
            else:
                range_filt[i] = min(range_filt[i - 1], x[i] + r[i])

        hi_band[i] = range_filt[i] + r[i]
        lo_band[i] = range_filt[i] - r[i]

    for i in range(1, n):
        if isnan(range_filt[i]):
            range_filt[i] = range_filt[i - 1]
            hi_band[i] = hi_band[i - 1]
            lo_band[i] = lo_band[i - 1]

    return np.vstack((hi_band, lo_band, range_filt)).T

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def vumanchu_swing(np.ndarray[double, ndim=1] arr, int swing_period, double swing_multiplier):
    cdef np.ndarray[double, ndim=1] smrng = range_size(arr, swing_period, swing_multiplier)
    return range_filter(arr, smrng)

cpdef np.ndarray[np.float32_t, ndim=1] run_nwe(np.ndarray[np.float32_t, ndim=2] indata):
    cdef np.ndarray[np.float32_t, ndim=2] X = indata
    cdef np.ndarray[np.float32_t, ndim=1] y = indata[:, 0]
    cdef np.ndarray[np.float32_t, ndim=1] y_pred

    param_grid = {'alpha': [0.5, 1.5, 2.5, 3.0, 3.5, 5.0]}

    kr = GridSearchCV(KernelRidge(kernel='rbf'), param_grid)
    global best_alpha
    if not hasattr(run_nwe, 'best_alpha'):
        kr.fit(X, y)
        best_alpha = kr.best_params_['alpha']

    cdef np.float32_t alpha = best_alpha
    cdef int n_samples = X.shape[0]
    cdef np.ndarray[np.float32_t, ndim=1] weights = np.empty(n_samples, dtype=np.float32)
    y_pred = np.empty(n_samples, dtype=np.float32)

    cdef double sum_weights
    cdef double sum_weighted_y
    cdef double diff
    cdef int i, j, k

    for i in range(n_samples):
        sum_weights = 0.0
        sum_weighted_y = 0.0
        for j in range(n_samples):
            diff = 0.0
            for k in range(X.shape[1]):
                diff += (X[j, k] - X[i, k]) ** 2
            weights[j] = exp(-alpha * diff)
            sum_weights += weights[j]
            sum_weighted_y += weights[j] * y[j]
        y_pred[i] = sum_weighted_y / sum_weights

    cdef int window_size = 5
    cdef np.ndarray[np.float32_t, ndim=1] rolling_mean = np.empty(n_samples, dtype=np.float32)

    cdef double window_sum = 0.0

    for i in range(n_samples):
        window_sum += y_pred[i]
        if i >= window_size:
            window_sum -= y_pred[i - window_size]
        rolling_mean[i] = window_sum / min(i + 1, window_size)

    return rolling_mean
