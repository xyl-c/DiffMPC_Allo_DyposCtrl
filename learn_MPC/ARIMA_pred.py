import numpy as np
from statsmodels.tsa.arima.model import ARIMA


def arima_pre3d(history: np.ndarray, dt: float, prediction_time: float, order: tuple[int, int, int] = (20, 0, 2),) -> np.ndarray:
    u_history = history[0, :]
    v_history = history[1, :]
    r_history = history[2, :]
    u_pre = arima_predict(u_history, dt, prediction_time, order)
    v_pre = arima_predict(v_history, dt, prediction_time, order)
    r_pre = arima_predict(r_history, dt, prediction_time, order)
    pre_series = np.stack([u_pre, v_pre, r_pre])
    return pre_series

def arima_predict(history: np.ndarray, dt: float, prediction_time: float, order: tuple[int, int, int] = (20, 0, 2),) -> np.ndarray:
    history = np.asarray(history,dtype=np.float64).reshape(-1)
    if history.ndim != 1:
        raise ValueError("history must be a one-dimensional sequence.")
    if len(history) < 30:
        raise ValueError("The historical data is too short; it is recommended to use at least 30 data points.")
    if not np.all(np.isfinite(history)):
        raise ValueError("There are NaN or Inf values in history.")
    prediction_steps_float = (prediction_time / dt)
    prediction_steps = int(round(prediction_steps_float))
    if prediction_steps <= 0:
        raise ValueError("The number of prediction steps must be greater than 0.")
    if not np.isclose(prediction_steps * dt, prediction_time):
        raise ValueError("prediction_time must be an integer multiple of dt.")
    history_mean = np.mean(history)
    history_centered = (history - history_mean)
    model = ARIMA(history_centered, order=order, trend="n", enforce_stationarity=False, enforce_invertibility=False)
    fitted_model = model.fit(method="statespace", method_kwargs={"maxiter": 100, "disp": False},)
    prediction_centered = (fitted_model.forecast(steps=prediction_steps))
    prediction = np.asarray(prediction_centered, dtype=np.float64).reshape(-1)
    prediction += history_mean
    return prediction


