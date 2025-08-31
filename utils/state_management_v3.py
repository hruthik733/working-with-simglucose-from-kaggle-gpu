import collections
import numpy as np
from scipy.stats import gamma
import warnings
from datetime import datetime

def get_pkpd_discount_factors(t_peak, t_end, n_steps):
    shape_k = 2
    scale_theta = t_peak / (shape_k - 1)
    time_points = np.linspace(0, t_end, n_steps)
    pdf_values = gamma.pdf(time_points, a=shape_k, scale=scale_theta)
    f_k = pdf_values / np.max(pdf_values)
    cdf_values = gamma.cdf(time_points, a=shape_k, scale=scale_theta)
    F_k = cdf_values
    return f_k, F_k

class StateRewardManager:
    def __init__(self, state_dim=12):
        self.glucose_history = collections.deque(maxlen=6)
        self.insulin_history = collections.deque(maxlen=160)
        self.reset()
        _, self.F_k = get_pkpd_discount_factors(t_peak=55, t_end=480, n_steps=160)

        self.state_mean = np.array([140.0, 0.0, 2.0, 10.0, 0.0, 0.0] + [140.0] * 6)
        self.state_std = np.array([40.0, 1.5, 2.0, 20.0, 1.0, 1.0] + [40.0] * 6)

    def get_normalized_state(self, state):
        return (state - self.state_mean) / (self.state_std + 1e-8)

    def calculate_iob(self):
        # ! FIX: Corrected a closing parenthesis typo
        return np.sum(np.array(list(self.insulin_history)[::-1]) * (1 - self.F_k))

    def get_full_state(self, observation, current_sim_time, upcoming_carbs=0):
        glucose_value = observation
        self.glucose_history.append(glucose_value)
        
        padded_history = list(self.glucose_history)
        while len(padded_history) < 6:
            padded_history.insert(0, padded_history[0])

        rate = (padded_history[-1] - padded_history[-2]) / 5.0
        iob = self.calculate_iob()

        time_in_seconds = current_sim_time.hour * 3600 + current_sim_time.minute * 60
        sin_time = np.sin(2 * np.pi * time_in_seconds / 86400)
        cos_time = np.cos(2 * np.pi * time_in_seconds / 86400)

        # ! FIX: Create a single flat list first, then convert to a NumPy array.
        # This resolves the inhomogeneous shape error.
        all_features = [glucose_value, rate, iob, upcoming_carbs, sin_time, cos_time] + padded_history
        return np.array(all_features)

    def get_reward(self, state):
        glucose = state[0]
        reward = 0.0
        
        if 70 <= glucose <= 180:
            reward += 1.0
        
        if glucose < 70:
            reward -= 0.1 * ((70 - glucose)**2)
            
        if glucose > 180:
            reward -= 0.01 * ((glucose - 180)**2)
            
        return reward

    def reset(self):
        self.glucose_history.clear()
        for _ in range(6): self.glucose_history.append(140)
        self.insulin_history.clear()
        for _ in range(160): self.insulin_history.append(0)