import numpy as np
import math  
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def norm_cdf(x):
    """Compute the CDF of the standard normal distribution"""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def black_scholes(s, t, sigma, r, k, option_type):
    """Calculates Black-Scholes price for Call (C) and Put (P) options"""
    d1 = (np.log(s / k) + (r + (sigma**2) / 2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)

    if option_type == "C":  # Call option
        price = s * norm_cdf(d1) - k * np.exp(-r * t) * norm_cdf(d2)
    elif option_type == "P":  # Put option
        price = k * np.exp(-r * t) * norm_cdf(-d2) - s * norm_cdf(-d1)
    else:
        raise ValueError("Invalid option type. Use 'C' for Call and 'P' for Put.")

    return price

# Get user inputs
s = float(input("The price of the underlying asset is: "))  
t = float(input("The annualized time to maturity is: "))     
sigma = float(input("The annualized implied volatility is: "))  
r = float(input("The discount rate is: "))  
k = float(input("The strike price of the option is: "))   
option_type = input("The option type (C for Calls, P for Puts): ").upper()

# Compute and print the option price
option_price = black_scholes(s, t, sigma, r, k, option_type)
print("Option price is:", round(option_price, 2))

# Graph 1: Option Price vs. Stock Price
s_range = np.linspace(50, 150, 100)  # Stock prices from 50 to 150
prices = [black_scholes(s, t, sigma, r, k, option_type) for s in s_range]

plt.figure(figsize=(10, 5))
plt.plot(s_range, prices, label=f"{'Call' if option_type == 'C' else 'Put'} Option Price", color='b')
plt.axvline(s, color='r', linestyle='--', label="Current Stock Price")
plt.xlabel("Stock Price (S)")
plt.ylabel("Option Price")
plt.title("Black-Scholes Option Price vs. Stock Price")
plt.legend()
plt.grid()
plt.show(block=False)  # <---- Allows multiple plots to show at once

# Graph 2: 3D Surface Plot (Stock Price vs. Time to Maturity)
s_range = np.linspace(50, 150, 30)  # Stock price range
t_range = np.linspace(0.01, 2, 30)  # Time to maturity range

S, T = np.meshgrid(s_range, t_range)
Z = np.array([[black_scholes(s, t, sigma, r, k, option_type) for s in s_range] for t in t_range])

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(S, T, Z, cmap='viridis')

ax.set_xlabel("Stock Price (S)")
ax.set_ylabel("Time to Maturity (T)")
ax.set_zlabel("Option Price")
ax.set_title("Black-Scholes Option Price Surface")

plt.show(block=False)  # <---- Keeps the first plot open

# Keep the plots open
input("Press Enter to exit...")