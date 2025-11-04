# Black-Scholes Option Pricer

A Python implementation of the Black-Scholes model for pricing European call and put options, with interactive visualizations.

## Requirement

```bash
pip install numpy matplotlib
```

## How to run

Run the script :

```bash
python3 coding-a-option-pricing-model.py
```

### Input Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| **Spot price (S)** | Current price of underlying asset | 100 |
| **Time to maturity (t)** | Years until expiration | 0.25 (3 months) |
| **Volatility (σ)** | Annualized volatility as decimal | 0.20 (20%) |
| **Risk-free rate (r)** | Annualized rate as decimal | 0.04 (4%) |
| **Strike (K)** | Option strike price | 105 |
| **Type** | C for call, P for put | C |

### Example Session

```
The price of the underlying asset is: 100
The annualized time to maturity is: 0.25
The annualized implied volatility is: 0.20
The discount rate is: 0.04
The strike price of the option is: 105
The option type (C for Calls, P for Puts): C
Option price is: 2.39
```

## Outputs

1. **Console**: Fair value of the option
2. **2D Plot**: Option price vs. spot price
3. **3D Surface**: Option price vs. spot and time to maturity

## Assumptions

- Frictionless markets
- Continuous trading
- Constant risk-free rate
- Constant volatility
- European exercise (expiration only)
- No dividends

## Formula

The Black-Scholes formula for a call option:

```
C = S·N(d₁) - K·e^(-rT)·N(d₂)
```

Where:
- d₁ = [ln(S/K) + (r + σ²/2)T] / (σ√T)
- d₂ = d₁ - σ√T
- N(x) = Standard normal cumulative distribution function

For puts, use put-call parity or the adjusted formula in the code.
