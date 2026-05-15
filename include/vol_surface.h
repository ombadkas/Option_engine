#pragma once
#include "pricer.h"
#include <vector>

namespace options {

// ── Implied volatility result ─────────────────────────────────────────────────
struct IVResult {
    double implied_vol;
    int    iterations;
    bool   converged;
    double residual;   // |BS(iv) - market_price|
};

// Newton-Raphson implied-vol solver.
// Brackets with bisection fallback when vega is near zero.
// initial_guess: starting sigma (default 0.20 = 20 %)
IVResult implied_vol(double market_price, double S, double K,
                     double r, double T, double q = 0.0,
                     OptionType type  = OptionType::CALL,
                     double initial_guess = 0.20,
                     double tol      = 1e-8,
                     int    max_iter = 100);

// ── Volatility surface ────────────────────────────────────────────────────────
// Stores a grid of implied vols across strikes (absolute) and expiries (years).
// Bilinear interpolation in log-strike / sqrt(T) space.
struct VolSurface {
    double              S;         // spot at construction
    double              r;
    double              q;
    std::vector<double> expiries;  // sorted, in years
    std::vector<double> strikes;   // sorted, absolute
    // vols[i][j] = IV for expiry i, strike j
    std::vector<std::vector<double>> vols;

    // Interpolated IV for arbitrary (K, T)
    double interpolate(double K, double T) const;

    // Price via interpolated IV
    double price(double K, double T, OptionType type = OptionType::CALL) const;
};

// Build a vol surface by solving for implied vols from a matrix of market prices.
// market_prices[i][j] corresponds to expiries[i], strikes[j].
VolSurface build_vol_surface(double S, double r, double q,
                             const std::vector<double>& expiries,
                             const std::vector<double>& strikes,
                             const std::vector<std::vector<double>>& market_prices,
                             OptionType type = OptionType::CALL);

} // namespace options
