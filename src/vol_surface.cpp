#include "vol_surface.h"
#include "greeks.h"
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <limits>

namespace options {

// ── Newton-Raphson + bisection fallback ──────────────────────────────────────
IVResult implied_vol(double market_price, double S, double K,
                     double r, double T, double q,
                     OptionType type, double initial_guess,
                     double tol, int max_iter)
{
    if (market_price <= 0.0)
        return {0.0, 0, false, market_price};

    // Intrinsic value check (price below intrinsic is arbitrage)
    double intrinsic = (type == OptionType::CALL)
        ? std::max(S * std::exp(-q * T) - K * std::exp(-r * T), 0.0)
        : std::max(K * std::exp(-r * T) - S * std::exp(-q * T), 0.0);

    if (market_price < intrinsic - tol)
        return {std::numeric_limits<double>::quiet_NaN(), 0, false, market_price - intrinsic};

    double sigma = initial_guess;
    if (sigma <= 0.0) sigma = 0.20;

    // Bisection bracket
    double lo = 1e-6, hi = 10.0;

    IVResult res{sigma, 0, false, 0.0};

    for (int iter = 0; iter < max_iter; ++iter) {
        BSResult bs  = black_scholes(S, K, r, T, sigma, q, type);
        double   diff = bs.price - market_price;
        double   vega = S * std::exp(-q * T) * norm_pdf(bs.d1) * std::sqrt(T);

        res.iterations = iter + 1;
        res.residual   = std::abs(diff);

        if (res.residual < tol) {
            res.implied_vol = sigma;
            res.converged   = true;
            return res;
        }

        // Newton step; fall back to bisection when vega is too small
        if (vega > 1e-10) {
            double step = diff / vega;
            sigma -= step;
        }

        // Keep within bracket
        if (sigma <= lo || sigma >= hi) {
            // Bisection
            double f_lo = black_scholes(S, K, r, T, lo, q, type).price - market_price;
            double f_hi = black_scholes(S, K, r, T, hi, q, type).price - market_price;
            if (f_lo * f_hi > 0.0) break; // no root in bracket
            sigma = 0.5 * (lo + hi);
            double f_mid = black_scholes(S, K, r, T, sigma, q, type).price - market_price;
            if (f_lo * f_mid <= 0.0) hi = sigma; else lo = sigma;
        } else {
            if (diff < 0) lo = sigma; else hi = sigma;
        }
    }

    res.implied_vol = sigma;
    res.converged   = res.residual < tol * 10.0;
    return res;
}

// ── VolSurface interpolation ──────────────────────────────────────────────────
// Bilinear in (log-strike, sqrt(T)) space for smoother behaviour
double VolSurface::interpolate(double K, double T) const {
    if (expiries.empty() || strikes.empty()) return 0.0;

    // Clamp to surface boundaries
    double lK  = std::log(K / S);
    double sqT = std::sqrt(T);

    // Build log-strike and sqrt-expiry grids
    std::vector<double> lKs(strikes.size());
    for (size_t j = 0; j < strikes.size(); ++j)
        lKs[j] = std::log(strikes[j] / S);

    std::vector<double> sqTs(expiries.size());
    for (size_t i = 0; i < expiries.size(); ++i)
        sqTs[i] = std::sqrt(expiries[i]);

    // Clamp
    lK  = std::clamp(lK,  lKs.front(), lKs.back());
    sqT = std::clamp(sqT, sqTs.front(), sqTs.back());

    // Find surrounding indices
    auto it_t  = std::lower_bound(sqTs.begin(), sqTs.end(), sqT);
    auto it_k  = std::lower_bound(lKs.begin(),  lKs.end(),  lK);

    size_t ti = std::min((size_t)std::max((ptrdiff_t)(it_t - sqTs.begin()) - 1, (ptrdiff_t)0),
                         expiries.size() - 2);
    size_t ki = std::min((size_t)std::max((ptrdiff_t)(it_k - lKs.begin()) - 1, (ptrdiff_t)0),
                         strikes.size() - 2);

    double t0 = sqTs[ti], t1 = sqTs[ti+1];
    double k0 = lKs[ki],  k1 = lKs[ki+1];

    double wt = (t1 > t0) ? (sqT - t0) / (t1 - t0) : 0.0;
    double wk = (k1 > k0) ? (lK  - k0) / (k1 - k0) : 0.0;

    double v00 = vols[ti  ][ki  ];
    double v10 = vols[ti+1][ki  ];
    double v01 = vols[ti  ][ki+1];
    double v11 = vols[ti+1][ki+1];

    return (1-wt)*(1-wk)*v00 + wt*(1-wk)*v10
         + (1-wt)*wk*v01     + wt*wk*v11;
}

double VolSurface::price(double K, double T, OptionType type) const {
    double iv = interpolate(K, T);
    return black_scholes(S, K, r, T, iv, q, type).price;
}

// ── Build surface from market prices ─────────────────────────────────────────
VolSurface build_vol_surface(double S, double r, double q,
                             const std::vector<double>& expiries,
                             const std::vector<double>& strikes,
                             const std::vector<std::vector<double>>& market_prices,
                             OptionType type)
{
    VolSurface surf;
    surf.S = S; surf.r = r; surf.q = q;
    surf.expiries = expiries;
    surf.strikes  = strikes;
    surf.vols.resize(expiries.size(), std::vector<double>(strikes.size(), 0.0));

    for (size_t i = 0; i < expiries.size(); ++i) {
        for (size_t j = 0; j < strikes.size(); ++j) {
            IVResult iv = implied_vol(market_prices[i][j], S, strikes[j],
                                      r, expiries[i], q, type);
            surf.vols[i][j] = iv.converged ? iv.implied_vol : 0.20;
        }
    }
    return surf;
}

} // namespace options
