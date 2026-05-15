#include "pricer.h"
#include <stdexcept>
#include <cmath>
#include <algorithm>

namespace options {

BSResult black_scholes(double S, double K, double r, double T,
                       double sigma, double q, OptionType type)
{
    if (S <= 0.0) throw std::invalid_argument("spot S must be positive");
    if (K <= 0.0) throw std::invalid_argument("strike K must be positive");
    if (sigma < 0.0) throw std::invalid_argument("volatility sigma must be >= 0");
    if (T < 0.0)  throw std::invalid_argument("expiry T must be >= 0");

    // ── Boundary: expired option ──────────────────────────────────────────────
    if (T == 0.0) {
        double iv = (type == OptionType::CALL) ? std::max(S - K, 0.0)
                                               : std::max(K - S, 0.0);
        return {iv, 0.0, 0.0};
    }

    // ── Boundary: zero vol → deterministic payoff ─────────────────────────────
    if (sigma == 0.0) {
        double fwd  = S * std::exp((r - q) * T);
        double disc = std::exp(-r * T);
        double pv   = (type == OptionType::CALL) ? disc * std::max(fwd - K, 0.0)
                                                 : disc * std::max(K - fwd, 0.0);
        return {pv, 0.0, 0.0};
    }

    double sqrtT  = std::sqrt(T);
    double d1     = (std::log(S / K) + (r - q + 0.5 * sigma * sigma) * T)
                    / (sigma * sqrtT);
    double d2     = d1 - sigma * sqrtT;
    double disc   = std::exp(-r * T);
    double disc_q = std::exp(-q * T);

    double price{};
    if (type == OptionType::CALL)
        price = S * disc_q * norm_cdf(d1) - K * disc * norm_cdf(d2);
    else
        price = K * disc * norm_cdf(-d2) - S * disc_q * norm_cdf(-d1);

    return {price, d1, d2};
}

double bs_call(double S, double K, double r, double T, double sigma, double q) noexcept {
    try { return black_scholes(S, K, r, T, sigma, q, OptionType::CALL).price; }
    catch (...) { return 0.0; }
}

double bs_put(double S, double K, double r, double T, double sigma, double q) noexcept {
    try { return black_scholes(S, K, r, T, sigma, q, OptionType::PUT).price; }
    catch (...) { return 0.0; }
}

bool validate_put_call_parity(double S, double K, double r, double T,
                              double sigma, double q, double tol)
{
    double C   = bs_call(S, K, r, T, sigma, q);
    double P   = bs_put (S, K, r, T, sigma, q);
    double rhs = S * std::exp(-q * T) - K * std::exp(-r * T);
    return std::abs((C - P) - rhs) < tol;
}

} // namespace options
