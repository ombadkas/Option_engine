#include "greeks.h"
#include <cmath>

namespace options {

Greeks analytical_greeks(double S, double K, double r, double T,
                         double sigma, double q, OptionType type)
{
    BSResult bs = black_scholes(S, K, r, T, sigma, q, type);
    double d1   = bs.d1;
    double d2   = bs.d2;

    double sqrtT  = std::sqrt(T);
    double disc   = std::exp(-r * T);
    double disc_q = std::exp(-q * T);
    double nd1    = norm_pdf(d1);

    Greeks g{};

    // ── Delta ─────────────────────────────────────────────────────────────────
    if (type == OptionType::CALL)
        g.delta = disc_q * norm_cdf(d1);
    else
        g.delta = -disc_q * norm_cdf(-d1);

    // ── Gamma (same for call and put by put-call parity) ──────────────────────
    g.gamma = disc_q * nd1 / (S * sigma * sqrtT);

    // ── Vega (per 1 vol point = 0.01) ─────────────────────────────────────────
    g.vega = S * disc_q * nd1 * sqrtT / 100.0;

    // ── Theta (per calendar day) ──────────────────────────────────────────────
    double term1 = -S * disc_q * nd1 * sigma / (2.0 * sqrtT);
    if (type == OptionType::CALL) {
        double term2 = -r * K * disc * norm_cdf(d2);
        double term3 =  q * S * disc_q * norm_cdf(d1);
        g.theta = (term1 + term2 + term3) / 365.0;
    } else {
        double term2 =  r * K * disc * norm_cdf(-d2);
        double term3 = -q * S * disc_q * norm_cdf(-d1);
        g.theta = (term1 + term2 + term3) / 365.0;
    }

    // ── Rho (per 1 basis point = 0.0001) ─────────────────────────────────────
    if (type == OptionType::CALL)
        g.rho =  K * T * disc * norm_cdf(d2)  / 10000.0;
    else
        g.rho = -K * T * disc * norm_cdf(-d2) / 10000.0;

    // ── Vanna: d(delta)/d(sigma) = -disc_q * nd1 * d2/sigma ─────────────────
    g.vanna = -disc_q * nd1 * d2 / sigma;

    // ── Volga / Vomma: d(vega)/d(sigma) ──────────────────────────────────────
    // vega * d1 * d2 / sigma  (the /100 matches the vega scaling above)
    g.volga = g.vega * d1 * d2 / sigma;

    return g;
}

Greeks numerical_greeks(double S, double K, double r, double T,
                        double sigma, double q, OptionType type,
                        const FDParams& p)
{
    auto price = [&](double s, double sig, double rate, double t) {
        return black_scholes(s, K, rate, t, sig, q, type).price;
    };

    double dS  = S * p.dS_rel;
    double dT  = p.dT_days / 365.0;

    Greeks g{};

    // Delta — central difference in S
    g.delta = (price(S + dS, sigma, r, T) - price(S - dS, sigma, r, T))
              / (2.0 * dS);

    // Gamma — second difference in S
    double V0  = price(S, sigma, r, T);
    g.gamma    = (price(S + dS, sigma, r, T) - 2.0 * V0 + price(S - dS, sigma, r, T))
                 / (dS * dS);

    // Vega — central difference in sigma (scaled to 1 vol point)
    g.vega  = (price(S, sigma + p.dsigma, r, T) - price(S, sigma - p.dsigma, r, T))
              / (2.0 * p.dsigma * 100.0);

    // Theta — backward difference in T (1 calendar day)
    g.theta = (price(S, sigma, r, T - dT) - V0) / p.dT_days;

    // Rho — central difference in r (scaled to 1 basis point)
    g.rho   = (price(S, sigma, r + p.dr, T) - price(S, sigma, r - p.dr, T))
              / (2.0 * p.dr * 10000.0);

    // Vanna — mixed S / sigma
    double Vup_sig = price(S + dS, sigma + p.dsigma, r, T);
    double Vdn_sig = price(S - dS, sigma + p.dsigma, r, T);
    double Vup_msig= price(S + dS, sigma - p.dsigma, r, T);
    double Vdn_msig= price(S - dS, sigma - p.dsigma, r, T);
    g.vanna = (Vup_sig - Vdn_sig - Vup_msig + Vdn_msig)
              / (4.0 * dS * p.dsigma);

    // Volga — second difference in sigma (scaled)
    g.volga = (price(S, sigma + p.dsigma, r, T) - 2.0 * V0
               + price(S, sigma - p.dsigma, r, T))
              / (p.dsigma * p.dsigma * 100.0);

    return g;
}

} // namespace options
