#pragma once
#include "pricer.h"

namespace options {

// ── Full Greeks bundle ────────────────────────────────────────────────────────
struct Greeks {
    double delta;  // dV/dS
    double gamma;  // d²V/dS²
    double vega;   // dV/d(sigma), per 1 vol point (i.e. /100 of dV/dsigma)
    double theta;  // dV/dT (per calendar day, negative for long options)
    double rho;    // dV/dr (per 1 basis point, i.e. /10000 of dV/dr)
    double vanna;  // d²V/(dS d(sigma))
    double volga;  // d²V/d(sigma)²  (also called vomma)
};

// Closed-form analytical Greeks (exact for European B-S)
Greeks analytical_greeks(double S, double K, double r, double T,
                         double sigma, double q = 0.0,
                         OptionType type = OptionType::CALL);

// ── Numerical Greeks via central finite differences ───────────────────────────
// Useful for cross-validating MC/tree prices against closed-form.
struct FDParams {
    double dS_rel  = 0.001;      // relative spot bump   (0.1 %)
    double dsigma  = 0.0001;     // absolute vol bump    (1 bp vol)
    double dr      = 0.0001;     // absolute rate bump   (1 bp)
    double dT_days = 1.0;        // theta: 1 calendar day
};

Greeks numerical_greeks(double S, double K, double r, double T,
                        double sigma, double q = 0.0,
                        OptionType type = OptionType::CALL,
                        const FDParams& p = {});

} // namespace options
