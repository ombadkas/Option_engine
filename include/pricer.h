#pragma once
#include <stdexcept>
#include <cmath>

namespace options {

// ── Enumerations ──────────────────────────────────────────────────────────────
enum class OptionType   { CALL, PUT };
enum class ExerciseStyle{ EUROPEAN, AMERICAN };

// ── Normal distribution helpers ───────────────────────────────────────────────
inline double norm_cdf(double x) noexcept {
    return 0.5 * std::erfc(-x * 0.7071067811865475); // 1/sqrt(2)
}
inline double norm_pdf(double x) noexcept {
    return 0.3989422804014327 * std::exp(-0.5 * x * x); // 1/sqrt(2*pi)
}

// ── Result type ───────────────────────────────────────────────────────────────
struct BSResult {
    double price;
    double d1;
    double d2;
};

// ── Black-Scholes European pricing ────────────────────────────────────────────
// S     : current spot price
// K     : strike price
// r     : continuously compounded risk-free rate
// T     : time to expiry in years
// sigma : annualised volatility
// q     : continuous dividend yield (default 0)
BSResult black_scholes(double S, double K, double r, double T,
                       double sigma, double q = 0.0,
                       OptionType type = OptionType::CALL);

double bs_call(double S, double K, double r, double T, double sigma, double q = 0.0) noexcept;
double bs_put (double S, double K, double r, double T, double sigma, double q = 0.0) noexcept;

// Put-call parity check: |C - P - (S*e^{-qT} - K*e^{-rT})| < tol
bool validate_put_call_parity(double S, double K, double r, double T,
                              double sigma, double q = 0.0, double tol = 1e-8);

} // namespace options
