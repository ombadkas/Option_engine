#include "binomial_tree.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace options {

TreeResult crr_tree(double S, double K, double r, double T,
                    double sigma, double q,
                    OptionType type, ExerciseStyle style, int steps)
{
    if (steps < 2) throw std::invalid_argument("steps must be >= 2");
    if (S <= 0 || K <= 0 || sigma < 0 || T <= 0)
        throw std::invalid_argument("Invalid input parameters");

    double dt   = T / steps;
    double u    = std::exp(sigma * std::sqrt(dt));
    double d    = 1.0 / u;                        // CRR symmetry: d = 1/u
    double disc = std::exp(-r * dt);
    double pu   = (std::exp((r - q) * dt) - d) / (u - d);  // risk-neutral up prob
    double pd   = 1.0 - pu;

    if (pu < 0.0 || pu > 1.0)
        throw std::runtime_error("Risk-neutral probability outside [0,1]: increase steps or reduce sigma*sqrt(dt)");

    // ── Build terminal node values ────────────────────────────────────────────
    // Node (i, j): j up-moves from S, so S * u^j * d^(i-j) at step i.
    // At maturity (i = steps): S * u^j * d^(steps-j) for j = 0..steps.
    int N = steps + 1;
    std::vector<double> V(N);
    for (int j = 0; j < N; ++j) {
        double ST = S * std::pow(u, 2 * j - steps); // u^j * d^(steps-j) = u^(2j-steps)
        if (type == OptionType::CALL)
            V[j] = std::max(ST - K, 0.0);
        else
            V[j] = std::max(K - ST, 0.0);
    }

    // ── Backward induction ────────────────────────────────────────────────────
    for (int i = steps - 1; i >= 0; --i) {
        for (int j = 0; j <= i; ++j) {
            double hold = disc * (pu * V[j+1] + pd * V[j]);

            if (style == ExerciseStyle::AMERICAN) {
                double ST = S * std::pow(u, 2 * j - i);
                double intrinsic = (type == OptionType::CALL) ? std::max(ST - K, 0.0)
                                                              : std::max(K - ST, 0.0);
                V[j] = std::max(hold, intrinsic);
            } else {
                V[j] = hold;
            }
        }
    }

    // ── Extract price and first-order Greeks from root neighbours ─────────────
    // Rerun one step forward to get V at step 1 (3 nodes: down, mid≈0, up)
    // We stored the final V[0] = price.  For Greeks we need the step-1 values.
    // Trick: re-expand the already-collapsed V array's first two values.
    double price = V[0];

    // Re-calculate step-1 node values (still in V after i=0 iteration)
    // V[0] = price at node (0,0). After the loop V only has one element used.
    // Compute delta and gamma from i=1 sub-tree stored in V[0], V[1] (still valid).
    // Actually after full collapse: we need a partial re-run. Store i=1 values.

    // Partial second pass to get the three step-1 values for delta/gamma/theta
    // Reset terminal nodes
    for (int j = 0; j < N; ++j) {
        double ST = S * std::pow(u, 2 * j - steps);
        if (type == OptionType::CALL) V[j] = std::max(ST - K, 0.0);
        else                          V[j] = std::max(K - ST, 0.0);
    }
    // Backward to step 2
    for (int i = steps - 1; i >= 2; --i)
        for (int j = 0; j <= i; ++j) {
            double hold = disc * (pu * V[j+1] + pd * V[j]);
            if (style == ExerciseStyle::AMERICAN) {
                double ST = S * std::pow(u, 2 * j - i);
                double intr = (type == OptionType::CALL) ? std::max(ST - K, 0.0)
                                                         : std::max(K - ST, 0.0);
                V[j] = std::max(hold, intr);
            } else {
                V[j] = hold;
            }
        }
    // V[0], V[1], V[2] are now at step 2 (3 nodes)
    double Vuu = V[2], Vud = V[1], Vdd = V[0];
    double Su  = S * u, Sd = S * d;
    // One more step to get V at step 1
    double Vu  = disc * (pu * Vuu + pd * Vud);
    double Vd  = disc * (pu * Vud + pd * Vdd);
    if (style == ExerciseStyle::AMERICAN) {
        double intr_u = (type == OptionType::CALL) ? std::max(Su - K, 0.0) : std::max(K - Su, 0.0);
        double intr_d = (type == OptionType::CALL) ? std::max(Sd - K, 0.0) : std::max(K - Sd, 0.0);
        Vu = std::max(Vu, intr_u);
        Vd = std::max(Vd, intr_d);
    }

    double delta = (Vu - Vd) / (Su - Sd);
    double gamma = 2.0 * ((Vu - price) / (Su - S) - (price - Vd) / (S - Sd))
                   / (Su - Sd);

    // Theta from price at T-dt (one step in): approximate as finite difference
    // We use the i=1 estimate of "discounted step-1 average" vs price at T
    double V_step1 = disc * (pu * Vu + pd * Vd);
    double theta   = (V_step1 - price) / (dt * 365.0); // per calendar day

    return { price, delta, gamma, theta, steps };
}

} // namespace options
