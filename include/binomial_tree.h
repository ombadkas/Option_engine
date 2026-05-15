#pragma once
#include "pricer.h"

namespace options {

// ── CRR binomial tree result ──────────────────────────────────────────────────
struct TreeResult {
    double price;
    double delta;   // from first-order tree differences
    double gamma;   // from second-order tree differences
    double theta;   // from central time difference
    int    steps;
};

// Cox-Ross-Rubinstein binomial lattice.
//
// Works for both European and American exercise.
// Convergence to B-S improves as steps → ∞; 500 steps gives < 0.01 % error.
//
// American put: early exercise checked at every node.
// American call on non-dividend stock: always equals European (never optimal
//   to exercise early) — the tree handles this correctly.
TreeResult crr_tree(double S, double K, double r, double T,
                    double sigma, double q = 0.0,
                    OptionType     type  = OptionType::CALL,
                    ExerciseStyle  style = ExerciseStyle::EUROPEAN,
                    int steps = 500);

} // namespace options
