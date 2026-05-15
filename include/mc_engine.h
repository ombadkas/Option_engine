#pragma once
#include "pricer.h"
#include <vector>
#include <cstdint>

namespace options {

// ── Configuration ─────────────────────────────────────────────────────────────
struct MCConfig {
    uint64_t num_paths     = 100'000;
    uint32_t num_steps     = 1;       // 1 = exact terminal dist for European
    uint32_t seed          = 42;
    bool     antithetic    = true;    // antithetic variates (halves variance)
    bool     control_vrt   = false;   // control variate using B-S price
    bool     use_sobol     = false;   // quasi-random Van der Corput / Sobol dim-1
    int      num_threads   = 0;       // 0 = all available cores
};

// ── Result ────────────────────────────────────────────────────────────────────
struct MCResult {
    double   price;
    double   std_error;
    double   ci_low_95;
    double   ci_high_95;
    uint64_t paths_used;
    double   elapsed_ms;
};

// ── Sobol / Van der Corput generator (dim 1 & 2) ─────────────────────────────
// Implements Gray-code enumeration with Joe-Kuo 2010 direction numbers.
// Dim 0 = Van der Corput (base-2 radical inverse).
// Dim 1 = Sobol dim-2 (primitive polynomial x+1, s=1, a=0, m=[1]).
class SobolGenerator {
public:
    explicit SobolGenerator(uint32_t skip = 0);
    // Returns next point in [0,1); call twice for antithetic pair
    double next();
    void   reset(uint32_t skip = 0);

private:
    static const uint32_t V[32]; // direction numbers for dim 1 (Van der Corput)
    uint32_t x_;   // current fixed-point value
    uint32_t n_;   // current index
};

// ── Monte Carlo engine ────────────────────────────────────────────────────────
class MonteCarloEngine {
public:
    explicit MonteCarloEngine(const MCConfig& cfg = {});

    // European call / put — uses terminal GBM distribution (exact)
    MCResult price_european(double S, double K, double r, double T,
                            double sigma, double q = 0.0,
                            OptionType type = OptionType::CALL) const;

    // Arithmetic-average Asian option (path-dependent, needs num_steps > 1)
    MCResult price_asian(double S, double K, double r, double T,
                         double sigma, double q = 0.0,
                         OptionType type = OptionType::CALL) const;

    // Down-and-out barrier call (B = barrier level below spot)
    MCResult price_barrier(double S, double K, double B,
                           double r, double T, double sigma, double q = 0.0) const;

    // Generate a single GBM price path (length = num_steps+1, starts at S)
    std::vector<double> generate_path(double S, double r, double T,
                                      double sigma, double q = 0.0) const;

    const MCConfig& config() const { return cfg_; }

private:
    MCConfig cfg_;

    MCResult run_european_pseudo (double S, double K, double r, double T,
                                  double sigma, double q, OptionType type) const;
    MCResult run_european_sobol  (double S, double K, double r, double T,
                                  double sigma, double q, OptionType type) const;
};

} // namespace options
