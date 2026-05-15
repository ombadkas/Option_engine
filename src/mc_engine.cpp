#include "mc_engine.h"
#include "pricer.h"
#include <chrono>
#include <cmath>
#include <random>
#include <stdexcept>
#include <numeric>

#ifdef OMP_AVAILABLE
#  include <omp.h>
#  define THREAD_ID omp_get_thread_num()
#else
#  define THREAD_ID 0
#endif

namespace options {

// ── Sobol / Van der Corput ────────────────────────────────────────────────────
// Direction numbers for dim 0 (Van der Corput, base 2): V[i] = 2^(31-i).
// Sequence generated via Gray-code enumeration: x_n = x_{n-1} XOR V[c(n)]
// where c(n) = position of least-significant 0 bit of (n-1).
const uint32_t SobolGenerator::V[32] = {
    0x80000000u, 0x40000000u, 0x20000000u, 0x10000000u,
    0x08000000u, 0x04000000u, 0x02000000u, 0x01000000u,
    0x00800000u, 0x00400000u, 0x00200000u, 0x00100000u,
    0x00080000u, 0x00040000u, 0x00020000u, 0x00010000u,
    0x00008000u, 0x00004000u, 0x00002000u, 0x00001000u,
    0x00000800u, 0x00000400u, 0x00000200u, 0x00000100u,
    0x00000080u, 0x00000040u, 0x00000020u, 0x00000010u,
    0x00000008u, 0x00000004u, 0x00000002u, 0x00000001u
};

SobolGenerator::SobolGenerator(uint32_t skip) : x_(0), n_(0) {
    for (uint32_t i = 0; i < skip; ++i) next();
}

double SobolGenerator::next() {
    // c = position of rightmost 0 bit of n_ (0-indexed)
    uint32_t c = 0;
    uint32_t tmp = n_;
    while (tmp & 1u) { tmp >>= 1; ++c; }
    x_ ^= V[c];
    ++n_;
    // Map fixed-point to (0,1): add 0.5 to stay away from boundaries
    return (static_cast<double>(x_) + 0.5) / 4294967296.0; // / 2^32
}

void SobolGenerator::reset(uint32_t skip) {
    x_ = 0; n_ = 0;
    for (uint32_t i = 0; i < skip; ++i) next();
}

// ── Inverse normal CDF (Acklam rational approximation, |err| < 1.15e-9) ──────
static double norm_ppf(double p) {
    static constexpr double a[] = {
        -3.969683028665376e+01,  2.209460984245205e+02,
        -2.759285104469687e+02,  1.383577518672690e+02,
        -3.066479806614716e+01,  2.506628277459239e+00 };
    static constexpr double b[] = {
        -5.447609879822406e+01,  1.615858368580409e+02,
        -1.556989798598866e+02,  6.680131188771972e+01,
        -1.328068155288572e+01 };
    static constexpr double c[] = {
        -7.784894002430293e-03, -3.223964580411365e-01,
        -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00,  2.938163982698783e+00 };
    static constexpr double d[] = {
         7.784695709041462e-03,  3.224671290700398e-01,
         2.445134137142996e+00,  3.754408661907416e+00 };

    static constexpr double lo = 0.02425;
    static constexpr double hi = 1.0 - lo;

    if (p <= 0.0) return -38.0;
    if (p >= 1.0) return  38.0;

    double q, r;
    if (p < lo) {
        q = std::sqrt(-2.0 * std::log(p));
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])
              /((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
    }
    if (p <= hi) {
        q = p - 0.5; r = q * q;
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q
              /(((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0);
    }
    q = std::sqrt(-2.0 * std::log(1.0 - p));
    return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])
            /((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
}

// ── Engine implementation ─────────────────────────────────────────────────────
MonteCarloEngine::MonteCarloEngine(const MCConfig& cfg) : cfg_(cfg) {}

MCResult MonteCarloEngine::price_european(double S, double K, double r, double T,
                                          double sigma, double q, OptionType type) const
{
    if (cfg_.use_sobol)
        return run_european_sobol(S, K, r, T, sigma, q, type);
    return run_european_pseudo(S, K, r, T, sigma, q, type);
}

// ── Pseudo-random path (Mersenne Twister + antithetic) ────────────────────────
MCResult MonteCarloEngine::run_european_pseudo(double S, double K, double r, double T,
                                               double sigma, double q, OptionType type) const
{
    auto t0 = std::chrono::high_resolution_clock::now();

    const bool  anti   = cfg_.antithetic;
    const bool  cv     = cfg_.control_vrt;
    const uint64_t N   = cfg_.num_paths;
    const uint64_t Nhalf = N / 2;

    // Terminal log-return parameters (exact GBM, no discretisation error)
    double mu    = (r - q - 0.5 * sigma * sigma) * T;
    double vol   = sigma * std::sqrt(T);
    double disc  = std::exp(-r * T);

    // Control variate coefficient (for CV estimator only)
    double bs_price = black_scholes(S, K, r, T, sigma, q, type).price;
    double cv_coeff = cv ? 1.0 : 0.0; // beta=1 is optimal for B-S (see Glasserman)

    double sum   = 0.0;
    double sumSq = 0.0;
    uint64_t paths_used = anti ? 2 * Nhalf : N;

#pragma omp parallel reduction(+:sum) reduction(+:sumSq) if(cfg_.num_threads != 1)
    {
        int tid = THREAD_ID;
        std::mt19937_64 rng(static_cast<uint64_t>(cfg_.seed) + static_cast<uint64_t>(tid) * 6364136223846793005ULL);
        std::normal_distribution<double> normal(0.0, 1.0);

        if (anti) {
#pragma omp for schedule(static)
            for (int64_t i = 0; i < static_cast<int64_t>(Nhalf); ++i) {
                double z  = normal(rng);
                double S1 = S * std::exp(mu + vol * z);
                double S2 = S * std::exp(mu - vol * z);

                double p1 = (type == OptionType::CALL) ? std::max(S1 - K, 0.0)
                                                       : std::max(K - S1, 0.0);
                double p2 = (type == OptionType::CALL) ? std::max(S2 - K, 0.0)
                                                       : std::max(K - S2, 0.0);
                double avg = 0.5 * (p1 + p2);

                if (cv) avg -= cv_coeff * (0.5 * (S1 + S2) - S * std::exp((r - q) * T));

                sum   += avg;
                sumSq += avg * avg;
            }
        } else {
#pragma omp for schedule(static)
            for (int64_t i = 0; i < static_cast<int64_t>(N); ++i) {
                double z  = normal(rng);
                double ST = S * std::exp(mu + vol * z);
                double pv = (type == OptionType::CALL) ? std::max(ST - K, 0.0)
                                                       : std::max(K - ST, 0.0);
                if (cv) pv -= cv_coeff * (ST - S * std::exp((r - q) * T));
                sum   += pv;
                sumSq += pv * pv;
            }
        }
    }

    uint64_t n_eff = anti ? Nhalf : N;
    double mean    = sum / static_cast<double>(n_eff);
    double var     = (sumSq / static_cast<double>(n_eff) - mean * mean)
                     / static_cast<double>(n_eff);
    double se      = std::sqrt(std::max(var, 0.0));

    if (cv) mean += bs_price; // add back control mean

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    return { disc * mean,
             disc * se,
             disc * (mean - 1.96 * se),
             disc * (mean + 1.96 * se),
             paths_used, ms };
}

// ── Quasi-random Sobol path (single-threaded) ─────────────────────────────────
MCResult MonteCarloEngine::run_european_sobol(double S, double K, double r, double T,
                                              double sigma, double q, OptionType type) const
{
    auto t0 = std::chrono::high_resolution_clock::now();

    const uint64_t N = cfg_.num_paths;
    double mu   = (r - q - 0.5 * sigma * sigma) * T;
    double vol  = sigma * std::sqrt(T);
    double disc = std::exp(-r * T);

    SobolGenerator sobol;
    double sum = 0.0, sumSq = 0.0;

    // Use antithetic pairs drawn from Sobol — variance further reduced
    for (uint64_t i = 0; i < N / 2; ++i) {
        double u  = sobol.next();
        double z  = norm_ppf(u);
        double S1 = S * std::exp(mu + vol * z);
        double S2 = S * std::exp(mu - vol * z);

        double p1 = (type == OptionType::CALL) ? std::max(S1 - K, 0.0)
                                               : std::max(K - S1, 0.0);
        double p2 = (type == OptionType::CALL) ? std::max(S2 - K, 0.0)
                                               : std::max(K - S2, 0.0);
        double avg = 0.5 * (p1 + p2);
        sum   += avg;
        sumSq += avg * avg;
    }

    uint64_t n_eff = N / 2;
    double mean = sum   / static_cast<double>(n_eff);
    double var  = (sumSq / static_cast<double>(n_eff) - mean * mean)
                  / static_cast<double>(n_eff);
    double se   = std::sqrt(std::max(var, 0.0));

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    return { disc * mean,
             disc * se,
             disc * (mean - 1.96 * se),
             disc * (mean + 1.96 * se),
             n_eff, ms };
}

// ── Asian option ──────────────────────────────────────────────────────────────
MCResult MonteCarloEngine::price_asian(double S, double K, double r, double T,
                                       double sigma, double q, OptionType type) const
{
    auto t0 = std::chrono::high_resolution_clock::now();

    const uint32_t steps = std::max(cfg_.num_steps, 1u);
    const uint64_t N     = cfg_.num_paths;
    double dt     = T / steps;
    double drift  = (r - q - 0.5 * sigma * sigma) * dt;
    double vdt    = sigma * std::sqrt(dt);
    double disc   = std::exp(-r * T);

    double sum = 0.0, sumSq = 0.0;

#pragma omp parallel reduction(+:sum) reduction(+:sumSq) if(cfg_.num_threads != 1)
    {
        int tid = THREAD_ID;
        std::mt19937_64 rng(static_cast<uint64_t>(cfg_.seed) + static_cast<uint64_t>(tid + 1) * 1234567891ULL);
        std::normal_distribution<double> normal(0.0, 1.0);

#pragma omp for schedule(static)
        for (int64_t i = 0; i < static_cast<int64_t>(N); ++i) {
            double Si = S, avg = 0.0;
            for (uint32_t step = 0; step < steps; ++step) {
                Si  *= std::exp(drift + vdt * normal(rng));
                avg += Si;
            }
            avg /= steps;
            double pv = (type == OptionType::CALL) ? std::max(avg - K, 0.0)
                                                   : std::max(K - avg, 0.0);
            sum   += pv;
            sumSq += pv * pv;
        }
    }

    double mean = sum   / static_cast<double>(N);
    double var  = (sumSq / static_cast<double>(N) - mean * mean)
                  / static_cast<double>(N);
    double se   = std::sqrt(std::max(var, 0.0));

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    return { disc * mean, disc * se,
             disc * (mean - 1.96 * se), disc * (mean + 1.96 * se),
             N, ms };
}

// ── Down-and-out barrier call ─────────────────────────────────────────────────
MCResult MonteCarloEngine::price_barrier(double S, double K, double B,
                                         double r, double T, double sigma, double q) const
{
    if (B >= S) throw std::invalid_argument("Barrier B must be below spot S");

    auto t0 = std::chrono::high_resolution_clock::now();

    const uint32_t steps = std::max(cfg_.num_steps, 50u);
    const uint64_t N     = cfg_.num_paths;
    double dt    = T / steps;
    double drift = (r - q - 0.5 * sigma * sigma) * dt;
    double vdt   = sigma * std::sqrt(dt);
    double disc  = std::exp(-r * T);

    double sum = 0.0, sumSq = 0.0;

#pragma omp parallel reduction(+:sum) reduction(+:sumSq) if(cfg_.num_threads != 1)
    {
        int tid = THREAD_ID;
        std::mt19937_64 rng(static_cast<uint64_t>(cfg_.seed) + static_cast<uint64_t>(tid + 2) * 9876543211ULL);
        std::normal_distribution<double> normal(0.0, 1.0);

#pragma omp for schedule(static)
        for (int64_t i = 0; i < static_cast<int64_t>(N); ++i) {
            double Si = S;
            bool   knocked_out = false;
            for (uint32_t step = 0; step < steps && !knocked_out; ++step) {
                Si *= std::exp(drift + vdt * normal(rng));
                if (Si <= B) knocked_out = true;
            }
            double pv = knocked_out ? 0.0 : std::max(Si - K, 0.0);
            sum   += pv;
            sumSq += pv * pv;
        }
    }

    double mean = sum   / static_cast<double>(N);
    double var  = (sumSq / static_cast<double>(N) - mean * mean)
                  / static_cast<double>(N);
    double se   = std::sqrt(std::max(var, 0.0));

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    return { disc * mean, disc * se,
             disc * (mean - 1.96 * se), disc * (mean + 1.96 * se),
             N, ms };
}

// ── Single GBM path ───────────────────────────────────────────────────────────
std::vector<double> MonteCarloEngine::generate_path(double S, double r, double T,
                                                     double sigma, double q) const
{
    uint32_t steps = std::max(cfg_.num_steps, 1u);
    double dt      = T / steps;
    double drift   = (r - q - 0.5 * sigma * sigma) * dt;
    double vdt     = sigma * std::sqrt(dt);

    std::mt19937_64 rng(cfg_.seed);
    std::normal_distribution<double> normal(0.0, 1.0);

    std::vector<double> path;
    path.reserve(steps + 1);
    path.push_back(S);
    for (uint32_t i = 0; i < steps; ++i) {
        S *= std::exp(drift + vdt * normal(rng));
        path.push_back(S);
    }
    return path;
}

} // namespace options
