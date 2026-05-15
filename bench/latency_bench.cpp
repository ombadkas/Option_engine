// Google Benchmark suite for the Options Pricing & Risk Engine.
// Build: cmake .. -DBUILD_BENCHMARKS=ON -Dbenchmark_DIR=<path>
//        cmake --build . --config Release
//        ./latency_bench --benchmark_format=json

#include <benchmark/benchmark.h>
#include "pricer.h"
#include "greeks.h"
#include "mc_engine.h"
#include "vol_surface.h"
#include "binomial_tree.h"

using namespace options;

// ── Test parameters ───────────────────────────────────────────────────────────
static constexpr double S     = 100.0;
static constexpr double K     = 100.0;
static constexpr double r     = 0.05;
static constexpr double T     = 1.0;
static constexpr double sigma = 0.20;
static constexpr double q     = 0.0;

// ── Black-Scholes pricing ─────────────────────────────────────────────────────
static void BM_BS_Call(benchmark::State& state) {
    for (auto _ : state) {
        double p = bs_call(S, K, r, T, sigma, q);
        benchmark::DoNotOptimize(p);
    }
}
BENCHMARK(BM_BS_Call)->MinTime(1.0);

static void BM_BS_Put(benchmark::State& state) {
    for (auto _ : state) {
        double p = bs_put(S, K, r, T, sigma, q);
        benchmark::DoNotOptimize(p);
    }
}
BENCHMARK(BM_BS_Put)->MinTime(1.0);

// ── Greeks ────────────────────────────────────────────────────────────────────
static void BM_AnalyticalGreeks(benchmark::State& state) {
    for (auto _ : state) {
        Greeks g = analytical_greeks(S, K, r, T, sigma, q, OptionType::CALL);
        benchmark::DoNotOptimize(g);
    }
}
BENCHMARK(BM_AnalyticalGreeks)->MinTime(1.0);

static void BM_NumericalGreeks(benchmark::State& state) {
    for (auto _ : state) {
        Greeks g = numerical_greeks(S, K, r, T, sigma, q, OptionType::CALL);
        benchmark::DoNotOptimize(g);
    }
}
BENCHMARK(BM_NumericalGreeks)->MinTime(1.0);

// ── Implied volatility ────────────────────────────────────────────────────────
static void BM_ImpliedVol(benchmark::State& state) {
    double mkt = bs_call(S, K, r, T, sigma, q);
    for (auto _ : state) {
        IVResult iv = implied_vol(mkt, S, K, r, T, q, OptionType::CALL);
        benchmark::DoNotOptimize(iv);
    }
}
BENCHMARK(BM_ImpliedVol)->MinTime(1.0);

// ── Monte Carlo ───────────────────────────────────────────────────────────────
static void BM_MC_100k_Standard(benchmark::State& state) {
    MCConfig cfg;
    cfg.num_paths  = 100'000;
    cfg.antithetic = false;
    MonteCarloEngine eng(cfg);
    for (auto _ : state) {
        MCResult r = eng.price_european(S, K, ::r, T, sigma, q, OptionType::CALL);
        benchmark::DoNotOptimize(r);
    }
    state.SetItemsProcessed(state.iterations() * 100'000);
}
BENCHMARK(BM_MC_100k_Standard)->MinTime(2.0)->Unit(benchmark::kMillisecond);

static void BM_MC_100k_Antithetic(benchmark::State& state) {
    MCConfig cfg;
    cfg.num_paths  = 100'000;
    cfg.antithetic = true;
    MonteCarloEngine eng(cfg);
    for (auto _ : state) {
        MCResult r = eng.price_european(S, K, ::r, T, sigma, q, OptionType::CALL);
        benchmark::DoNotOptimize(r);
    }
    state.SetItemsProcessed(state.iterations() * 100'000);
}
BENCHMARK(BM_MC_100k_Antithetic)->MinTime(2.0)->Unit(benchmark::kMillisecond);

static void BM_MC_100k_Sobol(benchmark::State& state) {
    MCConfig cfg;
    cfg.num_paths  = 100'000;
    cfg.antithetic = true;
    cfg.use_sobol  = true;
    MonteCarloEngine eng(cfg);
    for (auto _ : state) {
        MCResult r = eng.price_european(S, K, ::r, T, sigma, q, OptionType::CALL);
        benchmark::DoNotOptimize(r);
    }
    state.SetItemsProcessed(state.iterations() * 100'000);
}
BENCHMARK(BM_MC_100k_Sobol)->MinTime(2.0)->Unit(benchmark::kMillisecond);

static void BM_MC_1M_Antithetic(benchmark::State& state) {
    MCConfig cfg;
    cfg.num_paths  = 1'000'000;
    cfg.antithetic = true;
    MonteCarloEngine eng(cfg);
    for (auto _ : state) {
        MCResult r = eng.price_european(S, K, ::r, T, sigma, q, OptionType::CALL);
        benchmark::DoNotOptimize(r);
    }
    state.SetItemsProcessed(state.iterations() * 1'000'000);
}
BENCHMARK(BM_MC_1M_Antithetic)->MinTime(2.0)->Unit(benchmark::kMillisecond);

static void BM_MC_10M_Antithetic(benchmark::State& state) {
    MCConfig cfg;
    cfg.num_paths  = 10'000'000;
    cfg.antithetic = true;
    MonteCarloEngine eng(cfg);
    for (auto _ : state) {
        MCResult r = eng.price_european(S, K, ::r, T, sigma, q, OptionType::CALL);
        benchmark::DoNotOptimize(r);
    }
    state.SetItemsProcessed(state.iterations() * 10'000'000);
}
BENCHMARK(BM_MC_10M_Antithetic)->MinTime(2.0)->Unit(benchmark::kMillisecond);

// ── Asian option (path-dependent, 252 steps) ──────────────────────────────────
static void BM_MC_Asian_100k(benchmark::State& state) {
    MCConfig cfg;
    cfg.num_paths  = 100'000;
    cfg.num_steps  = 252;
    cfg.antithetic = false;
    MonteCarloEngine eng(cfg);
    for (auto _ : state) {
        MCResult r = eng.price_asian(S, K, ::r, T, sigma, q, OptionType::CALL);
        benchmark::DoNotOptimize(r);
    }
    state.SetItemsProcessed(state.iterations() * 100'000);
}
BENCHMARK(BM_MC_Asian_100k)->MinTime(2.0)->Unit(benchmark::kMillisecond);

// ── Binomial tree ─────────────────────────────────────────────────────────────
static void BM_CRR_500(benchmark::State& state) {
    for (auto _ : state) {
        TreeResult t = crr_tree(S, K, r, T, sigma, q,
                                OptionType::CALL, ExerciseStyle::EUROPEAN, 500);
        benchmark::DoNotOptimize(t);
    }
}
BENCHMARK(BM_CRR_500)->MinTime(1.0);

static void BM_CRR_American_500(benchmark::State& state) {
    for (auto _ : state) {
        TreeResult t = crr_tree(S, K, r, T, sigma, q,
                                OptionType::PUT, ExerciseStyle::AMERICAN, 500);
        benchmark::DoNotOptimize(t);
    }
}
BENCHMARK(BM_CRR_American_500)->MinTime(1.0);

BENCHMARK_MAIN();
