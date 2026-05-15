// Validates analytical Greeks against finite-difference numerical Greeks.
// Also checks key mathematical properties (put-call parity for delta/rho,
// gamma non-negativity, theta sign, etc.)

#include "greeks.h"
#include "pricer.h"
#include <cmath>
#include <cstdio>
#include <vector>

using namespace options;

static int g_pass = 0, g_fail = 0;

static void check_near(const char* name, double got, double expected, double tol) {
    bool ok = std::abs(got - expected) <= tol;
    if (ok) {
        std::printf("  PASS  %-50s  Δ=%.2e\n", name, std::abs(got - expected));
    } else {
        std::printf("  FAIL  %-50s  got=%.8f  exp=%.8f  |err|=%.2e  tol=%.2e\n",
                    name, got, expected, std::abs(got - expected), tol);
    }
    ok ? ++g_pass : ++g_fail;
}

static void check(const char* name, bool cond) {
    if (cond) { ++g_pass; std::printf("  PASS  %s\n", name); }
    else       { ++g_fail; std::printf("  FAIL  %s\n", name); }
}

// ── Analytical vs numerical cross-validation ──────────────────────────────────
static void test_greeks_analytical_vs_numerical(double S, double K, double r,
                                                 double T, double sigma, double q,
                                                 OptionType type, const char* tag)
{
    Greeks a = analytical_greeks(S, K, r, T, sigma, q, type);

    // Use tighter bumps for numerical Greeks
    FDParams fp;
    fp.dS_rel  = 0.0005;
    fp.dsigma  = 0.00005;
    fp.dr      = 0.00005;
    fp.dT_days = 0.1;

    Greeks n = numerical_greeks(S, K, r, T, sigma, q, type, fp);

    char buf[128];
    // Delta: tight tolerance
    std::snprintf(buf, sizeof(buf), "[%s] delta  analytical vs numerical", tag);
    check_near(buf, a.delta, n.delta, 1e-4);

    // Gamma
    std::snprintf(buf, sizeof(buf), "[%s] gamma  analytical vs numerical", tag);
    check_near(buf, a.gamma, n.gamma, 1e-4);

    // Vega (per vol pt)
    std::snprintf(buf, sizeof(buf), "[%s] vega   analytical vs numerical", tag);
    check_near(buf, a.vega, n.vega, 1e-4);

    // Theta (per day)
    std::snprintf(buf, sizeof(buf), "[%s] theta  analytical vs numerical", tag);
    check_near(buf, a.theta, n.theta, 5e-4);

    // Rho (per bp)
    std::snprintf(buf, sizeof(buf), "[%s] rho    analytical vs numerical", tag);
    check_near(buf, a.rho, n.rho, 1e-5);
}

// ── Properties ───────────────────────────────────────────────────────────────
static void test_delta_bounds() {
    std::printf("\n[Delta bounds]\n");
    // Call delta in (0, 1), put delta in (-1, 0)
    for (double s : {70.0, 90.0, 100.0, 110.0, 130.0}) {
        Greeks gc = analytical_greeks(s, 100, 0.05, 1.0, 0.20, 0.0, OptionType::CALL);
        Greeks gp = analytical_greeks(s, 100, 0.05, 1.0, 0.20, 0.0, OptionType::PUT);
        char buf[64];
        std::snprintf(buf, sizeof(buf), "call delta in (0,1) S=%.0f", s);
        check(buf, gc.delta > 0.0 && gc.delta < 1.0);
        std::snprintf(buf, sizeof(buf), "put  delta in (-1,0) S=%.0f", s);
        check(buf, gp.delta < 0.0 && gp.delta > -1.0);
    }
}

static void test_put_call_delta_relation() {
    std::printf("\n[Delta put-call parity: Δ_call - Δ_put = e^(-qT)]\n");
    double S=100, K=100, r=0.05, T=1.0, sigma=0.20, q=0.02;
    Greeks gc = analytical_greeks(S, K, r, T, sigma, q, OptionType::CALL);
    Greeks gp = analytical_greeks(S, K, r, T, sigma, q, OptionType::PUT);
    check_near("Δ_call - Δ_put = e^(-qT)", gc.delta - gp.delta, std::exp(-q * T), 1e-12);
}

static void test_gamma_nonneg() {
    std::printf("\n[Gamma >= 0 for long options]\n");
    for (double s : {70.0, 90.0, 100.0, 110.0, 130.0}) {
        Greeks gc = analytical_greeks(s, 100, 0.05, 1.0, 0.20);
        char buf[64];
        std::snprintf(buf, sizeof(buf), "gamma >= 0 S=%.0f", s);
        check(buf, gc.gamma >= 0.0);
    }
}

static void test_gamma_put_call_equal() {
    std::printf("\n[Gamma call == Gamma put (same parameters)]\n");
    double S=100, K=100, r=0.05, T=1.0, sigma=0.20, q=0.0;
    Greeks gc = analytical_greeks(S, K, r, T, sigma, q, OptionType::CALL);
    Greeks gp = analytical_greeks(S, K, r, T, sigma, q, OptionType::PUT);
    check_near("gamma_call == gamma_put", gc.gamma, gp.gamma, 1e-14);
}

static void test_vega_put_call_equal() {
    std::printf("\n[Vega call == Vega put (same parameters)]\n");
    double S=100, K=100, r=0.05, T=1.0, sigma=0.20, q=0.0;
    Greeks gc = analytical_greeks(S, K, r, T, sigma, q, OptionType::CALL);
    Greeks gp = analytical_greeks(S, K, r, T, sigma, q, OptionType::PUT);
    check_near("vega_call == vega_put", gc.vega, gp.vega, 1e-14);
}

static void test_theta_sign() {
    std::printf("\n[Theta <= 0 for long European options (time decay)]\n");
    double S=100, K=100, r=0.05, T=1.0, sigma=0.20, q=0.0;
    Greeks gc = analytical_greeks(S, K, r, T, sigma, q, OptionType::CALL);
    Greeks gp = analytical_greeks(S, K, r, T, sigma, q, OptionType::PUT);
    // Theta is negative for long ATM options (when r > 0, ITM put can have positive theta)
    check("ATM call theta < 0", gc.theta < 0.0);
    // Deep ITM put with high rate may have positive theta (receiving carry exceeds time decay)
    // For ATM put with r=0.05: should also be negative
    check("ATM put theta < 0", gp.theta < 0.0);
}

static void test_rho_sign() {
    std::printf("\n[Rho sign: call rho > 0, put rho < 0]\n");
    Greeks gc = analytical_greeks(100, 100, 0.05, 1.0, 0.20, 0.0, OptionType::CALL);
    Greeks gp = analytical_greeks(100, 100, 0.05, 1.0, 0.20, 0.0, OptionType::PUT);
    check("call rho > 0", gc.rho > 0.0);
    check("put  rho < 0", gp.rho < 0.0);
}

static void test_greeks_known_values() {
    std::printf("\n[Greeks known values — ATM call S=100, K=100, r=0.05, T=1, σ=0.20]\n");
    // Reference: hand-calculated / QuantLib
    double S=100, K=100, r=0.05, T=1.0, sigma=0.20, q=0.0;
    Greeks g = analytical_greeks(S, K, r, T, sigma, q, OptionType::CALL);

    check_near("delta ATM call ≈ 0.6368", g.delta,  0.636831, 1e-4);
    check_near("gamma ATM call ≈ 0.0188", g.gamma,  0.018762, 1e-4);
    check_near("vega  ATM call ≈ 0.3752", g.vega,   0.375241, 1e-4);  // per 1 vol pt
    check_near("theta ATM call ≈ -0.0176",g.theta, -0.017619, 1e-4);  // per day
    check_near("rho   ATM call ≈ 0.0053", g.rho,   0.005323, 1e-5);   // per bp
}

static void test_vanna_volga() {
    std::printf("\n[Vanna and Volga — cross-check with finite differences]\n");
    double S=100, K=100, r=0.05, T=1.0, sigma=0.20, q=0.0;
    Greeks a = analytical_greeks(S, K, r, T, sigma, q, OptionType::CALL);

    // Numerical vanna: (delta(sigma+dsig) - delta(sigma-dsig)) / (2*dsig)
    double dsig = 0.001;
    Greeks g_up = analytical_greeks(S, K, r, T, sigma+dsig, q, OptionType::CALL);
    Greeks g_dn = analytical_greeks(S, K, r, T, sigma-dsig, q, OptionType::CALL);
    double num_vanna = (g_up.delta - g_dn.delta) / (2.0 * dsig);
    check_near("vanna numerical vs analytical", a.vanna, num_vanna, 2e-5);

    // Numerical volga: d(vega)/d(sigma). Both vega and volga carry the same /100 scaling
    // (per vol point), so the finite difference is simply delta_vega / (2*dsig).
    double num_volga = (g_up.vega - g_dn.vega) / (2.0 * dsig);
    check_near("volga numerical vs analytical", a.volga, num_volga, 1e-5);
}

int main() {
    std::printf("=== Greeks Test Suite ===\n");

    // Cross-validation: analytical vs numerical for various configurations
    std::printf("\n[Analytical vs Numerical — ATM call]\n");
    test_greeks_analytical_vs_numerical(100, 100, 0.05, 1.0, 0.20, 0.0,
                                        OptionType::CALL, "ATM call");

    std::printf("\n[Analytical vs Numerical — OTM call]\n");
    test_greeks_analytical_vs_numerical(100, 120, 0.05, 0.5, 0.25, 0.0,
                                        OptionType::CALL, "OTM call");

    std::printf("\n[Analytical vs Numerical — ITM put + dividend]\n");
    test_greeks_analytical_vs_numerical(100, 90, 0.05, 2.0, 0.30, 0.03,
                                        OptionType::PUT, "ITM put+div");

    std::printf("\n[Analytical vs Numerical — short-dated NTM]\n");
    test_greeks_analytical_vs_numerical(100, 102, 0.03, 0.1, 0.18, 0.01,
                                        OptionType::CALL, "short NTM");

    // Property tests
    test_delta_bounds();
    test_put_call_delta_relation();
    test_gamma_nonneg();
    test_gamma_put_call_equal();
    test_vega_put_call_equal();
    test_theta_sign();
    test_rho_sign();
    test_greeks_known_values();
    test_vanna_volga();

    std::printf("\n=== Results: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
