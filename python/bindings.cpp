#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pricer.h"
#include "greeks.h"
#include "mc_engine.h"
#include "vol_surface.h"
#include "binomial_tree.h"

namespace py = pybind11;
using namespace options;

PYBIND11_MODULE(options_py, m) {
    m.doc() = "Options Pricing & Risk Engine — Black-Scholes / Monte Carlo (C++17)";

    // ── Enums ────────────────────────────────────────────────────────────────
    py::enum_<OptionType>(m, "OptionType")
        .value("CALL", OptionType::CALL)
        .value("PUT",  OptionType::PUT)
        .export_values();

    py::enum_<ExerciseStyle>(m, "ExerciseStyle")
        .value("EUROPEAN", ExerciseStyle::EUROPEAN)
        .value("AMERICAN", ExerciseStyle::AMERICAN)
        .export_values();

    // ── BSResult ─────────────────────────────────────────────────────────────
    py::class_<BSResult>(m, "BSResult")
        .def_readonly("price", &BSResult::price)
        .def_readonly("d1",    &BSResult::d1)
        .def_readonly("d2",    &BSResult::d2)
        .def("__repr__", [](const BSResult& r) {
            return "<BSResult price=" + std::to_string(r.price) + ">";
        });

    // ── Greeks ───────────────────────────────────────────────────────────────
    py::class_<Greeks>(m, "Greeks")
        .def_readonly("delta", &Greeks::delta)
        .def_readonly("gamma", &Greeks::gamma)
        .def_readonly("vega",  &Greeks::vega)
        .def_readonly("theta", &Greeks::theta)
        .def_readonly("rho",   &Greeks::rho)
        .def_readonly("vanna", &Greeks::vanna)
        .def_readonly("volga", &Greeks::volga)
        .def("__repr__", [](const Greeks& g) {
            return "<Greeks Δ=" + std::to_string(g.delta)
                 + " Γ=" + std::to_string(g.gamma)
                 + " ν=" + std::to_string(g.vega)
                 + " θ=" + std::to_string(g.theta) + ">";
        });

    // ── MCConfig ─────────────────────────────────────────────────────────────
    py::class_<MCConfig>(m, "MCConfig")
        .def(py::init<>())
        .def_readwrite("num_paths",   &MCConfig::num_paths)
        .def_readwrite("num_steps",   &MCConfig::num_steps)
        .def_readwrite("seed",        &MCConfig::seed)
        .def_readwrite("antithetic",  &MCConfig::antithetic)
        .def_readwrite("control_vrt", &MCConfig::control_vrt)
        .def_readwrite("use_sobol",   &MCConfig::use_sobol)
        .def_readwrite("num_threads", &MCConfig::num_threads);

    // ── MCResult ─────────────────────────────────────────────────────────────
    py::class_<MCResult>(m, "MCResult")
        .def_readonly("price",      &MCResult::price)
        .def_readonly("std_error",  &MCResult::std_error)
        .def_readonly("ci_low_95",  &MCResult::ci_low_95)
        .def_readonly("ci_high_95", &MCResult::ci_high_95)
        .def_readonly("paths_used", &MCResult::paths_used)
        .def_readonly("elapsed_ms", &MCResult::elapsed_ms)
        .def("__repr__", [](const MCResult& r) {
            return "<MCResult price=" + std::to_string(r.price)
                 + " se=" + std::to_string(r.std_error)
                 + " ms=" + std::to_string(r.elapsed_ms) + ">";
        });

    // ── IVResult ─────────────────────────────────────────────────────────────
    py::class_<IVResult>(m, "IVResult")
        .def_readonly("implied_vol", &IVResult::implied_vol)
        .def_readonly("iterations",  &IVResult::iterations)
        .def_readonly("converged",   &IVResult::converged)
        .def_readonly("residual",    &IVResult::residual);

    // ── TreeResult ───────────────────────────────────────────────────────────
    py::class_<TreeResult>(m, "TreeResult")
        .def_readonly("price", &TreeResult::price)
        .def_readonly("delta", &TreeResult::delta)
        .def_readonly("gamma", &TreeResult::gamma)
        .def_readonly("theta", &TreeResult::theta)
        .def_readonly("steps", &TreeResult::steps);

    // ── VolSurface ───────────────────────────────────────────────────────────
    py::class_<VolSurface>(m, "VolSurface")
        .def_readonly("S",        &VolSurface::S)
        .def_readonly("expiries", &VolSurface::expiries)
        .def_readonly("strikes",  &VolSurface::strikes)
        .def_readonly("vols",     &VolSurface::vols)
        .def("interpolate", &VolSurface::interpolate, py::arg("K"), py::arg("T"))
        .def("price",       &VolSurface::price,
             py::arg("K"), py::arg("T"), py::arg("type") = OptionType::CALL);

    // ── MonteCarloEngine ─────────────────────────────────────────────────────
    py::class_<MonteCarloEngine>(m, "MonteCarloEngine")
        .def(py::init<const MCConfig&>(), py::arg("config") = MCConfig{})
        .def("price_european", &MonteCarloEngine::price_european,
             py::arg("S"), py::arg("K"), py::arg("r"), py::arg("T"),
             py::arg("sigma"), py::arg("q") = 0.0,
             py::arg("type") = OptionType::CALL)
        .def("price_asian",   &MonteCarloEngine::price_asian,
             py::arg("S"), py::arg("K"), py::arg("r"), py::arg("T"),
             py::arg("sigma"), py::arg("q") = 0.0,
             py::arg("type") = OptionType::CALL)
        .def("price_barrier", &MonteCarloEngine::price_barrier,
             py::arg("S"), py::arg("K"), py::arg("B"),
             py::arg("r"), py::arg("T"), py::arg("sigma"), py::arg("q") = 0.0)
        .def("generate_path", &MonteCarloEngine::generate_path,
             py::arg("S"), py::arg("r"), py::arg("T"), py::arg("sigma"), py::arg("q") = 0.0)
        .def_property_readonly("config", &MonteCarloEngine::config);

    // ── Free functions ────────────────────────────────────────────────────────
    m.def("black_scholes", &black_scholes,
          py::arg("S"), py::arg("K"), py::arg("r"), py::arg("T"),
          py::arg("sigma"), py::arg("q") = 0.0,
          py::arg("type") = OptionType::CALL,
          "Full Black-Scholes result (price, d1, d2).");

    m.def("bs_call", &bs_call,
          py::arg("S"), py::arg("K"), py::arg("r"), py::arg("T"),
          py::arg("sigma"), py::arg("q") = 0.0,
          "Black-Scholes European call price.");

    m.def("bs_put", &bs_put,
          py::arg("S"), py::arg("K"), py::arg("r"), py::arg("T"),
          py::arg("sigma"), py::arg("q") = 0.0,
          "Black-Scholes European put price.");

    m.def("analytical_greeks", &analytical_greeks,
          py::arg("S"), py::arg("K"), py::arg("r"), py::arg("T"),
          py::arg("sigma"), py::arg("q") = 0.0,
          py::arg("type") = OptionType::CALL,
          "Closed-form Greeks (delta, gamma, vega, theta, rho, vanna, volga).");

    m.def("numerical_greeks",
          [](double S, double K, double r, double T,
             double sigma, double q, OptionType type) {
              return numerical_greeks(S, K, r, T, sigma, q, type);
          },
          py::arg("S"), py::arg("K"), py::arg("r"), py::arg("T"),
          py::arg("sigma"), py::arg("q") = 0.0,
          py::arg("type") = OptionType::CALL,
          "Finite-difference numerical Greeks for cross-validation.");

    m.def("implied_vol", &implied_vol,
          py::arg("market_price"), py::arg("S"), py::arg("K"),
          py::arg("r"), py::arg("T"), py::arg("q") = 0.0,
          py::arg("type") = OptionType::CALL,
          py::arg("initial_guess") = 0.20,
          py::arg("tol") = 1e-8, py::arg("max_iter") = 100,
          "Newton-Raphson implied volatility solver.");

    m.def("build_vol_surface", &build_vol_surface,
          py::arg("S"), py::arg("r"), py::arg("q"),
          py::arg("expiries"), py::arg("strikes"), py::arg("market_prices"),
          py::arg("type") = OptionType::CALL,
          "Build volatility surface from market prices.");

    m.def("crr_tree", &crr_tree,
          py::arg("S"), py::arg("K"), py::arg("r"), py::arg("T"),
          py::arg("sigma"), py::arg("q") = 0.0,
          py::arg("type")  = OptionType::CALL,
          py::arg("style") = ExerciseStyle::EUROPEAN,
          py::arg("steps") = 500,
          "Cox-Ross-Rubinstein binomial tree (European & American).");

    m.def("norm_cdf", &norm_cdf, py::arg("x"), "Standard normal CDF.");
    m.def("norm_pdf", &norm_pdf, py::arg("x"), "Standard normal PDF.");
}
