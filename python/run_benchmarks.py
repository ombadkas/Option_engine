"""
Runs the full pricing engine and captures real benchmark numbers for README.
"""
import sys, time, statistics
sys.path.insert(0, '.')
import options_py as op
import numpy as np

S, K, r, T, sigma, q = 100.0, 100.0, 0.05, 1.0, 0.20, 0.0

# â”€â”€ Helper â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def bench(fn, reps=200):
    # warm-up
    for _ in range(10): fn()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e6)  # microseconds
    return statistics.median(times)

# â”€â”€ 1. Black-Scholes pricing â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
bs_call_us = bench(lambda: op.bs_call(S, K, r, T, sigma, q))
bs_put_us  = bench(lambda: op.bs_put (S, K, r, T, sigma, q))
call_price = op.bs_call(S, K, r, T, sigma, q)
put_price  = op.bs_put (S, K, r, T, sigma, q)

print("=== Black-Scholes Pricing ===")
print(f"  Call price : {call_price:.6f}")
print(f"  Put  price : {put_price:.6f}")
print(f"  bs_call latency : {bs_call_us:.3f} Âµs")
print(f"  bs_put  latency : {bs_put_us:.3f} Âµs")

# â”€â”€ 2. Greeks â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
g = op.analytical_greeks(S, K, r, T, sigma, q, op.CALL)
g_us = bench(lambda: op.analytical_greeks(S, K, r, T, sigma, q, op.CALL))
gn = op.numerical_greeks(S, K, r, T, sigma, q, op.CALL)
gn_us = bench(lambda: op.numerical_greeks(S, K, r, T, sigma, q, op.CALL))

print("\n=== Greeks (ATM Call) ===")
print(f"  {'Greek':<8} {'Analytical':>12} {'Numerical':>12} {'Abs error':>12}")
print(f"  {'-'*48}")
for name, av, nv in [('delta', g.delta, gn.delta), ('gamma', g.gamma, gn.gamma),
                     ('vega',  g.vega,  gn.vega),  ('theta', g.theta, gn.theta),
                     ('rho',   g.rho,   gn.rho)]:
    print(f"  {name:<8} {av:>12.8f} {nv:>12.8f} {abs(av-nv):>12.2e}")
print(f"  Analytical Greeks latency : {g_us:.3f} Âµs")
print(f"  Numerical  Greeks latency : {gn_us:.3f} Âµs")

# â”€â”€ 3. Implied vol â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
mkt = op.bs_call(S, K, r, T, sigma, q)
iv  = op.implied_vol(mkt, S, K, r, T, q, op.CALL)
iv_us = bench(lambda: op.implied_vol(mkt, S, K, r, T, q, op.CALL))

print("\n=== Implied Volatility ===")
print(f"  Market price  : {mkt:.6f}")
print(f"  Recovered Ïƒ   : {iv.implied_vol:.8f}  (true = {sigma})")
print(f"  |error|       : {abs(iv.implied_vol - sigma):.2e}")
print(f"  Iterations    : {iv.iterations}")
print(f"  Latency       : {iv_us:.3f} Âµs")

# â”€â”€ 4. Monte Carlo â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n=== Monte Carlo (European Call, S=K=100, r=5%, T=1, Ïƒ=20%) ===")
print(f"  {'Method':<30} {'Price':>10} {'Std Err':>10} {'|vs B-S|':>10} {'ms':>8} {'paths/s':>12}")
print(f"  {'-'*86}")

bs_ref = op.bs_call(S, K, r, T, sigma, q)

configs = [
    ('Standard MC  100k',   {'num_paths': 100_000,   'antithetic': False, 'use_sobol': False}),
    ('Antithetic   100k',   {'num_paths': 100_000,   'antithetic': True,  'use_sobol': False}),
    ('Sobol+anti   100k',   {'num_paths': 100_000,   'antithetic': True,  'use_sobol': True }),
    ('Antithetic     1M',   {'num_paths': 1_000_000, 'antithetic': True,  'use_sobol': False}),
    ('Antithetic    10M',   {'num_paths': 10_000_000,'antithetic': True,  'use_sobol': False}),
]

mc_results = {}
for label, kwargs in configs:
    cfg = op.MCConfig()
    cfg.num_paths  = kwargs['num_paths']
    cfg.antithetic = kwargs['antithetic']
    cfg.use_sobol  = kwargs['use_sobol']
    eng = op.MonteCarloEngine(cfg)
    res = eng.price_european(S, K, r, T, sigma, q, op.CALL)
    paths_per_s = res.paths_used / (res.elapsed_ms / 1000)
    mc_results[label] = res
    print(f"  {label:<30} {res.price:>10.6f} {res.std_error:>10.6f} "
          f"{abs(res.price-bs_ref):>10.6f} {res.elapsed_ms:>8.1f} {paths_per_s:>12,.0f}")

# â”€â”€ 5. Asian option â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cfg_asian = op.MCConfig()
cfg_asian.num_paths = 100_000
cfg_asian.num_steps = 252
cfg_asian.antithetic = False
eng_asian = op.MonteCarloEngine(cfg_asian)
asian = eng_asian.price_asian(S, K, r, T, sigma, q, op.CALL)

print(f"\n=== Asian Option (arithmetic avg, 252 steps, 100k paths) ===")
print(f"  Price      : {asian.price:.6f}")
print(f"  Std error  : {asian.std_error:.6f}")
print(f"  Elapsed    : {asian.elapsed_ms:.1f} ms")

# â”€â”€ 6. Barrier option â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cfg_barrier = op.MCConfig()
cfg_barrier.num_paths = 100_000
cfg_barrier.num_steps = 252
eng_barrier = op.MonteCarloEngine(cfg_barrier)
barrier = eng_barrier.price_barrier(S, K, 80.0, r, T, sigma, q)

print(f"\n=== Down-and-Out Barrier Call (B=80, 252 steps, 100k paths) ===")
print(f"  Price      : {barrier.price:.6f}")
print(f"  Std error  : {barrier.std_error:.6f}")
print(f"  Elapsed    : {barrier.elapsed_ms:.1f} ms")

# â”€â”€ 7. CRR binomial tree â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n=== CRR Binomial Tree â€” European Call Convergence ===")
print(f"  {'Steps':>6} {'Tree price':>12} {'B-S price':>12} {'|error|':>10} {'ms':>8}")
print(f"  {'-'*52}")

import time as _time
for n in [10, 50, 100, 200, 500, 1000]:
    t0 = _time.perf_counter()
    tree = op.crr_tree(S, K, r, T, sigma, q, op.CALL, op.EUROPEAN, n)
    ms = (_time.perf_counter() - t0) * 1000
    print(f"  {n:>6} {tree.price:>12.7f} {bs_ref:>12.7f} {abs(tree.price-bs_ref):>10.2e} {ms:>8.3f}")

# â”€â”€ 8. American put â€” early exercise premium â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n=== American Put vs European Put (CRR 500 steps) ===")
print(f"  {'K':>6} {'Euro put':>10} {'Amer put':>10} {'Premium':>10}")
print(f"  {'-'*42}")
for Kv in [90., 95., 100., 105., 110.]:
    euro = op.crr_tree(S, Kv, r, T, sigma, q, op.PUT, op.EUROPEAN, 500)
    amer = op.crr_tree(S, Kv, r, T, sigma, q, op.PUT, op.AMERICAN, 500)
    print(f"  {Kv:>6.0f} {euro.price:>10.6f} {amer.price:>10.6f} {amer.price-euro.price:>10.6f}")

# â”€â”€ 9. Vol surface â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
expiries = [0.25, 0.5, 1.0, 2.0]
strikes  = [80., 90., 95., 100., 105., 110., 120.]

def smile_vol(Kv, Tv):
    return max(0.20 - 0.10*(Kv/S - 1.0)/Tv**0.5, 0.01)

market_prices = [[op.bs_call(S, Kv, r, Tv, smile_vol(Kv, Tv), q) for Kv in strikes]
                 for Tv in expiries]

surf = op.build_vol_surface(S, r, q, expiries, strikes, market_prices, op.CALL)

print("\n=== Volatility Surface (implied vols, %) ===")
print(f"  {'':>6}", ''.join(f"{k:>8.0f}" for k in strikes))
for i, Tv in enumerate(expiries):
    row = ''.join(f"{surf.vols[i][j]*100:>8.2f}" for j in range(len(strikes)))
    print(f"  T={Tv:<5}{row}")

iv_interp = surf.interpolate(97.5, 0.75)
print(f"  Interpolated IV at K=97.5, T=0.75y : {iv_interp*100:.4f}%")

# â”€â”€ 10. Accuracy table â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n=== Accuracy Table â€” B-S vs CRR vs MC (500k antithetic) ===")
print(f"  {'Config':<35} {'B-S':>10} {'CRR-500':>10} {'MC-500k':>10} {'MC se':>8}")
print(f"  {'-'*75}")

cfg500k = op.MCConfig(); cfg500k.num_paths = 500_000; cfg500k.antithetic = True
eng500k = op.MonteCarloEngine(cfg500k)

cases = [
    ("S=100 K=100 T=1 Ïƒ=20% r=5%",   100, 100, 0.05, 1.0, 0.20, 0.0),
    ("S=100 K=110 T=1 Ïƒ=25% r=5%",   100, 110, 0.05, 1.0, 0.25, 0.0),
    ("S=100 K=90  T=0.5 Ïƒ=15% r=3%", 100,  90, 0.03, 0.5, 0.15, 0.0),
    ("S=100 K=100 T=2 Ïƒ=30% r=5% q=2%", 100, 100, 0.05, 2.0, 0.30, 0.02),
]
for label, s, k, rv, tv, sv, qv in cases:
    bs  = op.bs_call(s, k, rv, tv, sv, qv)
    crt = op.crr_tree(s, k, rv, tv, sv, qv, op.CALL, op.EUROPEAN, 500)
    mc  = eng500k.price_european(s, k, rv, tv, sv, qv, op.CALL)
    print(f"  {label:<35} {bs:>10.4f} {crt.price:>10.4f} {mc.price:>10.4f} {mc.std_error:>8.4f}")

print("\n=== ALL BENCHMARKS COMPLETE ===")

