"""
Visualisation helpers for the Options Pricing & Risk Engine.
Requires: numpy, matplotlib, scipy (optional).
The C++ extension 'options_py' must be built and on sys.path.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

try:
    import options_py as op
    _HAS_CPP = True
except ImportError:
    _HAS_CPP = False
    print("WARNING: options_py C++ extension not found — using fallback pure-Python pricer.")
    from scipy.stats import norm  # type: ignore

    class _FallbackOp:
        CALL = "CALL"
        PUT  = "PUT"

        @staticmethod
        def bs_call(S, K, r, T, sigma, q=0.0):
            if T <= 0: return max(S - K, 0.0)
            sqT = T**0.5
            d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*sqT)
            d2 = d1 - sigma*sqT
            return S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

        @staticmethod
        def bs_put(S, K, r, T, sigma, q=0.0):
            if T <= 0: return max(K - S, 0.0)
            sqT = T**0.5
            d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*sqT)
            d2 = d1 - sigma*sqT
            return K*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp(-q*T)*norm.cdf(-d1)

    op = _FallbackOp()


# ── Greeks vs. spot ────────────────────────────────────────────────────────────
def plot_greeks(K: float = 100.0, r: float = 0.05, T: float = 1.0,
                sigma: float = 0.20, q: float = 0.0,
                spot_range: tuple[float, float] = (60.0, 140.0),
                n_points: int = 200,
                option_type=None) -> plt.Figure:
    """Plot delta, gamma, vega, and theta for a European call across spot prices."""
    if option_type is None:
        option_type = op.CALL if _HAS_CPP else "CALL"

    spots = np.linspace(spot_range[0], spot_range[1], n_points)

    deltas = np.zeros(n_points)
    gammas = np.zeros(n_points)
    vegas  = np.zeros(n_points)
    thetas = np.zeros(n_points)

    for i, S in enumerate(spots):
        if _HAS_CPP:
            g = op.analytical_greeks(S, K, r, T, sigma, q, option_type)
            deltas[i] = g.delta
            gammas[i] = g.gamma
            vegas[i]  = g.vega
            thetas[i] = g.theta
        else:
            # Numerical approximation
            eps = S * 0.001
            fn  = op.bs_call if str(option_type) == "CALL" else op.bs_put
            deltas[i] = (fn(S+eps,K,r,T,sigma,q) - fn(S-eps,K,r,T,sigma,q))/(2*eps)
            gammas[i] = (fn(S+eps,K,r,T,sigma,q)-2*fn(S,K,r,T,sigma,q)+fn(S-eps,K,r,T,sigma,q))/(eps**2)
            dvol = 0.001
            vegas[i]  = (fn(S,K,r,T,sigma+dvol,q)-fn(S,K,r,T,sigma-dvol,q))/(2*dvol*100)
            dt   = 1.0/365
            thetas[i] = (fn(S,K,r,T-dt,sigma,q)-fn(S,K,r,T,sigma,q))/1.0

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Greeks — K={K}, T={T}y, σ={sigma:.0%}, r={r:.1%}", fontsize=14)

    pairs = [("Delta Δ", deltas, axes[0,0]),
             ("Gamma Γ", gammas, axes[0,1]),
             ("Vega ν (per 1 vol pt)", vegas, axes[1,0]),
             ("Theta θ (per day)", thetas, axes[1,1])]

    for label, y, ax in pairs:
        ax.plot(spots, y, lw=2)
        ax.axvline(K, color="red", linestyle="--", alpha=0.5, label="Strike")
        ax.set_xlabel("Spot price")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


# ── Volatility surface ─────────────────────────────────────────────────────────
def plot_vol_surface(S: float = 100.0, r: float = 0.05, q: float = 0.0,
                     base_vol: float = 0.20,
                     skew: float = -0.10,
                     expiries: list[float] | None = None,
                     strikes:  list[float] | None = None) -> plt.Figure:
    """
    Plot a parameterised implied-vol surface.
    Vol(K, T) = base_vol + skew * (K/S - 1) / sqrt(T)  [simple SABR-like smile]
    """
    if expiries is None:
        expiries = [1/12, 3/12, 6/12, 1.0, 2.0]
    if strikes is None:
        strikes = np.linspace(0.7*S, 1.3*S, 20).tolist()

    T_arr = np.array(expiries)
    K_arr = np.array(strikes)
    KK, TT = np.meshgrid(K_arr, T_arr)

    # Parameterised smile: ATM vol + skew component decaying with sqrt(T)
    moneyness = (KK / S - 1.0)
    VOL = base_vol + skew * moneyness / np.sqrt(TT)
    VOL = np.clip(VOL, 0.01, 2.0)

    fig = plt.figure(figsize=(12, 6))
    gs  = gridspec.GridSpec(1, 2, figure=fig)

    # 3-D surface
    ax3d = fig.add_subplot(gs[0], projection='3d')
    ax3d.plot_surface(KK, TT, VOL * 100, cmap='RdYlGn_r', alpha=0.9)
    ax3d.set_xlabel("Strike K")
    ax3d.set_ylabel("Expiry T (y)")
    ax3d.set_zlabel("Implied Vol (%)")
    ax3d.set_title("Implied Volatility Surface")

    # Smile slices
    ax2d = fig.add_subplot(gs[1])
    colors = plt.cm.viridis(np.linspace(0, 1, len(expiries)))
    for i, T in enumerate(expiries):
        smile = base_vol + skew * (K_arr / S - 1.0) / np.sqrt(T)
        smile = np.clip(smile, 0.01, 2.0)
        ax2d.plot(K_arr, smile * 100, color=colors[i],
                  label=f"T={T:.2f}y", lw=1.5)
    ax2d.axvline(S, color="black", linestyle="--", alpha=0.4, label="ATM")
    ax2d.set_xlabel("Strike K")
    ax2d.set_ylabel("Implied Vol (%)")
    ax2d.set_title("Vol Smile by Expiry")
    ax2d.legend(fontsize=8)
    ax2d.grid(True, alpha=0.3)

    fig.suptitle(f"Volatility Surface — S={S}, base σ={base_vol:.0%}, skew={skew:.2f}", fontsize=12)
    fig.tight_layout()
    return fig


# ── P&L scenario analysis (bump-and-reprice) ─────────────────────────────────
def plot_pnl_scenarios(S: float = 100.0, K: float = 100.0,
                       r: float = 0.05, T: float = 1.0,
                       sigma: float = 0.20, q: float = 0.0,
                       spot_shocks: np.ndarray | None = None,
                       vol_shocks:  np.ndarray | None = None) -> plt.Figure:
    """
    2-D heatmap of P&L for a long call under joint spot and vol moves.
    Also plots the 1-D P&L at expiry (intrinsic payoff diagram).
    """
    if spot_shocks is None:
        spot_shocks = np.linspace(-0.25, 0.25, 11)  # -25% to +25%
    if vol_shocks is None:
        vol_shocks  = np.linspace(-0.10, 0.10, 11)  # -10 to +10 vol pts

    fn = (op.bs_call if (not _HAS_CPP or str(op.CALL) == "CALL") else op.bs_call)
    base_price = fn(S, K, r, T, sigma, q)

    pnl = np.zeros((len(vol_shocks), len(spot_shocks)))
    for i, dv in enumerate(vol_shocks):
        for j, ds in enumerate(spot_shocks):
            new_price = fn(S*(1+ds), K, r, T, sigma+dv, q)
            pnl[i, j] = new_price - base_price

    fig, (ax_heat, ax_payoff) = plt.subplots(1, 2, figsize=(14, 5))

    # Heat-map
    im = ax_heat.imshow(pnl, cmap='RdYlGn', aspect='auto',
                        extent=[spot_shocks[0]*100, spot_shocks[-1]*100,
                                vol_shocks[-1]*100,  vol_shocks[0]*100])
    plt.colorbar(im, ax=ax_heat, label="P&L (currency)")
    ax_heat.set_xlabel("Spot shock (%)")
    ax_heat.set_ylabel("Vol shock (vol pts %)")
    ax_heat.set_title("P&L Scenario Grid — Long European Call")
    ax_heat.axvline(0, color='white', lw=0.8, alpha=0.6)
    ax_heat.axhline(0, color='white', lw=0.8, alpha=0.6)

    # Payoff at expiry
    spot_range = np.linspace(S * 0.5, S * 1.5, 300)
    payoff     = np.maximum(spot_range - K, 0.0) - base_price
    ax_payoff.plot(spot_range, payoff, lw=2, label="P&L at expiry")
    ax_payoff.axhline(0, color='black', lw=0.8, alpha=0.5)
    ax_payoff.axvline(K, color='red', linestyle='--', alpha=0.6, label=f"Strike K={K}")
    ax_payoff.fill_between(spot_range, payoff, 0, where=payoff > 0, alpha=0.2, color='green')
    ax_payoff.fill_between(spot_range, payoff, 0, where=payoff < 0, alpha=0.2, color='red')
    ax_payoff.set_xlabel("Spot at expiry")
    ax_payoff.set_ylabel("P&L")
    ax_payoff.set_title("P&L at Expiry (intrinsic)")
    ax_payoff.legend()
    ax_payoff.grid(True, alpha=0.3)

    fig.suptitle(f"P&L Analysis — Call K={K}, T={T}y, σ={sigma:.0%}", fontsize=12)
    fig.tight_layout()
    return fig


# ── MC convergence plot ───────────────────────────────────────────────────────
def plot_mc_convergence(S: float = 100.0, K: float = 100.0,
                        r: float = 0.05, T: float = 1.0,
                        sigma: float = 0.20, q: float = 0.0) -> plt.Figure:
    """Show MC price converging to B-S as num_paths increases (log scale)."""
    bs_price = op.bs_call(S, K, r, T, sigma, q) if _HAS_CPP else None

    if not _HAS_CPP:
        print("C++ extension required for MC convergence plot.")
        return plt.figure()

    path_counts = [100, 500, 1000, 5000, 10_000, 50_000, 100_000, 500_000, 1_000_000]
    mc_prices   = []
    mc_se       = []

    for n in path_counts:
        cfg = op.MCConfig()
        cfg.num_paths  = n
        cfg.antithetic = True
        eng = op.MonteCarloEngine(cfg)
        res = eng.price_european(S, K, r, T, sigma, q, op.CALL)
        mc_prices.append(res.price)
        mc_se.append(res.std_error)

    prices = np.array(mc_prices)
    errors = np.array(mc_se)
    ns     = np.array(path_counts)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.semilogx(ns, prices, 'o-', lw=2, label="MC (antithetic)")
    ax1.axhline(bs_price, color='red', linestyle='--', lw=1.5, label=f"B-S = {bs_price:.4f}")
    ax1.fill_between(ns, prices-1.96*errors, prices+1.96*errors, alpha=0.2, label="95% CI")
    ax1.set_xlabel("Number of paths")
    ax1.set_ylabel("Option price")
    ax1.set_title("MC Convergence vs. Black-Scholes")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.loglog(ns, errors, 'o-', lw=2, label="Std error")
    ax2.loglog(ns, errors[0] * np.sqrt(path_counts[0] / ns), '--',
               color='grey', alpha=0.7, label="O(1/√N)")
    ax2.set_xlabel("Number of paths")
    ax2.set_ylabel("Standard error")
    ax2.set_title("MC Standard Error Convergence")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    fig.suptitle(f"Monte Carlo Convergence — ATM Call K={K}, T={T}y, σ={sigma:.0%}", fontsize=12)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    fig1 = plot_greeks()
    fig2 = plot_vol_surface()
    fig3 = plot_pnl_scenarios()
    if _HAS_CPP:
        fig4 = plot_mc_convergence()
    plt.show()
