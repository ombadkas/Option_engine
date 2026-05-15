"""Save all visualisation figures to ../images/ for README embedding."""
import sys, os
sys.path.insert(0, '.')

import matplotlib
matplotlib.use('Agg')   # non-interactive backend — no window needed
import matplotlib.pyplot as plt

from viz import plot_greeks, plot_vol_surface, plot_pnl_scenarios, plot_mc_convergence

OUT = os.path.join('..', 'images')
os.makedirs(OUT, exist_ok=True)

print("Saving Figure 1 — Greeks...")
fig1 = plot_greeks(K=100, r=0.05, T=1.0, sigma=0.20, q=0.0)
fig1.savefig(os.path.join(OUT, 'greeks.png'), dpi=150, bbox_inches='tight')
plt.close(fig1)

print("Saving Figure 2 — Volatility Surface...")
fig2 = plot_vol_surface(S=100, r=0.05, q=0.0, base_vol=0.20, skew=-0.10)
fig2.savefig(os.path.join(OUT, 'vol_surface.png'), dpi=150, bbox_inches='tight')
plt.close(fig2)

print("Saving Figure 3 — P&L Scenarios...")
fig3 = plot_pnl_scenarios(S=100, K=100, r=0.05, T=1.0, sigma=0.20)
fig3.savefig(os.path.join(OUT, 'pnl_scenarios.png'), dpi=150, bbox_inches='tight')
plt.close(fig3)

print("Saving Figure 4 — MC Convergence...")
fig4 = plot_mc_convergence(S=100, K=100, r=0.05, T=1.0, sigma=0.20)
fig4.savefig(os.path.join(OUT, 'mc_convergence.png'), dpi=150, bbox_inches='tight')
plt.close(fig4)

print(f"Done — images saved to {os.path.abspath(OUT)}")
