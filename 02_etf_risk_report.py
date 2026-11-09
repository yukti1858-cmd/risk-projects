"""
Project 2: Multi-asset ETF portfolio RISK REPORT (extends your existing ETF vs 60/40 backtest)
Adds: historical + parametric VaR and Expected Shortfall, risk contribution, stress tests, one-page dashboard.
Run:   python 02_etf_risk_report.py          (real data, needs internet)
Test:  python 02_etf_risk_report.py --demo   (synthetic data; code check only)

EDIT THE CONFIG BELOW so that the tickers and weights match YOUR existing project.
Assumption: constant weights, rebalanced daily (simple, state this on the CV/in interviews).
"""
import argparse
from datetime import date
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- CONFIG: edit to match your project ----------------
TICKERS = ["SPY", "EFA", "EEM", "AGG", "GLD", "VNQ"]   # US equity, DM ex-US, EM, bonds, gold, real estate
PORTFOLIOS = {
    "Growth":    dict(SPY=.35, EFA=.20, EEM=.15, AGG=.10, GLD=.10, VNQ=.10),
    "Balanced":  dict(SPY=.30, EFA=.10, EEM=.05, AGG=.35, GLD=.10, VNQ=.10),
    "Defensive": dict(SPY=.15, EFA=.05, EEM=.00, AGG=.60, GLD=.15, VNQ=.05),
}
BENCHMARK = dict(SPY=.60, AGG=.40)                       # 60/40
RISK_CONTRIB_PORTFOLIO = "Balanced"
RF_ANNUAL = 0.02
START = f"{date.today().year - 10}-01-01"
# Historical stress windows (start, end)
HIST_STRESS = {"COVID crash 2020": ("2020-02-19", "2020-03-23"),
               "Rate shock 2022": ("2022-01-03", "2022-10-12")}
# Hypothetical one-off shocks per ETF (ILLUSTRATIVE assumptions - state them as such)
HYP_STRESS = {
    "Equity crash":      dict(SPY=-.20, EFA=-.22, EEM=-.28, AGG=.02, GLD=.04, VNQ=-.25),
    "Rates +200bp":      dict(SPY=-.08, EFA=-.08, EEM=-.10, AGG=-.08, GLD=-.05, VNQ=-.15),
    "Stagflation":       dict(SPY=-.15, EFA=-.15, EEM=-.18, AGG=-.06, GLD=.12, VNQ=-.12),
}
# --------------------------------------------------------------------


def load_prices():
    import yfinance as yf
    return yf.download(TICKERS, start=START, auto_adjust=True, progress=False)["Close"][TICKERS].dropna()


def demo_prices():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2016-01-01", "2025-12-31")
    mu = np.array([.0004, .0003, .0003, .0001, .0002, .0003])
    vol = np.array([.011, .011, .013, .003, .009, .013])
    corr = np.full((6, 6), .5); np.fill_diagonal(corr, 1); corr[3, :3] = corr[:3, 3] = -.1
    r = rng.multivariate_normal(mu, corr * np.outer(vol, vol), size=len(idx))
    return pd.DataFrame(100 * np.cumprod(1 + r, axis=0), index=idx, columns=TICKERS)


def weights(d):
    return np.array([d.get(t, 0.0) for t in TICKERS])


def var_es(x, level):
    q = np.quantile(x, 1 - level)
    return -q, -x[x <= q].mean()


def metrics(r, bench):
    nav = (1 + r).cumprod()
    return {"Ann. return": (nav.iloc[-1]) ** (252 / len(r)) - 1,
            "Ann. volatility": r.std() * np.sqrt(252),
            "Sharpe": (r.mean() * 252 - RF_ANNUAL) / (r.std() * np.sqrt(252)),
            "Max drawdown": (nav / nav.cummax() - 1).min(),
            "Tracking error": (r - bench).std() * np.sqrt(252),
            "Hist VaR 95% (1d)": var_es(r.values, .95)[0], "Hist ES 95% (1d)": var_es(r.values, .95)[1],
            "Hist VaR 99% (1d)": var_es(r.values, .99)[0], "Hist ES 99% (1d)": var_es(r.values, .99)[1],
            "Param VaR 95% (1d)": -(r.mean() - 1.645 * r.std()),
            "Param VaR 99% (1d)": -(r.mean() - 2.326 * r.std())}


def main(demo):
    px = demo_prices() if demo else load_prices()
    rets = px.pct_change().dropna()
    bench = rets.values @ weights(BENCHMARK)
    bench = pd.Series(bench, index=rets.index)
    port = {n: pd.Series(rets.values @ weights(w), index=rets.index) for n, w in PORTFOLIOS.items()}
    port["60/40 benchmark"] = bench
    table = pd.DataFrame({n: metrics(r, bench) for n, r in port.items()})

    # risk contribution (share of portfolio variance)
    w = weights(PORTFOLIOS[RISK_CONTRIB_PORTFOLIO]); cov = rets.cov().values * 252
    mrc = cov @ w
    rc = pd.Series(w * mrc / (w @ cov @ w), index=TICKERS)
    rc_tab = pd.DataFrame({"weight": w, "risk_contribution": rc.values}, index=TICKERS)

    # stress tests
    stress = {}
    for name, (a, b) in HIST_STRESS.items():
        seg = rets.loc[a:b]
        if len(seg) > 5:
            stress[f"{name} (historical)"] = {n: (1 + r.loc[a:b]).prod() - 1 for n, r in port.items() if n != "60/40 benchmark"}
    for name, shock in HYP_STRESS.items():
        stress[f"{name} (hypothetical)"] = {n: float(weights(wd) @ weights(shock)) for n, wd in PORTFOLIOS.items()}
    stress = pd.DataFrame(stress).T

    tag = "DEMO_" if demo else ""
    table.round(4).to_csv(f"output/{tag}etf_risk_metrics.csv")
    rc_tab.round(4).to_csv(f"output/{tag}etf_risk_contribution.csv")
    stress.round(4).to_csv(f"output/{tag}etf_stress_tests.csv")

    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    for n, r in port.items():
        (1 + r).cumprod().plot(ax=ax[0, 0], label=n)
        ((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).plot(ax=ax[0, 1], label=n)
    ax[0, 0].set_title("Growth of 1"); ax[0, 0].legend(); ax[0, 1].set_title("Drawdown")
    v = table.loc[["Hist VaR 95% (1d)", "Hist ES 95% (1d)", "Param VaR 95% (1d)"]].T
    v.plot.bar(ax=ax[0, 2], rot=20); ax[0, 2].set_title("1-day 95% VaR / ES")
    rc.plot.bar(ax=ax[1, 0], rot=0); ax[1, 0].set_title(f"Risk contribution: {RISK_CONTRIB_PORTFOLIO}")
    stress.plot.bar(ax=ax[1, 1], rot=15); ax[1, 1].set_title("Stress tests (portfolio return)"); ax[1, 1].legend(fontsize=7)
    ax[1, 2].axis("off")
    t = table.loc[["Ann. return", "Ann. volatility", "Sharpe", "Max drawdown", "Tracking error"]].round(3)
    tb = ax[1, 2].table(cellText=t.values, rowLabels=t.index, colLabels=t.columns, loc="center"); tb.auto_set_font_size(False); tb.set_fontsize(8)
    fig.suptitle(("DEMO - SYNTHETIC DATA - " if demo else "") + "Multi-asset ETF risk dashboard", fontsize=14)
    plt.tight_layout(); plt.savefig(f"output/{tag}etf_risk_dashboard.png", dpi=150)

    print(("DEMO - SYNTHETIC DATA\n" if demo else "") + table.round(4).to_string())
    print("\nRISK CONTRIBUTION\n", rc_tab.round(3).to_string())
    print("\nSTRESS TESTS\n", stress.round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--demo", action="store_true")
    main(ap.parse_args().demo)
