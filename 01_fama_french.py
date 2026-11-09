"""
Project 1: Fama-French three-factor analysis of the Magnificent Seven (2020-2024)
Run:   python 01_fama_french.py           (real data: needs internet)
Test:  python 01_fama_french.py --demo    (synthetic data, only checks the code runs)

Needs: pip install pandas numpy scipy statsmodels matplotlib yfinance requests
Outputs go to ./output/ff_*  -- the numbers in ff_summary.txt are the ones to put on your CV.
"""
import argparse, io, zipfile
import numpy as np, pandas as pd, statsmodels.api as sm
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
START, END = "2020-01-01", "2024-12-31"
FF_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
          "F-F_Research_Data_Factors_daily_CSV.zip")
SEED, N_SIMS, HORIZON = 42, 10_000, 252          # Monte Carlo: 10,000 paths, 1 year
MAX_WEIGHT = 0.40                                # cap per stock in the optimiser


def load_ff():
    import requests
    r = requests.get(FF_URL, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    lines = z.read(z.namelist()[0]).decode("latin-1").splitlines()
    first = next(i for i, l in enumerate(lines) if "Mkt-RF" in l)
    rows = []
    for l in lines[first + 1:]:
        p = [x.strip() for x in l.split(",")]
        if len(p) >= 5 and p[0].isdigit() and len(p[0]) == 8:
            rows.append(p[:5])
        elif rows:
            break
    df = pd.DataFrame(rows, columns=["date", "Mkt-RF", "SMB", "HML", "RF"])
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    return (df.set_index("date").astype(float) / 100).loc[START:END]


def load_prices():
    import yfinance as yf
    px = yf.download(TICKERS, start=START, end="2025-01-01", auto_adjust=True, progress=False)["Close"]
    return px[TICKERS].dropna()


def demo_data():
    rng = np.random.default_rng(SEED)
    idx = pd.bdate_range(START, END)
    ff = pd.DataFrame({"Mkt-RF": rng.normal(0.0005, 0.012, len(idx)),
                       "SMB": rng.normal(0, 0.006, len(idx)),
                       "HML": rng.normal(0, 0.007, len(idx)),
                       "RF": 0.00008}, index=idx)
    rets = {}
    for t in TICKERS:
        b = rng.normal([1.2, 0.0, -0.3], [0.3, 0.3, 0.3])
        rets[t] = ff["RF"] + ff[["Mkt-RF", "SMB", "HML"]].values @ b + rng.normal(0, 0.012, len(idx))
    px = (1 + pd.DataFrame(rets, index=idx)).cumprod() * 100
    return px, ff


def regress(rets, ff):
    out = []
    for t in rets.columns:
        d = pd.concat([rets[t] - ff["RF"], ff[["Mkt-RF", "SMB", "HML"]]], axis=1, join="inner").dropna()
        y, X = d.iloc[:, 0], sm.add_constant(d[["Mkt-RF", "SMB", "HML"]])
        m = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
        out.append({"ticker": t, "alpha_annual": m.params["const"] * 252,
                    "beta_mkt": m.params["Mkt-RF"], "beta_smb": m.params["SMB"], "beta_hml": m.params["HML"],
                    "t_mkt": m.tvalues["Mkt-RF"], "t_smb": m.tvalues["SMB"], "t_hml": m.tvalues["HML"],
                    "r_squared": m.rsquared, "n_obs": int(m.nobs)})
    return pd.DataFrame(out).set_index("ticker")


def port_stats(w, mu, cov, rf):
    r, v = w @ mu * 252, np.sqrt(w @ cov @ w * 252)
    return r, v, (r - rf * 252) / v


def optimise(mu, cov, rf):
    n = len(mu)
    res = minimize(lambda w: -port_stats(w, mu, cov, rf)[2], np.repeat(1 / n, n), method="SLSQP",
                   bounds=[(0, MAX_WEIGHT)] * n, constraints={"type": "eq", "fun": lambda w: w.sum() - 1})
    return res.x


def var_es(x, level):
    q = np.quantile(x, 1 - level)
    return -q, -x[x <= q].mean()


def main(demo):
    if demo:
        px, ff = demo_data()
    else:
        px, ff = load_prices(), load_ff()
    rets = px.pct_change().dropna()
    rets = rets.loc[rets.index.intersection(ff.index)]
    reg = regress(rets, ff)

    rf = ff["RF"].reindex(rets.index).mean()
    mu, cov = rets.mean().values, rets.cov().values
    w_eq = np.repeat(1 / len(TICKERS), len(TICKERS))
    w_opt = optimise(mu, cov, rf)
    rows = {}
    for name, w in [("Equal weight", w_eq), ("Max Sharpe (cap 40%)", w_opt)]:
        pr = rets.values @ w
        r, v, s = port_stats(w, mu, cov, rf)
        dd = ((1 + pd.Series(pr, index=rets.index)).cumprod() / (1 + pd.Series(pr, index=rets.index)).cumprod().cummax() - 1).min()
        v95, e95 = var_es(pr, 0.95)
        v99, e99 = var_es(pr, 0.99)
        rows[name] = {"ann_return": r, "ann_vol": v, "sharpe": s, "max_drawdown": dd,
                      "hist_VaR95_1d": v95, "hist_ES95_1d": e95, "hist_VaR99_1d": v99, "hist_ES99_1d": e99,
                      "param_VaR95_1d": -(pr.mean() - 1.645 * pr.std(ddof=1))}
    pstats = pd.DataFrame(rows)

    rng = np.random.default_rng(SEED)
    sims = rng.multivariate_normal(mu, cov, size=(N_SIMS, HORIZON))
    term = {}
    for name, w in [("Equal weight", w_eq), ("Max Sharpe (cap 40%)", w_opt)]:
        path = (1 + sims @ w)
        term[name] = np.prod(path, axis=1) - 1
    mc = {k: {"mean_1y": v.mean(), "VaR95_1y": var_es(v, 0.95)[0], "ES95_1y": var_es(v, 0.95)[1],
              "prob_loss": (v < 0).mean()} for k, v in term.items()}
    mc = pd.DataFrame(mc)

    tag = "DEMO_" if demo else ""
    reg.round(4).to_csv(f"output/{tag}ff_regression.csv")
    pstats.round(4).to_csv(f"output/{tag}ff_portfolio_risk.csv")
    mc.round(4).to_csv(f"output/{tag}ff_monte_carlo.csv")
    pd.Series(w_opt, index=TICKERS).round(4).to_csv(f"output/{tag}ff_opt_weights.csv")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    reg[["beta_mkt", "beta_smb", "beta_hml"]].plot.bar(ax=ax[0])
    ax[0].set_title("Fama-French factor betas (daily, 2020-2024)"); ax[0].axhline(0, color="k", lw=.6)
    for name, v in term.items():
        ax[1].hist(v, bins=60, alpha=.55, label=name)
    ax[1].axvline(-mc.loc["VaR95_1y", "Equal weight"], color="r", ls="--", label="95% VaR (equal weight)")
    ax[1].set_title("Monte Carlo: 1-year portfolio return"); ax[1].legend()
    plt.tight_layout(); plt.savefig(f"output/{tag}ff_charts.png", dpi=150)

    with open(f"output/{tag}ff_summary.txt", "w") as f:
        f.write(("DEMO RUN - SYNTHETIC DATA - DO NOT USE THESE NUMBERS ANYWHERE\n\n" if demo else "") +
                "REGRESSIONS (HAC standard errors)\n" + reg.round(3).to_string() +
                "\n\nPORTFOLIO RISK (daily)\n" + pstats.round(4).to_string() +
                "\n\nMONTE CARLO (1 year, %d paths)\n" % N_SIMS + mc.round(4).to_string() +
                "\n\nOPTIMAL WEIGHTS\n" + pd.Series(w_opt, index=TICKERS).round(3).to_string() + "\n")
    print(open(f"output/{tag}ff_summary.txt").read())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--demo", action="store_true")
    main(ap.parse_args().demo)
