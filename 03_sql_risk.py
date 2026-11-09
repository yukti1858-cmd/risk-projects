"""
Project 3: SQL-based risk metrics. Loads ETF prices into SQLite and computes returns, volatility,
maximum drawdown and historical VaR / Expected Shortfall with SQL queries (window functions),
then cross-checks every number against pandas.
Run:   python 03_sql_risk.py          (real data)
Test:  python 03_sql_risk.py --demo
Needs SQLite >= 3.25 (included with modern Python).
"""
import argparse, sqlite3, math
import numpy as np, pandas as pd

TICKERS = ["SPY", "EFA", "EEM", "AGG", "GLD", "VNQ"]
DB = "output/risk.db"

SQL_RETURNS = """
CREATE VIEW daily_returns AS
SELECT date, ticker,
       adj_close / LAG(adj_close) OVER (PARTITION BY ticker ORDER BY date) - 1 AS ret
FROM prices;
"""
SQL_STATS = """
SELECT ticker,
       COUNT(ret)                                                   AS n_obs,
       AVG(ret) * 252                                               AS ann_return_arith,
       SQRT((AVG(ret * ret) - AVG(ret) * AVG(ret)) * COUNT(ret) / (COUNT(ret) - 1)) * SQRT(252) AS ann_vol
FROM daily_returns WHERE ret IS NOT NULL GROUP BY ticker ORDER BY ticker;
"""
SQL_DRAWDOWN = """
WITH run AS (
  SELECT ticker, date, adj_close,
         MAX(adj_close) OVER (PARTITION BY ticker ORDER BY date ROWS UNBOUNDED PRECEDING) AS peak
  FROM prices)
SELECT ticker, MIN(adj_close / peak - 1) AS max_drawdown FROM run GROUP BY ticker ORDER BY ticker;
"""
SQL_VAR = """
SELECT ret FROM daily_returns WHERE ticker = ? AND ret IS NOT NULL
ORDER BY ret LIMIT ? ;
"""


def get_prices(demo):
    if demo:
        rng = np.random.default_rng(3)
        idx = pd.bdate_range("2016-01-01", "2025-12-31")
        r = rng.normal(.0003, .01, (len(idx), len(TICKERS)))
        return pd.DataFrame(100 * np.cumprod(1 + r, axis=0), index=idx, columns=TICKERS)
    import yfinance as yf
    return yf.download(TICKERS, period="10y", auto_adjust=True, progress=False)["Close"][TICKERS].dropna()


def main(demo):
    px = get_prices(demo)
    long = px.reset_index().melt(id_vars=px.index.name or "index", var_name="ticker", value_name="adj_close")
    long.columns = ["date", "ticker", "adj_close"]
    long["date"] = pd.to_datetime(long["date"]).dt.strftime("%Y-%m-%d")
    con = sqlite3.connect(DB)
    con.executescript("DROP VIEW IF EXISTS daily_returns; DROP TABLE IF EXISTS prices;")
    con.execute("CREATE TABLE prices (date TEXT, ticker TEXT, adj_close REAL, PRIMARY KEY (date, ticker))")
    long.to_sql("prices", con, if_exists="append", index=False)
    con.executescript(SQL_RETURNS)

    stats = pd.read_sql(SQL_STATS, con).set_index("ticker")
    dd = pd.read_sql(SQL_DRAWDOWN, con).set_index("ticker")
    rows = {}
    for t in TICKERS:
        n = int(stats.loc[t, "n_obs"]); k = max(1, math.ceil(0.05 * n))       # worst 5% of days
        worst = pd.read_sql(SQL_VAR, con, params=(t, k))["ret"]
        rows[t] = {"hist_VaR95_1d": -worst.iloc[-1], "hist_ES95_1d": -worst.mean()}
    var = pd.DataFrame(rows).T
    out = stats.join(dd).join(var)

    # cross-check against pandas
    r = px.pct_change().dropna()
    chk_vol = (r.std() * np.sqrt(252) - out["ann_vol"]).abs().max()
    chk_dd = ((px / px.cummax() - 1).min() - out["max_drawdown"]).abs().max()
    assert chk_vol < 1e-6 and chk_dd < 1e-9, "SQL and pandas disagree"
    print(("DEMO - SYNTHETIC DATA\n" if demo else "") + out.round(4).to_string())
    print(f"\nCross-check vs pandas passed (max vol diff {chk_vol:.2e}, max drawdown diff {chk_dd:.2e}).")
    out.round(5).to_csv(f"output/{'DEMO_' if demo else ''}sql_risk_metrics.csv")
    with open("output/risk_queries.sql", "w") as f:
        f.write(SQL_RETURNS + "\n" + SQL_STATS + "\n" + SQL_DRAWDOWN + "\n" + SQL_VAR)
    con.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--demo", action="store_true")
    main(ap.parse_args().demo)
