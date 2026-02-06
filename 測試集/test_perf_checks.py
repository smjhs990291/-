import pandas as pd
from merge_trade_reports import _drawdown_episodes, _runup_episodes, _max_drawdown_from_pnl, build_report_tables

def test_drawdown_incomplete_episode():
    times = pd.date_range("2024-01-01", periods=5, freq="D")
    equity = pd.Series([1000, 1100, 1050, 1020, 1010], index=times)
    df = pd.DataFrame({"日期/時間": equity.index, "equity": equity.values, "淨損益 USD": [0, 100, -50, -30, -10]})
    episodes, max_dd_usd, max_dd_pct = _drawdown_episodes(df[["日期/時間", "equity"]])
    assert max_dd_usd > 0
    assert len(episodes) >= 1

def test_runup_pct_uses_trough_denominator():
    times = pd.date_range("2024-02-01", periods=4, freq="D")
    equity = pd.Series([1000, 900, 950, 970], index=times)
    df = pd.DataFrame({"日期/時間": equity.index, "equity": equity.values})
    episodes, max_ru_usd, max_ru_pct = _runup_episodes(df[["日期/時間", "equity"]])
    assert max_ru_usd == 70
    assert round(max_ru_pct, 2) == round(70 / 900 * 100, 2)

def test_max_drawdown_from_pnl_positive():
    pnl = pd.Series([100, -50, -100, 200])
    val = _max_drawdown_from_pnl(pnl)
    assert val > 0

def test_unrealized_pct_in_perf():
    merged = pd.DataFrame(
        {
            "來源檔案": ["a.xlsx"],
            "商品": ["SYM"],
            "時間週期": ["1天"],
            "點值": [1.0],
            "初始資本": [1000.0],
            "交易鍵": ["SYM#1"],
            "類型": ["做多 進場"],
            "日期/時間": [pd.Timestamp("2024-01-01")],
            "持倉大小(數量)": [1],
            "持倉大小(值)": [1000.0],
            "淨損益 USD": [100.0],
            "淨損益 %": [10.0],
        }
    )
    perf, analysis, risk, trade_list_out, attrs = build_report_tables(merged, None)
    row = perf.loc[perf["Unnamed: 0"] == "未實現盈虧"].iloc[0]
    assert round(float(row["全部 %"]), 2) == 10.0

if __name__ == "__main__":
    test_drawdown_incomplete_episode()
    test_runup_pct_uses_trough_denominator()
    test_max_drawdown_from_pnl_positive()
    test_unrealized_pct_in_perf()
    print("OK")
