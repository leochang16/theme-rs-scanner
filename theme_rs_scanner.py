#!/usr/bin/env python3
"""主題股票籃子 vs SPY 相對強弱與動能掃描器。

資料源:Yahoo Finance(yfinance)。所有數值皆來自實際抓取的原始資料,
不使用任何記憶/內建假設填補股價、報酬或產業歸屬。
"""

import argparse
import sys
from datetime import datetime

import pandas as pd

try:
    import yfinance as yf
except ImportError:
    sys.exit("缺少 yfinance,請先 pip install yfinance")

# rich 為可選;沒裝就 fallback 成純文字
try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
    HAVE_RICH = True
except ImportError:
    HAVE_RICH = False


# ---------------------------------------------------------------------------
# 主題籃子(寫死在頂部,方便日後修改)
# ---------------------------------------------------------------------------
BASKETS = {
    "光通訊": ["COHR", "LITE", "AAOI", "POET", "MRVL", "CRDO", "ALAB", "FN"],
    "衛星太空": ["RKLB", "ASTS", "GSAT", "IRDM", "PL", "RDW", "LUNR"],
    "AI電力散熱": ["VRT", "BE", "GEV", "CEG", "TLN", "SMCI"],
    # 記憶體/儲存:DRAM/NAND/HBM 與控制器(僅美股可抓者;三星/SK海力士/鎧俠非美股不列)
    "記憶體": ["MU", "WDC", "STX", "SNDK", "NLST", "SIMO"],
    # AI 軟體/應用:資料平台、可觀測性、資安、企業工作流
    "AI軟體": ["PLTR", "SNOW", "NOW", "MDB", "DDOG", "CRWD", "NET"],
    # 半導體設備(WFE):蝕刻/沉積/微影/製程檢測
    "半導體設備": ["AMAT", "LRCX", "KLAC", "ASML", "ACLS", "ONTO"],
    # GPU/AI 運算晶片:加速器與核心運算
    "GPU運算": ["NVDA", "AMD", "AVGO", "TSM", "ARM", "QCOM"],
}
BENCHMARK = "SPY"

# 報酬期間用的交易日天數
WIN_1W = 5
WIN_1M = 21
WIN_3M = 63
# 52 週與均線
WIN_52W = 252


# ---------------------------------------------------------------------------
# 計算工具
# ---------------------------------------------------------------------------
def pct_return(close: pd.Series, n: int):
    """close 序列回看 n 個交易日的報酬率(%)。資料不足回 None。"""
    if len(close) <= n:
        return None
    prev = close.iloc[-1 - n]
    if prev == 0 or pd.isna(prev):
        return None
    return (close.iloc[-1] / prev - 1.0) * 100.0


def compute_metrics(close: pd.Series, volume: pd.Series, rs_window: int,
                    spy_rs_return):
    """回傳單檔指標 dict。close/volume 為已去除 NaN 的序列。"""
    m = {}
    # 1. 當日漲跌幅
    m["day"] = pct_return(close, 1)
    # 2. 1週/1月/3月
    m["r1w"] = pct_return(close, WIN_1W)
    m["r1m"] = pct_return(close, WIN_1M)
    m["r3m"] = pct_return(close, WIN_3M)
    # 3. 相對強度 RS = 個股 N 日報酬 - SPY 同期報酬
    own = pct_return(close, rs_window)
    if own is None or spy_rs_return is None:
        m["rs"] = None
    else:
        m["rs"] = own - spy_rs_return
    # 4. 距 52 週高點
    lookback = close.iloc[-WIN_52W:] if len(close) >= WIN_52W else close
    hi = float(lookback.max())
    m["from_high"] = (close.iloc[-1] / hi - 1.0) * 100.0 if hi else None
    m["high_partial"] = len(close) < WIN_52W  # 資料不足 52 週的標記
    # 5. 現價(市值於主流程以股數 × 現價計算)
    m["price"] = float(close.iloc[-1])
    return m


# ---------------------------------------------------------------------------
# 抓取
# ---------------------------------------------------------------------------
def fetch_all(tickers):
    """批次抓 1 年日線。回傳 (data_dict, failed_list, last_date)。

    data_dict[ticker] = {"close": Series, "volume": Series}
    """
    data = {}
    failed = []
    try:
        raw = yf.download(
            tickers,
            # 抓 2 年:確保 52 週(252 交易日)與 200MA 都有足量資料,
            # 否則 ~250 交易日會讓老股也誤判為「未滿 52 週」。
            period="2y",
            interval="1d",
            auto_adjust=True,
            group_by="ticker",
            threads=True,
            progress=False,
        )
    except Exception as e:  # noqa: BLE001
        # 整批失敗:逐檔標記失敗,讓上層決定
        return {}, [(t, f"批次下載失敗: {e}") for t in tickers], None

    if raw is None or len(raw) == 0:
        return {}, [(t, "無回傳資料") for t in tickers], None

    single = len(tickers) == 1

    last_date = None
    for t in tickers:
        try:
            if single:
                sub = raw
            else:
                if t not in raw.columns.get_level_values(0):
                    failed.append((t, "回傳結果中無此 ticker"))
                    continue
                sub = raw[t]
            close = sub["Close"].dropna()
            volume = sub["Volume"].dropna()
            if len(close) < 2:
                failed.append((t, f"有效收盤資料不足(僅 {len(close)} 筆)"))
                continue
            data[t] = {"close": close, "volume": volume}
            d = close.index[-1]
            if last_date is None or d > last_date:
                last_date = d
        except Exception as e:  # noqa: BLE001
            failed.append((t, f"解析失敗: {e}"))

    return data, failed, last_date


def fetch_shares(tickers):
    """逐檔抓流通股數(fast_info['shares'])。市值 = 股數 × 收盤價。

    股數抓取失敗或缺值 → None(市值顯示 n/a),不視為整檔抓取失敗。
    """
    shares = {}
    for t in tickers:
        try:
            fi = yf.Ticker(t).fast_info
            sh = fi["shares"]
            shares[t] = float(sh) if sh else None
        except Exception:  # noqa: BLE001
            shares[t] = None
    return shares


# ---------------------------------------------------------------------------
# 顯示輔助
# ---------------------------------------------------------------------------
def fmt_pct(v, decimals=2):
    if v is None:
        return "n/a"
    return f"{v:+.{decimals}f}%"


def fmt_mcap(v):
    """市值人類可讀:T/B/M。"""
    if v is None:
        return "n/a"
    if v >= 1e12:
        return f"${v / 1e12:.2f}T"
    if v >= 1e9:
        return f"${v / 1e9:.1f}B"
    if v >= 1e6:
        return f"${v / 1e6:.0f}M"
    return f"${v:.0f}"


def color_for(v):
    """rich 顏色:正綠負紅。"""
    if v is None:
        return "dim"
    return "green" if v >= 0 else "red"


# ---------------------------------------------------------------------------
# rich 輸出
# ---------------------------------------------------------------------------
def render_rich(basket_rank, basket_details, fetch_ts, data_date, failed,
                rs_window, sort_label):
    # 互動終端機吃實際寬度;被導向 pipe/檔案時 rich 預設只給 80 欄會截斷數字,
    # 故非 TTY 時強制給足夠寬度。
    console = Console(width=None if sys.stdout.isatty() else 150)

    console.print(
        f"[bold]抓取時間戳[/bold]: {fetch_ts}    "
        f"[bold]資料日期(最後交易日)[/bold]: {data_date}    "
        f"[bold]RS 視窗[/bold]: {rs_window} 交易日"
    )
    console.print()

    # 籃子強弱排行
    rank_tbl = Table(title="籃子強弱排行(等權平均 RS,由強到弱)",
                     box=box.SIMPLE_HEAVY, header_style="bold cyan")
    rank_tbl.add_column("排名", justify="right")
    rank_tbl.add_column("籃子", style="bold")
    rank_tbl.add_column("平均 RS", justify="right")
    rank_tbl.add_column("成分股數", justify="right")
    for i, (name, avg_rs, n) in enumerate(basket_rank, 1):
        rs_txt = fmt_pct(avg_rs) if avg_rs is not None else "n/a"
        rank_tbl.add_row(str(i), name,
                         f"[{color_for(avg_rs)}]{rs_txt}[/]", str(n))
    console.print(rank_tbl)
    console.print()

    # 各籃子成分股明細
    for name, avg_rs, rows in basket_details:
        rs_hdr = fmt_pct(avg_rs) if avg_rs is not None else "n/a"
        title = (f"{name}  —  平均 RS [{color_for(avg_rs)}]{rs_hdr}[/]  "
                 f"(成分股{sort_label})")
        tbl = Table(title=title, box=box.SIMPLE_HEAVY,
                    header_style="bold cyan", title_justify="left")
        tbl.add_column("Ticker", style="bold")
        tbl.add_column("現價", justify="right")
        tbl.add_column("市值", justify="right")
        tbl.add_column("當日%", justify="right")
        tbl.add_column("1週%", justify="right")
        tbl.add_column("1月%", justify="right")
        tbl.add_column("3月%", justify="right")
        tbl.add_column(f"RS({rs_window}d)", justify="right")
        tbl.add_column("距52WH", justify="right")
        for t, m in rows:
            high_txt = fmt_pct(m["from_high"])
            if m.get("high_partial"):
                high_txt += "*"
            tbl.add_row(
                t,
                f"{m['price']:.2f}" if m["price"] is not None else "n/a",
                fmt_mcap(m["mcap"]),
                f"[{color_for(m['day'])}]{fmt_pct(m['day'])}[/]",
                f"[{color_for(m['r1w'])}]{fmt_pct(m['r1w'])}[/]",
                f"[{color_for(m['r1m'])}]{fmt_pct(m['r1m'])}[/]",
                f"[{color_for(m['r3m'])}]{fmt_pct(m['r3m'])}[/]",
                f"[{color_for(m['rs'])}]{fmt_pct(m['rs'])}[/]",
                f"[{color_for(m['from_high'])}]{high_txt}[/]",
            )
        console.print(tbl)
        console.print()

    if failed:
        console.print("[bold red]抓取失敗清單[/bold red]:")
        for t, reason in failed:
            console.print(f"  [red]✗[/red] {t}: {reason}")
    else:
        console.print("[green]所有 ticker 皆成功抓取。[/green]")
    if any(d for *_, rows in basket_details for _, d in rows if d.get("high_partial")):
        console.print("[dim]* 該檔上市未滿 52 週,距高點以可得資料的最高價計算。[/dim]")


# ---------------------------------------------------------------------------
# 純文字 fallback
# ---------------------------------------------------------------------------
def render_plain(basket_rank, basket_details, fetch_ts, data_date, failed,
                 rs_window, sort_label):
    print(f"抓取時間戳: {fetch_ts}    資料日期(最後交易日): {data_date}    "
          f"RS 視窗: {rs_window} 交易日")
    print()

    print("== 籃子強弱排行(等權平均 RS,由強到弱) ==")
    print(f"{'排名':<4}{'籃子':<14}{'平均RS':>10}{'成分股數':>8}")
    for i, (name, avg_rs, n) in enumerate(basket_rank, 1):
        rs_txt = fmt_pct(avg_rs) if avg_rs is not None else "n/a"
        print(f"{i:<4}{name:<14}{rs_txt:>10}{n:>8}")
    print()

    hdr = (f"{'Ticker':<7}{'現價':>9}{'市值':>9}{'當日%':>9}{'1週%':>9}{'1月%':>9}"
           f"{'3月%':>9}{'RS':>9}{'距52WH':>9}")
    for name, avg_rs, rows in basket_details:
        rs_hdr = fmt_pct(avg_rs) if avg_rs is not None else "n/a"
        print(f"== {name}  平均RS {rs_hdr}  (成分股{sort_label}) ==")
        print(hdr)
        for t, m in rows:
            high_txt = fmt_pct(m["from_high"]) + ("*" if m.get("high_partial") else "")
            price = f"{m['price']:.2f}" if m["price"] is not None else "n/a"
            print(f"{t:<7}{price:>9}{fmt_mcap(m['mcap']):>9}{fmt_pct(m['day']):>9}"
                  f"{fmt_pct(m['r1w']):>9}{fmt_pct(m['r1m']):>9}{fmt_pct(m['r3m']):>9}"
                  f"{fmt_pct(m['rs']):>9}{high_txt:>9}")
        print()

    if failed:
        print("== 抓取失敗清單 ==")
        for t, reason in failed:
            print(f"  ✗ {t}: {reason}")
    else:
        print("所有 ticker 皆成功抓取。")


# ---------------------------------------------------------------------------
# HTML 輸出(自包含單檔,內嵌 CSS,無外部相依、無 localStorage)
# ---------------------------------------------------------------------------
def _esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


def _pct_html(v, decimals=2):
    if v is None:
        return '<span class="dim">n/a</span>'
    cls = "pos" if v >= 0 else "neg"
    return f'<span class="{cls}">{v:+.{decimals}f}%</span>'


def render_html(basket_rank, basket_details, fetch_ts, data_date, failed,
                rs_window, sort_label, out_path):
    parts = []
    parts.append(f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>主題籃子 RS 掃描 — {_esc(data_date)}</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{ background:#0d1117; color:#e6edf3; margin:0; padding:24px;
         font-family:-apple-system,"PingFang TC","Microsoft JhengHei",
         "Helvetica Neue",Arial,monospace; }}
  h1 {{ font-size:20px; margin:0 0 4px; }}
  .meta {{ color:#8b949e; font-size:13px; margin-bottom:20px; }}
  .meta b {{ color:#e6edf3; }}
  h2 {{ font-size:16px; margin:28px 0 10px; border-left:4px solid #2f81f7;
        padding-left:10px; }}
  table {{ border-collapse:collapse; width:100%; margin-bottom:8px;
           font-variant-numeric:tabular-nums; }}
  th, td {{ padding:7px 10px; text-align:right; white-space:nowrap;
            border-bottom:1px solid #21262d; }}
  th {{ color:#58a6ff; font-weight:600; background:#161b22;
        position:sticky; top:0; }}
  th:first-child, td:first-child {{ text-align:left; }}
  td.ticker {{ font-weight:700; }}
  tr:hover td {{ background:#161b22; }}
  .pos {{ color:#3fb950; }}
  .neg {{ color:#f85149; }}
  .dim {{ color:#6e7681; }}
  .center {{ text-align:center; }}
  .rank-num {{ color:#8b949e; }}
  .fail {{ margin-top:24px; padding:12px 16px; background:#21121266;
           border:1px solid #f8514955; border-radius:8px; }}
  .fail h2 {{ border:0; padding:0; margin:0 0 8px; color:#f85149; }}
  .ok {{ color:#3fb950; margin-top:24px; }}
  .note {{ color:#6e7681; font-size:12px; margin-top:8px; }}
</style>
</head>
<body>
<h1>主題股票籃子 vs SPY — 相對強弱與動能掃描</h1>
<div class="meta">
  抓取時間戳 <b>{_esc(fetch_ts)}</b> &nbsp;·&nbsp;
  資料日期(最後交易日) <b>{_esc(data_date)}</b> &nbsp;·&nbsp;
  RS 視窗 <b>{rs_window}</b> 交易日 &nbsp;·&nbsp; 資料源 Yahoo Finance
</div>
""")

    # 籃子強弱排行
    parts.append("<h2>籃子強弱排行(等權平均 RS,由強到弱)</h2>")
    parts.append("<table><thead><tr><th>排名</th><th>籃子</th>"
                 "<th>平均 RS</th><th>成分股數</th></tr></thead><tbody>")
    for i, (name, avg_rs, n) in enumerate(basket_rank, 1):
        rs_cell = _pct_html(avg_rs) if avg_rs is not None else '<span class="dim">n/a</span>'
        parts.append(f'<tr><td class="rank-num">{i}</td>'
                     f'<td class="ticker">{_esc(name)}</td>'
                     f'<td>{rs_cell}</td><td>{n}</td></tr>')
    parts.append("</tbody></table>")

    # 各籃明細
    has_partial = False
    for name, avg_rs, rows in basket_details:
        rs_hdr = _pct_html(avg_rs) if avg_rs is not None else '<span class="dim">n/a</span>'
        parts.append(f"<h2>{_esc(name)} &nbsp;—&nbsp; 平均 RS {rs_hdr} "
                     f"<span class='dim'>(成分股{_esc(sort_label)})</span></h2>")
        parts.append("<table><thead><tr>"
                     "<th>Ticker</th><th>現價</th><th>市值</th><th>當日%</th>"
                     "<th>1週%</th><th>1月%</th><th>3月%</th>"
                     f"<th>RS({rs_window}d)</th><th>距52WH</th>"
                     "</tr></thead><tbody>")
        for t, m in rows:
            price = f"{m['price']:.2f}" if m["price"] is not None else "n/a"
            high_html = _pct_html(m["from_high"])
            if m.get("high_partial"):
                high_html += "<span class='dim'>*</span>"
                has_partial = True
            parts.append(
                f'<tr><td class="ticker">{_esc(t)}</td>'
                f'<td>{price}</td>'
                f'<td>{_esc(fmt_mcap(m["mcap"]))}</td>'
                f'<td>{_pct_html(m["day"])}</td>'
                f'<td>{_pct_html(m["r1w"])}</td>'
                f'<td>{_pct_html(m["r1m"])}</td>'
                f'<td>{_pct_html(m["r3m"])}</td>'
                f'<td>{_pct_html(m["rs"])}</td>'
                f'<td>{high_html}</td></tr>')
        parts.append("</tbody></table>")

    if failed:
        parts.append('<div class="fail"><h2>抓取失敗清單</h2><ul>')
        for t, reason in failed:
            parts.append(f"<li>✗ <b>{_esc(t)}</b>: {_esc(reason)}</li>")
        parts.append("</ul></div>")
    else:
        parts.append('<div class="ok">所有 ticker 皆成功抓取。</div>')

    if has_partial:
        parts.append('<div class="note">* 該檔上市未滿 52 週,'
                     '距高點以可得資料的最高價計算。</div>')

    parts.append("</body></html>")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="主題股票籃子 vs SPY 相對強弱與動能掃描器")
    parser.add_argument("--rs-window", type=int, default=21,
                        help="RS 計算的交易日天數(預設 21)")
    parser.add_argument("--basket", type=str, default=None,
                        help=f"只跑單一籃子,可選: {', '.join(BASKETS.keys())}")
    parser.add_argument("--html", nargs="?", const="theme_rs_report.html",
                        default=None, metavar="PATH",
                        help="輸出 HTML 報告檔(預設 theme_rs_report.html);"
                             "用瀏覽器開啟")
    parser.add_argument("--sort", choices=["mcap", "rs"], default="mcap",
                        help="籃內成分股排序依據:mcap=市值由大到小(預設)、"
                             "rs=相對強度由強到弱")
    args = parser.parse_args()

    if args.rs_window < 1:
        sys.exit("--rs-window 必須 >= 1")

    # 決定要跑哪些籃子
    if args.basket:
        if args.basket not in BASKETS:
            sys.exit(f"未知籃子 '{args.basket}',可選: {', '.join(BASKETS.keys())}")
        active_baskets = {args.basket: BASKETS[args.basket]}
    else:
        active_baskets = dict(BASKETS)

    # 收集所有需要抓的 ticker(含基準)
    all_tickers = []
    for syms in active_baskets.values():
        all_tickers.extend(syms)
    all_tickers = list(dict.fromkeys(all_tickers))  # 去重保序
    fetch_list = all_tickers + [BENCHMARK]

    fetch_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z").strip()
    print(f"[抓取中] {len(fetch_list)} 檔(含基準 {BENCHMARK})...", file=sys.stderr)

    data, failed, last_date = fetch_all(fetch_list)

    # 基準是 RS 計算的必要條件
    if BENCHMARK not in data:
        reason = next((r for t, r in failed if t == BENCHMARK), "未知原因")
        sys.exit(f"基準 {BENCHMARK} 抓取失敗,無法計算 RS:{reason}")

    spy_close = data[BENCHMARK]["close"]
    spy_rs_return = pct_return(spy_close, args.rs_window)

    data_date = last_date.strftime("%Y-%m-%d") if last_date is not None else "n/a"

    # 抓流通股數以計算市值(個股,基準不需要)
    print("[抓取市值] 取流通股數中...", file=sys.stderr)
    shares = fetch_shares([t for t in all_tickers if t in data])

    # 計算每檔指標
    metrics = {}
    for t in all_tickers:
        if t not in data:
            continue  # 已在 failed 清單
        m = compute_metrics(
            data[t]["close"], data[t]["volume"], args.rs_window, spy_rs_return)
        sh = shares.get(t)
        m["mcap"] = (sh * m["price"]) if (sh and m["price"] is not None) else None
        metrics[t] = m

    # 籃子層級彙總
    basket_avg = {}
    basket_rows = {}
    for name, syms in active_baskets.items():
        rows = [(t, metrics[t]) for t in syms if t in metrics]
        # 成分股排序(None 一律排最後)
        if args.sort == "mcap":
            rows.sort(key=lambda x: (x[1]["mcap"] is None, -(x[1]["mcap"] or 0)))
        else:
            rows.sort(key=lambda x: (x[1]["rs"] is None, -(x[1]["rs"] or 0)))
        basket_rows[name] = rows
        rs_vals = [m["rs"] for _, m in rows if m["rs"] is not None]
        basket_avg[name] = (sum(rs_vals) / len(rs_vals)) if rs_vals else None

    # 籃子排行:平均 RS 由強到弱(None 排最後)
    rank = sorted(
        active_baskets.keys(),
        key=lambda n: (basket_avg[n] is None, -(basket_avg[n] or 0)),
    )
    basket_rank = [(n, basket_avg[n], len(basket_rows[n])) for n in rank]
    basket_details = [(n, basket_avg[n], basket_rows[n]) for n in rank]

    sort_label = "按市值由大到小" if args.sort == "mcap" else "按 RS 由強到弱"

    if args.html is not None:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(args.html)), exist_ok=True)
        render_html(basket_rank, basket_details, fetch_ts, data_date, failed,
                    args.rs_window, sort_label, args.html)
        abspath = os.path.abspath(args.html)
        print(f"HTML 報告已輸出: {abspath}")
        print(f"用瀏覽器開啟:  file://{abspath}")
    elif HAVE_RICH:
        render_rich(basket_rank, basket_details, fetch_ts, data_date, failed,
                    args.rs_window, sort_label)
    else:
        render_plain(basket_rank, basket_details, fetch_ts, data_date, failed,
                     args.rs_window, sort_label)


if __name__ == "__main__":
    main()
