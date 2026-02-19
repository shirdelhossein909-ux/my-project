# -*- coding: utf-8 -*-
import time
import json
import os
import math
import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import MetaTrader5 as mt5

import strategy_backtest as st  # <-- دست نزن


# ================== تنظیمات ==================
SYMBOLS = ["EURJPY", "EURCAD", "GBPJPY", "CHFJPY", "XAUUSD" , "EURUSD"]

MAGIC = 123456
RR_TP = 3.0
RR_CANCEL = 4.0

CHECK_CANCEL_SEC = 300
POLL_SEC = 5
RECONNECT_SEC = 5

MAX_ORDERS_PER_SYMBOL = 3
MAX_PENDING_ORDERS = 5
MAX_OPEN_POS = 3

ENTRY_OFF = 0.10
SL_OFF = 0.25
RESERVE = 0.15
RISK_PER_TRADE = 0.01
MAX_TOTAL_RISK = 0.03

DRY_RUN = False
STATE_FILE = "live_state.json"
MT5_TERMINAL_PATH = None


# Lookback: لازم نیست «۳ سال» باشد؛ ولی باید به اندازه‌ای باشد که:
# - ATR(14) و range_filter(20) و swing/trend درست warmup شوند
# - زون‌ها تعداد کافی داشته باشند
H4_BARS = 2500
D1_BARS = 1200
W1_BARS = 400


# ================== لاگ ==================
LOG_DIR = os.path.join(os.path.expanduser("~"), "Desktop", "log")
USER_LOG_FILE = "live_user.log"
AGENT_LOG_FILE = "live_agent.log"
SYMBOL_RESOLVE_CACHE = {}


def ensure_agent_log_rotation(log_dir, agent_filename):
    os.makedirs(log_dir, exist_ok=True)
    agent_path = os.path.join(log_dir, agent_filename)

    if not os.path.exists(agent_path):
        return

    modified_dt = datetime.fromtimestamp(os.path.getmtime(agent_path))
    age = datetime.now() - modified_dt
    if age < timedelta(days=3):
        return

    start_dt = modified_dt
    end_dt = modified_dt + timedelta(days=2)
    archived_name = f"{start_dt.day}-{start_dt.month}-{start_dt.year}تا{end_dt.day}-{end_dt.month}-{end_dt.year}.log"
    archived_path = os.path.join(log_dir, archived_name)

    if os.path.exists(archived_path):
        suffix = datetime.now().strftime("_%H%M%S")
        archived_path = archived_path.replace(".log", f"{suffix}.log")

    os.rename(agent_path, archived_path)


def setup_logging():
    ensure_agent_log_rotation(LOG_DIR, AGENT_LOG_FILE)
    os.makedirs(LOG_DIR, exist_ok=True)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    user_file_handler = logging.FileHandler(os.path.join(LOG_DIR, USER_LOG_FILE), encoding="utf-8")
    user_file_handler.setFormatter(formatter)

    agent_file_handler = logging.FileHandler(os.path.join(LOG_DIR, AGENT_LOG_FILE), encoding="utf-8")
    agent_file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(user_file_handler)
    logger.addHandler(agent_file_handler)


setup_logging()

def log_info(msg): logging.info(msg)
def log_warn(msg): logging.warning(msg)
def log_err(msg):  logging.error(msg)


def _mt5_enum_name(prefix: str, value: int) -> str:
    for k in dir(mt5):
        if k.startswith(prefix) and int(getattr(mt5, k)) == int(value):
            return k
    return f"{prefix}UNKNOWN({value})"


# ================== State ==================
def load_state():
    if os.path.isfile(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            log_warn(f"STATE load failed, recreating. reason={e}")
    return {"last_h4_time": {}, "placed_zoneids": []}

def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


# ================== اتصال MT5 ==================
def connect():
    ok = mt5.initialize(path=MT5_TERMINAL_PATH) if MT5_TERMINAL_PATH else mt5.initialize()
    if not ok:
        code, msg = mt5.last_error()
        raise RuntimeError(f"MT5 initialize failed | code={code} msg={msg}")

    term = mt5.terminal_info()
    acc = mt5.account_info()

    log_info(f"MT5 initialized | name={getattr(term,'name',None)} build={getattr(term,'build',None)}")

    if acc:
        log_info(
            f"Account connected | login={acc.login} name={acc.name} server={acc.server} "
            f"balance={acc.balance} equity={acc.equity} currency={acc.currency}"
        )
        log_info(f"Trade allowed? terminal={term.trade_allowed if term else None} account={acc.trade_allowed}")
    else:
        log_warn("account_info() is None (احتمالاً داخل MT5 لاگین نیستی)")

def shutdown():
    mt5.shutdown()
    log_info("MT5 shutdown")


def is_mt5_connected():
    term = mt5.terminal_info()
    acc = mt5.account_info()
    if term is None or acc is None:
        return False
    connected = getattr(term, "connected", None)
    if connected is None:
        return True
    return bool(connected)


def ensure_connection():
    if is_mt5_connected():
        return True

    log_warn("MT5 connection lost. trying reconnect...")
    mt5.shutdown()

    while True:
        try:
            connect()
            if is_mt5_connected():
                log_info("MT5 reconnected successfully")
                return True
        except Exception as e:
            log_warn(f"MT5 reconnect failed: {e}")

        time.sleep(RECONNECT_SEC)


# ================== ابزار دیتا ==================
def rates_to_df(rates):
    if rates is None or len(rates) == 0:
        return pd.DataFrame()
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df[["time", "open", "high", "low", "close"]].copy()


def resolve_symbol(requested_symbol):
    cached = SYMBOL_RESOLVE_CACHE.get(requested_symbol)
    if cached:
        info = mt5.symbol_info(cached)
        if info is not None:
            return cached

    info = mt5.symbol_info(requested_symbol)
    if info is not None:
        SYMBOL_RESOLVE_CACHE[requested_symbol] = requested_symbol
        return requested_symbol

    all_symbols = mt5.symbols_get()
    if not all_symbols:
        return None

    base = str(requested_symbol).upper()
    names = [s.name for s in all_symbols if getattr(s, "name", None)]

    exact = [n for n in names if n.upper() == base]
    starts = [n for n in names if n.upper().startswith(base)]
    contains = [n for n in names if base in n.upper()]
    candidates = exact + starts + contains

    # اولویت: نماد قابل معامله
    for name in candidates:
        inf = mt5.symbol_info(name)
        if inf is None:
            continue
        if (inf.trade_mode != mt5.SYMBOL_TRADE_MODE_DISABLED):
            SYMBOL_RESOLVE_CACHE[requested_symbol] = name
            if name != requested_symbol:
                log_info(f"symbol mapped: {requested_symbol} -> {name}")
            return name

    # اگر نماد قابل معامله پیدا نشد، همان اولین کاندید را بده برای لاگ دقیق‌تر
    if candidates:
        chosen = candidates[0]
        SYMBOL_RESOLVE_CACHE[requested_symbol] = chosen
        if chosen != requested_symbol:
            log_warn(f"symbol mapped (possibly not tradable): {requested_symbol} -> {chosen}")
        return chosen

    return None

def ensure_symbol(symbol):
    real_symbol = resolve_symbol(symbol)
    if real_symbol is None:
        code, msg = mt5.last_error()
        log_warn(f"{symbol} symbol not found in terminal | code={code} msg={msg}")
        return None

    info = mt5.symbol_info(real_symbol)
    if info is None:
        code, msg = mt5.last_error()
        log_warn(f"{symbol} symbol_info=None (resolved={real_symbol}) | code={code} msg={msg}")
        return None

    if not info.visible:
        if not mt5.symbol_select(real_symbol, True):
            code, msg = mt5.last_error()
            log_warn(f"{symbol} symbol_select failed (resolved={real_symbol}) | code={code} msg={msg}")
            return None

        info = mt5.symbol_info(real_symbol)
        if info is None:
            log_warn(f"{symbol} symbol_info became None after select (resolved={real_symbol})")
            return None

    if info.trade_mode == mt5.SYMBOL_TRADE_MODE_DISABLED:
        log_warn(f"{symbol} trade_mode=DISABLED (resolved={real_symbol})")
        return None

    return real_symbol

def get_closed_bars(symbol, timeframe, n):
    real_symbol = ensure_symbol(symbol)
    if not real_symbol:
        return pd.DataFrame()
    rates = mt5.copy_rates_from_pos(real_symbol, timeframe, 1, int(n))  # pos=1 => کندل بسته
    return rates_to_df(rates)

def get_spread_now(symbol):
    real_symbol = ensure_symbol(symbol)
    if not real_symbol:
        return None
    tick = mt5.symbol_info_tick(real_symbol)
    if tick is None:
        return None
    return float(tick.ask - tick.bid)

def get_last_closed_time(symbol, timeframe):
    df = get_closed_bars(symbol, timeframe, 1)
    if df.empty:
        return None
    return pd.to_datetime(df["time"].iloc[-1])

def pending_orders(symbol):
    real_symbol = resolve_symbol(symbol) or symbol
    od = mt5.orders_get(symbol=real_symbol)
    return list(od) if od else []

def positions(symbol=None):
    real_symbol = (resolve_symbol(symbol) or symbol) if symbol else None
    ps = mt5.positions_get(symbol=real_symbol) if real_symbol else mt5.positions_get()
    return list(ps) if ps else []


def _risk_money_for_setup(symbol, direction, volume, entry_price, sl_price):
    real_symbol = resolve_symbol(symbol) or symbol
    order_type = mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL
    pnl = mt5.order_calc_profit(order_type, real_symbol, float(volume), float(entry_price), float(sl_price))
    if pnl is None:
        info = mt5.symbol_info(real_symbol)
        if info is None:
            return 0.0
        tick_value = float(getattr(info, "trade_tick_value", 0.0) or 0.0)
        tick_size = float(getattr(info, "trade_tick_size", 0.0) or 0.0)
        if tick_value <= 0.0 or tick_size <= 0.0:
            return 0.0
        ticks = abs(float(entry_price) - float(sl_price)) / tick_size
        return abs(ticks * tick_value * float(volume))
    return abs(float(pnl))


def account_total_risk_money():
    total = 0.0

    for p in positions():
        sl = float(getattr(p, "sl", 0.0) or 0.0)
        if sl <= 0.0:
            continue
        direction = "BUY" if int(p.type) == int(mt5.POSITION_TYPE_BUY) else "SELL"
        total += _risk_money_for_setup(p.symbol, direction, float(p.volume), float(p.price_open), sl)

    return float(total)


def total_pending_orders_count():
    all_orders = mt5.orders_get()
    if not all_orders:
        return 0

    return len([
        o for o in all_orders
        if int(getattr(o, "magic", 0) or 0) == int(MAGIC)
        and o.type in (mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_SELL_LIMIT)
    ])


# ================== ترید/کنسل ==================
def cancel_order(ticket, reason: Optional[str] = None):
    req = {"action": mt5.TRADE_ACTION_REMOVE, "order": int(ticket)}
    reason_txt = f" | reason={reason}" if reason else ""
    if DRY_RUN:
        log_info(f"DRY_RUN cancel -> {req}{reason_txt}")
        return None
    res = mt5.order_send(req)
    log_info(f"cancel result -> {res}{reason_txt}")
    return res

def calc_volume_by_risk(symbol, entry_price, sl_price, risk_pct, reserve_pct):
    acc = mt5.account_info()
    real_symbol = resolve_symbol(symbol) or symbol
    info = mt5.symbol_info(real_symbol)
    if acc is None or info is None:
        return 0.01

    equity = float(acc.equity)
    usable = equity * (1.0 - float(reserve_pct))
    risk_money = usable * float(risk_pct)

    tick_value = float(info.trade_tick_value)
    tick_size = float(info.trade_tick_size)
    if tick_value <= 0 or tick_size <= 0:
        return float(info.volume_min)

    price_dist = abs(float(entry_price) - float(sl_price))
    ticks = price_dist / tick_size
    risk_per_lot = ticks * tick_value
    if risk_per_lot <= 0:
        return float(info.volume_min)

    vol = risk_money / risk_per_lot

    step = float(info.volume_step)
    vmin = float(info.volume_min)
    vmax = float(info.volume_max)

    vol = max(vmin, min(vmax, vol))
    vol = math.floor(vol / step) * step
    return round(float(vol), 6)

def place_limit(symbol, direction, volume, entry_price, sl_price, tp_price, comment):
    real_symbol = ensure_symbol(symbol)
    if not real_symbol:
        log_warn(f"{symbol} place skipped: symbol not tradable")
        return None

    otype = mt5.ORDER_TYPE_BUY_LIMIT if direction == "BUY" else mt5.ORDER_TYPE_SELL_LIMIT

    base_req = {
        "action": mt5.TRADE_ACTION_PENDING,
        "symbol": real_symbol,
        "volume": float(volume),
        "type": int(otype),
        "price": float(entry_price),
        "sl": float(sl_price),
        "tp": float(tp_price),
        "deviation": 20,
        "magic": int(MAGIC),
        "comment": str(comment),
        "type_time": mt5.ORDER_TIME_GTC,
    }

    if DRY_RUN:
        dry_req = dict(base_req)
        dry_req["type_filling"] = mt5.ORDER_FILLING_RETURN
        log_info(f"DRY_RUN place -> {dry_req}")
        return None

    fill_policies = [mt5.ORDER_FILLING_RETURN, mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_FOK]
    for fill_type in fill_policies:
        req = dict(base_req)
        req["type_filling"] = fill_type

        res = mt5.order_send(req)
        log_info(f"place result -> fill={fill_type} | {res}")

        if res is None:
            code, msg = mt5.last_error()
            log_warn(f"order_send returned None | fill={fill_type} | code={code} msg={msg}")
            continue

        if int(getattr(res, "retcode", 0)) == mt5.TRADE_RETCODE_DONE:
            return res

        if int(getattr(res, "retcode", 0)) != mt5.TRADE_RETCODE_INVALID_FILL:
            return res

        log_warn(f"invalid filling mode for {symbol}, retrying with next policy")

    return res


# ================== قیمت‌سازی (سازگارتر با bid/ask) ==================
def make_mt5_prices_from_zone(direction, proximal, distal, spread):
    """
    فرض: OHLCهایی که از MT5 می‌گیریم معمولاً Bid هستند.
    BUY: ورود با Ask انجام می‌شود => order price = entry_bid + spread
         SL/TP برای پوزیشن BUY با Bid تریگر می‌شوند => sl,tp را bid-level می‌گذاریم
    SELL: ورود با Bid انجام می‌شود => order price = entry_bid
          SL/TP برای پوزیشن SELL با Ask تریگر می‌شوند => sl,tp را ask-level می‌گذاریم
    """
    height = max(1e-9, float(max(proximal, distal) - min(proximal, distal)))
    sp = float(max(0.0, spread))

    if direction == "BUY":
        entry_bid = float(proximal - ENTRY_OFF * height)
        sl_bid    = float(distal  - SL_OFF    * height)

        open_ask = entry_bid + sp
        risk = open_ask - sl_bid
        tp_bid = open_ask + RR_TP * risk

        order_price = open_ask   # BUY_LIMIT triggers on Ask
        sl_price = sl_bid        # triggers on Bid
        tp_price = tp_bid        # triggers on Bid
        return order_price, sl_price, tp_price, risk

    else:  # SELL
        entry_bid = float(proximal + ENTRY_OFF * height)
        sl_bid    = float(distal  + SL_OFF    * height)

        sl_ask = sl_bid + sp
        risk = sl_ask - entry_bid
        tp_ask = entry_bid - RR_TP * risk

        order_price = entry_bid  # SELL_LIMIT triggers on Bid
        sl_price = sl_ask        # triggers on Ask
        tp_price = tp_ask        # triggers on Ask
        return order_price, sl_price, tp_price, risk


# ================== ساخت کاندیدها از دیتای MT5 (همان منطق فعلی خودت) ==================
def find_active_pending_candidates(symbol, spread):
    h4 = get_closed_bars(symbol, mt5.TIMEFRAME_H4, H4_BARS)
    d1 = get_closed_bars(symbol, mt5.TIMEFRAME_D1, D1_BARS)
    w1 = get_closed_bars(symbol, mt5.TIMEFRAME_W1, W1_BARS)

    if h4.empty or d1.empty or w1.empty:
        log_warn(f"{symbol} not enough data | h4={len(h4)} d1={len(d1)} w1={len(w1)}")
        return []

    for df in (h4, d1, w1):
        df["open"] = df["open"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"]  = df["low"].astype(float)
        df["close"]= df["close"].astype(float)

    h4["atr"] = st.atr(h4); d1["atr"] = st.atr(d1); w1["atr"] = st.atr(w1)
    h4["range"] = st.range_filter(h4, h4["atr"])
    d1["range"] = st.range_filter(d1, d1["atr"])
    h4["trend"] = st.trend_from_swings(h4, n=1)
    d1["trend"] = st.trend_from_swings(d1, n=1)

    w_z = st.dedup_zones(st.build_zones(w1, symbol, "W1", 12, w1["atr"]))
    h_z = st.dedup_zones(st.build_zones(h4, symbol, "H4", 6,  h4["atr"]))

    h_z = sorted(h_z, key=lambda z: z.created_time)
    for idx, z in enumerate(h_z, start=1):
        z.zone_id = f"{symbol}_H4_{idx:05d}"

    d_times = d1["time"].values
    import numpy as np
    def last_idx_leq(times, t):
        return np.searchsorted(times, t, side="right") - 1

    pending = []
    used = set()

    for i in range(len(h4)):
        t = h4["time"].iloc[i]
        o = float(h4["open"].iloc[i])
        h = float(h4["high"].iloc[i])
        l = float(h4["low"].iloc[i])
        c = float(h4["close"].iloc[i])

        di = last_idx_leq(d_times, t.to_datetime64())
        if di < 0:
            continue

        dtr = int(d1["trend"].iloc[di])
        htr = int(h4["trend"].iloc[i])
        drg = bool(d1["range"].iloc[di]) if not pd.isna(d1["range"].iloc[di]) else False
        hrg = bool(h4["range"].iloc[i]) if not pd.isna(h4["range"].iloc[i]) else False

        # touches + expiry
        for z in h_z:
            if z.created_time > t or z.expired:
                continue

            touched = (h >= z.low() and l <= z.high())
            if touched:
                if z.touch_count == 0:
                    z.touch_count = 1; z.last_touch_i = i; z.clean_after_touch = 0
                else:
                    if z.clean_after_touch >= 3 and z.last_touch_i is not None and (i - z.last_touch_i) <= 50:
                        z.touch_count += 1; z.last_touch_i = i; z.clean_after_touch = 0
                    else:
                        z.clean_after_touch = 0
            else:
                if z.touch_count > 0:
                    z.clean_after_touch += 1

            if z.touch_count == 1 and z.last_touch_i is not None and (i - z.last_touch_i) > 50:
                z.expired = True

        # place orders
        for z in h_z:
            if z.created_time > t or z.expired or id(z) in used:
                continue
            if z.touch_count >= 3:
                used.add(id(z)); continue
            if z.touch_count == 0:
                continue
            if drg or hrg:
                used.add(id(z)); continue
            if dtr == 0 or htr == 0 or dtr != htr:
                used.add(id(z)); continue
            if len([p for p in pending if p["active"] and (not p["filled"])]) >= MAX_ORDERS_PER_SYMBOL:
                used.add(id(z)); continue

            order_price, sl_price, tp_price, risk = make_mt5_prices_from_zone(
                z.direction, z.proximal, z.distal, spread
            )
            if risk <= 0:
                used.add(id(z)); continue

            pending.append({
                "zone_id": z.zone_id,
                "direction": z.direction,
                "price": order_price,
                "sl": sl_price,
                "tp": tp_price,
                "risk": risk,
                "placed_time": t,
                "active": True,
                "filled": False,
                "o": o, "c": c, "h": h, "l": l,
            })
            used.add(id(z))

        # weekly cancel BEFORE fill (بدنه)
        wz_now = [wz for wz in w_z if wz.created_time <= t]
        for p in pending:
            if not p["active"] or p["filled"]:
                continue
            opp_dir = "SELL" if p["direction"] == "BUY" else "BUY"
            opp = [wz for wz in wz_now if wz.direction == opp_dir]
            if any(st.body_overlaps_zone(o, c, wz) for wz in opp):
                p["active"] = False

        # RR4 cancel (قانون جدید روی تاریخچه)
        for p in pending:
            if not p["active"] or p["filled"]:
                continue
            # روی bid-history این تقریبی است؛ برای لایو کنسل واقعی را جداگانه مدیریت می‌کنیم
            if p["direction"] == "BUY":
                # اگر قیمت خیلی بالا رفت دیگر سفارش را نمی‌خواهیم
                if h >= (p["price"] + RR_CANCEL * p["risk"]):
                    p["active"] = False
            else:
                if l <= (p["price"] - RR_CANCEL * p["risk"]):
                    p["active"] = False

        # اگر در گذشته فیل می‌شد، الان نباید فعال باشد
        for p in pending:
            if not p["active"] or p["filled"]:
                continue
            if p["direction"] == "BUY":
                # BUY_LIMIT triggers on Ask; روی bid-history دقیق نیست، ولی به‌عنوان فیلتر کفایت می‌کند
                if l <= (p["price"] - float(spread)):
                    p["filled"] = True; p["active"] = False
            else:
                if h >= p["price"]:
                    p["filled"] = True; p["active"] = False

    alive = [p for p in pending if p["active"] and (not p["filled"])]
    return alive


# ================== لایه‌ها (SRP) ==================
class MT5Gateway:
    """مسئول ارتباط با MT5 و عملیات سفارش/پوزیشن."""

    def connect(self):
        return connect()

    def shutdown(self):
        return shutdown()

    def ensure_connection(self):
        return ensure_connection()

    def get_last_closed_time(self, symbol, timeframe):
        return get_last_closed_time(symbol, timeframe)

    def get_spread_now(self, symbol):
        return get_spread_now(symbol)

    def pending_orders(self, symbol):
        return pending_orders(symbol)

    def positions(self, symbol=None):
        return positions(symbol)

    def total_pending_orders_count(self):
        return total_pending_orders_count()

    def cancel_order(self, ticket, reason=None):
        return cancel_order(ticket, reason=reason)

    def place_limit(self, symbol, direction, volume, entry_price, sl_price, tp_price, comment):
        return place_limit(symbol, direction, volume, entry_price, sl_price, tp_price, comment)

    def manage_cancel_rr4(self, symbol):
        real_symbol = resolve_symbol(symbol) or symbol
        tick = mt5.symbol_info_tick(real_symbol)
        if tick is None:
            return

        cur_bid = float(tick.bid)
        cur_ask = float(tick.ask)

        for o in self.pending_orders(symbol):
            if int(o.magic) != int(MAGIC):
                continue
            if o.type not in (mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_SELL_LIMIT):
                continue
            if o.sl is None or float(o.sl) == 0.0:
                continue

            entry = float(o.price_open)
            sl = float(o.sl)
            risk = abs(entry - sl)
            if risk <= 0:
                continue

            if o.type == mt5.ORDER_TYPE_BUY_LIMIT and cur_bid >= (entry + RR_CANCEL * risk):
                log_info(f"{symbol} RR4 cancel BUY_LIMIT | ticket={o.ticket}")
                self.cancel_order(o.ticket, reason="rr4_threshold_crossed")
            elif o.type == mt5.ORDER_TYPE_SELL_LIMIT and cur_ask <= (entry - RR_CANCEL * risk):
                log_info(f"{symbol} RR4 cancel SELL_LIMIT | ticket={o.ticket}")
                self.cancel_order(o.ticket, reason="rr4_threshold_crossed")

    def reconcile_orders_with_candidates(self, symbol, alive_candidates):
        desired = set([p["zone_id"] for p in alive_candidates])

        for o in self.pending_orders(symbol):
            if int(o.magic) != int(MAGIC):
                continue
            if o.type not in (mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_SELL_LIMIT):
                continue

            zid = str(o.comment).strip() if o.comment is not None else ""
            if zid and zid not in desired:
                log_info(f"{symbol} cancel stale order (not alive anymore) | ticket={o.ticket} comment={zid}")
                self.cancel_order(o.ticket, reason="stale_not_alive")

    def log_closed_deals(self, from_dt):
        """علت سود/ضرر شدن معاملات بسته‌شده را ثبت می‌کند."""
        if from_dt is None:
            return datetime.now()

        to_dt = datetime.now()
        deals = mt5.history_deals_get(from_dt, to_dt)
        if not deals:
            return to_dt

        for d in sorted(deals, key=lambda x: int(getattr(x, "time", 0) or 0)):
            if int(getattr(d, "magic", 0) or 0) != int(MAGIC):
                continue
            if int(getattr(d, "entry", -1)) != int(getattr(mt5, "DEAL_ENTRY_OUT", -1)):
                continue

            profit = float(getattr(d, "profit", 0.0) or 0.0)
            if profit > 0:
                outcome = "profit"
            elif profit < 0:
                outcome = "loss"
            else:
                outcome = "breakeven"

            reason_name = _mt5_enum_name("DEAL_REASON_", int(getattr(d, "reason", -1) or -1))
            type_name = _mt5_enum_name("DEAL_TYPE_", int(getattr(d, "type", -1) or -1))
            log_info(
                f"DEAL CLOSED | ticket={getattr(d,'position_id',None)} symbol={getattr(d,'symbol',None)} "
                f"outcome={outcome} profit={profit:.2f} reason={reason_name} type={type_name} "
                f"comment={getattr(d,'comment',None)}"
            )

        return to_dt


class RiskManager:
    """مسئول مدیریت ریسک سفارش جدید."""

    def can_place(self, symbol, direction, volume, entry_price, sl_price):
        next_risk = _risk_money_for_setup(symbol, direction, volume, entry_price, sl_price)
        acc = mt5.account_info()
        equity = float(acc.equity) if acc else 0.0
        max_allowed_risk = max(0.0, equity * MAX_TOTAL_RISK)
        current_risk = account_total_risk_money()

        if (current_risk + next_risk) > max_allowed_risk:
            reason = (
                f"risk cap reached | current={current_risk:.2f} "
                f"next={next_risk:.2f} cap={max_allowed_risk:.2f}"
            )
            return False, reason

        return True, "ok"


class StrategyEngine:
    """مسئول تولید کاندیداها از استراتژی."""

    def alive_candidates(self, symbol, spread):
        return find_active_pending_candidates(symbol, spread=spread)


class LiveTraderApp:
    """ارکستریتور اجرا: آماده توسعه چند نماد/چند استراتژی."""

    def __init__(self, symbols, gateway, strategy_engine, risk_manager):
        self.symbols = symbols
        self.gateway = gateway
        self.strategy_engine = strategy_engine
        self.risk_manager = risk_manager
        self.state = load_state()
        self.last_cancel_check = 0
        self.last_deal_scan = datetime.now() - timedelta(minutes=10)

    def run(self):
        self.gateway.connect()
        log_info("Watching symbols: " + ", ".join(self.symbols))

        try:
            while True:
                self.gateway.ensure_connection()
                now = time.time()

                self.last_deal_scan = self.gateway.log_closed_deals(self.last_deal_scan)

                if now - self.last_cancel_check >= CHECK_CANCEL_SEC:
                    for sym in self.symbols:
                        self.gateway.manage_cancel_rr4(sym)
                    self.last_cancel_check = now

                for sym in self.symbols:
                    t_h4 = self.gateway.get_last_closed_time(sym, mt5.TIMEFRAME_H4)
                    if t_h4 is None:
                        continue

                    last_done = self.state["last_h4_time"].get(sym)
                    t_key = t_h4.isoformat()
                    if last_done == t_key:
                        continue

                    self.state["last_h4_time"][sym] = t_key
                    save_state(self.state)

                    if len(self.gateway.positions()) >= MAX_OPEN_POS:
                        log_info(f"{sym} skip: max open positions reached")
                        continue

                    if self.gateway.total_pending_orders_count() >= MAX_PENDING_ORDERS:
                        log_info(f"{sym} skip: max pending orders reached ({MAX_PENDING_ORDERS})")
                        continue

                    spread = self.gateway.get_spread_now(sym)
                    if spread is None:
                        log_warn(f"{sym} skip: no spread/tick")
                        continue

                    alive = self.strategy_engine.alive_candidates(sym, spread=spread)
                    log_info(f"{sym} H4 closed -> {t_h4} | alive={len(alive)} | spread={spread}")

                    self.gateway.reconcile_orders_with_candidates(sym, alive)

                    my_orders = [o for o in self.gateway.pending_orders(sym) if int(o.magic) == int(MAGIC)]
                    free_slots = max(0, MAX_ORDERS_PER_SYMBOL - len(my_orders))
                    existing_zoneids = set([str(o.comment).strip() for o in my_orders if o.comment])

                    for p in alive:
                        if free_slots <= 0:
                            break

                        if self.gateway.total_pending_orders_count() >= MAX_PENDING_ORDERS:
                            log_info(f"{sym} skip place: reached max pending orders ({MAX_PENDING_ORDERS})")
                            break

                        zid = p["zone_id"]
                        if zid in existing_zoneids:
                            continue

                        vol = calc_volume_by_risk(sym, p["price"], p["sl"], RISK_PER_TRADE, RESERVE)

                        ok_risk, risk_reason = self.risk_manager.can_place(
                            sym, p["direction"], vol, p["price"], p["sl"]
                        )
                        if not ok_risk:
                            log_warn(f"{sym} skip place: {risk_reason}")
                            continue

                        log_info(
                            f"{sym} PLACE {p['direction']} | zone={zid} | price={p['price']} "
                            f"sl={p['sl']} tp={p['tp']} | vol={vol}"
                        )

                        self.gateway.place_limit(sym, p["direction"], vol, p["price"], p["sl"], p["tp"], comment=zid)

                        existing_zoneids.add(zid)
                        free_slots -= 1

                time.sleep(POLL_SEC)

        finally:
            self.gateway.shutdown()


# ================== MAIN ==================
def main():
    app = LiveTraderApp(
        symbols=SYMBOLS,
        gateway=MT5Gateway(),
        strategy_engine=StrategyEngine(),
        risk_manager=RiskManager(),
    )
    app.run()


if __name__ == "__main__":
    main()
