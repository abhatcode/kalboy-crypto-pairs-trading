import math

import pytest

from kalboy.engine import PairEngine


def engine(**kw):
    base = dict(y='Y', x='X', beta=1.0, alpha=0.0, process_noise=1e-9, measurement_noise=1e-6,
                zscore_window=10, burn_in=0, entry_z=2.0, exit_z=0.5, position_pct=0.10)
    base.update(kw)
    return PairEngine(**base)


def warm(e, n=30):
    for i in range(n):
        e.update(100.0 + 0.01 * (i % 2), 100.0)


def fill_all(e, decision, price=100.0, fee=0.0):
    for _, qty in decision.orders:
        e.record_fill(qty, price, fee)


def test_enter_long_when_z_below_minus_entry():
    e = engine()
    warm(e)
    e.update(99.0, 100.0)
    assert e.z < -2
    d = e.decide(100_000.0)
    assert d.action == 'enter_long'
    (sy, qy), (sx, qx) = d.orders
    assert (sy, sx) == ('Y', 'X') and qy > 0 and qx < 0
    gross = qy * 99.0 + abs(qx) * 100.0
    assert gross == pytest.approx(10_000.0, rel=1e-6)


def test_exit_unwinds_exactly_what_was_opened():
    e = engine()
    warm(e)
    e.update(99.0, 100.0)
    d = e.decide(100_000.0)
    fill_all(e, d, 99.0)
    held = dict(d.orders)
    for _ in range(12):
        e.update(100.0, 100.0)
        d2 = e.decide(100_000.0)
        if d2:
            break
    assert d2.action == 'exit_long'
    assert dict(d2.orders) == {s: -q for s, q in held.items()}
    assert e.held_y == 0 and e.held_x == 0


def test_round_trip_return_is_pnl_over_entry_equity():
    e = engine()
    warm(e)
    e.update(99.0, 100.0)
    d = e.decide(50_000.0)
    qy, qx = d.orders[0][1], d.orders[1][1]
    e.record_fill(qy, 99.0, 1.0)
    e.record_fill(qx, 100.0, 1.0)
    e.accrue_funding(0.001, 0.001)
    d2 = e.force_close()
    e.record_fill(-qy, 100.0, 1.0)
    e.record_fill(-qx, 100.0, 1.0)
    t = e.trade_log[0]
    funding = qy * 99.0 * 0.001 + qx * 100.0 * 0.001
    pnl = qy * (100.0 - 99.0) - 4.0 - funding
    assert t['pnl'] == pytest.approx(pnl)
    assert t['return'] == pytest.approx(pnl / 50_000.0)
    assert t['funding'] == pytest.approx(funding) and t['fees'] == pytest.approx(4.0)
    assert d2.action == 'period_end'


def test_funding_sign_long_pays_short_receives():
    e = engine()
    e.update(100.0, 100.0)
    e.held_y, e.held_x = 2.0, -1.0
    assert e.accrue_funding(0.01, 0.0) == pytest.approx(2.0)
    assert e.accrue_funding(0.0, 0.01) == pytest.approx(-1.0)


def test_z_is_bounded_by_window_length():
    # |z| can't exceed (n - 1) / sqrt(n) when the bar is in its own window
    e = engine(zscore_window=10)
    warm(e)
    e.update(10.0, 100.0)
    assert abs(e.z) <= 9 / 10 ** 0.5 + 1e-9


def test_stop_loss_and_time_stop():
    e = engine(stop_z=3.0, zscore_window=30)
    warm(e, 60)
    e.update(99.0, 100.0)
    fill_all(e, e.decide(100_000.0))
    e.update(90.0, 100.0)
    assert e.z < -3.0
    assert e.decide(100_000.0).action == 'stop_loss'

    e = engine(max_holding_bars=3, exit_z=-10.0)
    warm(e)
    e.update(99.0, 100.0)
    fill_all(e, e.decide(100_000.0))
    actions = []
    for _ in range(3):
        e.update(98.9, 100.0)
        d = e.decide(100_000.0)
        actions.append(d.action if d else None)
    assert actions[-1] == 'time_stop'


def test_breakdown_retires_the_pair():
    e = engine(formation_spread_std=1e-6, breakdown_vol_mult=3.0)
    warm(e)
    e.update(99.0, 100.0)
    d = e.decide(100_000.0)
    assert d.action == 'breakdown' and e.broken
    e.update(100.0, 100.0)
    assert e.decide(100_000.0) is None


def test_nan_formation_std_disables_breakdown():
    e = engine(formation_spread_std=math.nan)
    assert e.formation_spread_std is None


def test_no_trading_during_burn_in_or_when_disabled():
    e = engine(burn_in=100)
    warm(e)
    e.update(99.0, 100.0)
    assert e.decide(100_000.0) is None
    e = engine(tradeable=False)
    warm(e)
    e.update(99.0, 100.0)
    assert e.decide(100_000.0) is None


def test_anchored_exit_ignores_model_adaptation():
    live = engine(zscore_window=10)
    anchored = engine(zscore_window=10, anchor_exits=True, stop_z=3.0)
    for e in (live, anchored):
        warm(e)
        e.update(99.0, 100.0)
        fill_all(e, e.decide(100_000.0))
    live_actions, anchored_actions = [], []
    for _ in range(5):
        for e, acts in ((live, live_actions), (anchored, anchored_actions)):
            if e.status:
                e.update(98.5, 100.0)
                d = e.decide(100_000.0)
                acts.append(d.action if d else None)
    assert 'exit_long' in live_actions
    assert anchored_actions[0] == 'stop_loss'
    assert anchored.trade_log == [] and anchored._anchor is None


def test_funding_guard_skips_entries_that_would_bleed_carry():
    for guard, expect in ((5.0, None), (0.0, 'enter_long')):
        e = engine(max_funding_carry_bps=guard)
        for i in range(30):
            e.update(100.0 + 0.01 * (i % 2), 100.0)
            e.accrue_funding(0.01, 0.0)
        e.update(99.0, 100.0)
        d = e.decide(100_000.0)
        assert (d.action if d else None) == expect
    assert e.funding_skips == 0


def test_adverse_move_tracks_each_legs_worst_wick():
    e = engine()
    warm(e)
    e.update(99.0, 100.0)
    d = e.decide(100_000.0)
    assert d.action == 'enter_long'
    fill_all(e, d, 99.0)
    e.observe_range(99.5, 96.03, 103.0, 99.0)
    e.observe_range(99.2, 98.0, 101.0, 99.5)
    t = e._trade
    assert t['mae_y'] == pytest.approx((99.0 - 96.03) / 99.0)
    assert t['mae_x'] == pytest.approx(0.03)


def test_intrabar_wick_can_liquidate_when_the_close_cannot():
    from kalboy.execution import PaperBroker, PortfolioSimulator
    broker = PaperBroker(10_000.0, fee=0.0)
    broker.execute('A', 1000.0, 100.0)
    broker.execute('B', -1000.0, 100.0)
    sim = PortfolioSimulator(broker, maintenance_margin=0.01)
    closes = {'A': 100.0, 'B': 100.0}
    sim.step('t1', closes, closes, highs={'A': 100.5, 'B': 100.5}, lows={'A': 99.5, 'B': 99.5})
    assert sim.liquidations == [] and sim.max_leverage == pytest.approx(20.0)
    sim.step('t2', closes, closes, highs={'A': 100.0, 'B': 104.6}, lows={'A': 95.4, 'B': 100.0})
    assert sim.liquidations == ['t2']
    assert sim.min_margin_ratio < 0.01
    sim2 = PortfolioSimulator(PaperBroker(10_000.0, fee=0.0), maintenance_margin=0.01)
    sim2.broker.execute('A', 1000.0, 100.0); sim2.broker.execute('B', -1000.0, 100.0)
    sim2.step('t2', closes, closes)
    assert sim2.liquidations == []
