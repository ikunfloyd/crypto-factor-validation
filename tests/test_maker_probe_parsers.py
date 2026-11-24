"""
The maker probe's archive parsers and day checks (research/maker_probe.py):
read_book, read_trades, max_depth and trade_book_lags, on small archives
written here in the format the real files have (a .tar.gz holding one file of
JSON lines; a .zip holding one CSV).
"""
from __future__ import annotations

import io
import json
import tarfile
import zipfile

import numpy as np
import pytest

from research import maker_probe as probe
from research.maker_probe import ASK, BID, BUY, SELL

DAY = "2026-01-05"
T0 = probe._day_start_ms(DAY)
START, END = probe.required_coverage(DAY)
MIN = 60_000
INST = "DOGE-USDT-SWAP"


def lv(price, size):
    return [f"{price:.5f}", f"{size:.2f}", "1"]


def msg(ts, action="update", bids=(), asks=(), inst=INST):
    return {"instId": inst, "action": action, "ts": str(ts), "asks": [list(a) for a in asks],
            "bids": [list(b) for b in bids]}


def snapshot(ts, bid=0.20000, ask=0.20001, n=3, size=10.0):
    return msg(ts, "snapshot", bids=[lv(bid - k * 1e-5, size + k) for k in range(n)],
               asks=[lv(ask + k * 1e-5, size + k) for k in range(n)])


def book_file(tmp_path, name, messages):
    data = "".join(json.dumps(m) + "\n" for m in messages).encode()
    path = tmp_path / f"{name}.tar.gz"
    with tarfile.open(path, "w:gz") as tf:
        info = tarfile.TarInfo(f"{name}.data")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    return path


def trade_file(tmp_path, name, rows, header="instrument_name,trade_id,side,price,size,created_time"):
    text = header + "\n" + "".join(",".join(str(v) for v in r) + "\n" for r in rows)
    path = tmp_path / f"{name}.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(f"{name}.csv", text)
    return path


def trade(tid, ts, side="buy", price="0.20001", size="0.50", inst=INST):
    return (inst, tid, side, price, size, ts)


# --------------------------------------------------------------------------
# read_book
# --------------------------------------------------------------------------

class TestReadBook:
    def test_snapshot_and_updates_become_rows(self, tmp_path):
        f = book_file(tmp_path, "d", [snapshot(T0 + 4, n=2),
                                      msg(T0 + 14, bids=[lv(0.20000, 7.5)], asks=[lv(0.20002, 0)])])
        rep = {}
        book = probe.read_book([f], DAY, report=rep)
        # bids first, each side's first row a reset; prices in ticks, sizes in 0.01 contract
        assert book.side.tolist() == [BID, BID, ASK, ASK, BID, ASK]
        assert book.reset.tolist() == [True, False, True, False, False, False]
        assert book.price.tolist() == [20000, 19999, 20001, 20002, 20000, 20002]
        assert book.size.tolist() == [1000, 1100, 1000, 1100, 750, 0]
        assert book.msg.tolist() == [0, 0, 0, 0, 1, 1]
        assert book.ts_ms.tolist() == [T0 + 4] * 4 + [T0 + 14] * 2
        assert (book.tick_size, book.size_unit) == (probe.TICK_SIZE, probe.SIZE_UNIT)
        assert rep["messages"] == 2 and rep["snapshots"] == 1 and rep["ended_early"] is True
        assert book.end_ms == T0 + 15                          # the data ended inside the coverage

    def test_crops_to_the_coverage_and_stops_past_it(self, tmp_path):
        d = book_file(tmp_path, "d", [msg(T0 - 5, bids=[lv(0.2, 1)]), snapshot(T0 + 4)])
        d1 = book_file(tmp_path, "d1", [snapshot(END - 2), msg(END - 1, bids=[lv(0.2, 2)]), msg(END, bids=[lv(0.2, 3)]),
                                        {"broken": "past the end, never parsed"}])
        book = probe.read_book([d, d1], DAY)
        assert book.ts_ms[0] == T0 + 4 and book.ts_ms[-1] == END - 1
        assert (book.start_ms, book.end_ms) == (START, END)

    def test_a_timestamp_that_goes_back_is_clamped_and_counted(self, tmp_path):
        f = book_file(tmp_path, "d", [snapshot(T0 + 20), msg(T0 + 15, bids=[lv(0.2, 2)]), msg(T0 + 30, asks=[lv(0.20001, 3)])])
        rep = {}
        book = probe.read_book([f], DAY, report=rep)
        assert book.ts_ms.tolist()[-2:] == [T0 + 20, T0 + 30]
        assert rep["clamped"] == 1 and rep["max_clamp_ms"] == 5

    def test_a_long_pause_is_a_gap_to_the_next_snapshot(self, tmp_path):
        gap_from, back = T0 + 1000, T0 + 1000 + probe.MAX_SILENCE_MS
        f = book_file(tmp_path, "d", [snapshot(T0 + 4), msg(gap_from, bids=[lv(0.2, 2)]),
                                      msg(back, bids=[lv(0.2, 3)]), msg(back + 1000, bids=[lv(0.2, 4)]),
                                      snapshot(back + 2000), msg(back + 2010, asks=[lv(0.20001, 1)])])
        rep = {}
        book = probe.read_book([f], DAY, report=rep)
        assert book.gaps.tolist() == [[gap_from, back + 2000]]
        assert rep["pauses_as_gaps"] == [[gap_from, back]] and rep["longest_pause_ms"] == probe.MAX_SILENCE_MS

    def test_a_shorter_pause_is_not_a_gap(self, tmp_path):
        f = book_file(tmp_path, "d", [snapshot(T0 + 4), msg(T0 + 4 + probe.MAX_SILENCE_MS - 1, bids=[lv(0.2, 2)])])
        assert probe.read_book([f], DAY).gaps.size == 0

    def test_a_pause_with_no_snapshot_after_it_ends_the_data(self, tmp_path):
        f = book_file(tmp_path, "d", [snapshot(T0 + 4), msg(T0 + 100, bids=[lv(0.2, 2)]),
                                      msg(T0 + 100 + probe.MAX_SILENCE_MS, bids=[lv(0.2, 3)])])
        book = probe.read_book([f], DAY)
        assert book.end_ms == T0 + 101 and book.ts_ms[-1] == T0 + 100 and book.gaps.size == 0

    def test_an_empty_side_in_a_snapshot_is_a_reset_row_of_size_zero(self, tmp_path):
        f = book_file(tmp_path, "d", [snapshot(T0 + 4), msg(T0 + 14, "snapshot", bids=[lv(0.2, 1)], asks=[])])
        book = probe.read_book([f], DAY)
        last = np.flatnonzero(book.msg == 1)
        assert book.reset[last].tolist() == [True, True] and book.size[last].tolist() == [100, 0]
        assert book.side[last].tolist() == [BID, ASK] and book.price[last[1]] > 0

    def test_other_instruments_are_skipped(self, tmp_path):
        f = book_file(tmp_path, "d", [snapshot(T0 + 4), msg(T0 + 5, bids=[lv(0.3, 1)], inst="DOGE-USDT")])
        rep = {}
        assert len(probe.read_book([f], DAY, report=rep)) == 6 and rep["other_instruments"] == 1

    @pytest.mark.parametrize("bad", [msg(T0 + 14, bids=[["0.200005", "1.00", "1"]]),
                                     msg(T0 + 14, bids=[["0.20000", "1.005", "1"]]),
                                     msg(T0 + 14, action="partial", bids=[lv(0.2, 1)])])
    def test_refuses_values_off_the_grid_and_unknown_actions(self, tmp_path, bad):
        f = book_file(tmp_path, "d", [snapshot(T0 + 4), bad])
        with pytest.raises(ValueError):
            probe.read_book([f], DAY)

    def test_refuses_an_archive_with_two_files(self, tmp_path):
        path = tmp_path / "two.tar.gz"
        with tarfile.open(path, "w:gz") as tf:
            for name in ("a.data", "b.data"):
                data = (json.dumps(snapshot(T0 + 4)) + "\n").encode()
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))
        with pytest.raises(ValueError, match="more than one file"):
            probe.read_book([path], DAY)


# --------------------------------------------------------------------------
# read_trades
# --------------------------------------------------------------------------

class TestReadTrades:
    def test_joins_dedupes_crops_and_sorts(self, tmp_path):
        d = trade_file(tmp_path, "d", [trade(1, T0 - 1), trade(2, T0 + 5, "sell", "0.20000", "1.25"), trade(3, T0 + 5)])
        d1 = trade_file(tmp_path, "d1", [trade(3, T0 + 5), trade(5, T0 + 4), trade(4, T0 + 4), trade(6, END)])
        rep = {}
        tr = probe.read_trades([d, d1], DAY, report=rep)
        assert tr.ts_ms.tolist() == [T0 + 4, T0 + 4, T0 + 5, T0 + 5]      # (ts, id) order: ids 4, 5, 2, 3
        assert tr.price.tolist() == [20001, 20001, 20000, 20001]
        assert tr.size.tolist() == [50, 50, 125, 50]
        assert tr.aggressor.tolist() == [BUY, BUY, SELL, BUY]
        assert (tr.start_ms, tr.end_ms) == (START, END) and tr.gaps.size == 0
        assert rep["duplicate_rows"] == 1 and rep["id_jumps"] == 0 and rep["ids_out_of_time_order"] == 1

    def test_a_jump_in_trade_id_is_a_gap(self, tmp_path):
        d = trade_file(tmp_path, "d", [trade(10, T0 + 100), trade(13, T0 + 900), trade(14, T0 + 950), trade(15, END + 1)])
        tr = probe.read_trades([d], DAY)
        assert tr.gaps.tolist() == [[T0 + 100, T0 + 901]]

    def test_data_ending_inside_the_coverage_lowers_its_end(self, tmp_path):
        d = trade_file(tmp_path, "d", [trade(1, T0 + 10), trade(2, T0 + 20)])
        assert probe.read_trades([d], DAY).end_ms == T0 + 21

    def test_a_repeated_id_with_other_contents_is_refused(self, tmp_path):
        d = trade_file(tmp_path, "d", [trade(1, T0 + 10)])
        d1 = trade_file(tmp_path, "d1", [trade(1, T0 + 10, size="0.70")])
        with pytest.raises(ValueError, match="repeats with different contents"):
            probe.read_trades([d, d1], DAY)

    @pytest.mark.parametrize("bad, why", [((INST, 1, "buy", "0.200005", "0.5", T0 + 1), "not a whole multiple"),
                                          ((INST, 1, "buy", "0.2", "0.505", T0 + 1), "not a whole multiple"),
                                          ((INST, 1, "short", "0.2", "0.5", T0 + 1), "side")])
    def test_refuses_bad_values(self, tmp_path, bad, why):
        with pytest.raises(ValueError, match=why):
            probe.read_trades([trade_file(tmp_path, "d", [bad])], DAY)

    def test_refuses_other_columns(self, tmp_path):
        d = trade_file(tmp_path, "d", [trade(1, T0 + 1)], header="instrument_name,trade_id,side,px,size,created_time")
        with pytest.raises(ValueError, match="columns"):
            probe.read_trades([d], DAY)


# --------------------------------------------------------------------------
# day checks
# --------------------------------------------------------------------------

def test_max_depth_counts_levels_after_each_message(tmp_path):
    f = book_file(tmp_path, "d", [snapshot(T0 + 4, n=3), msg(T0 + 14, bids=[lv(0.19990, 1)]),
                                  msg(T0 + 24, bids=[lv(0.19990, 0), lv(0.19999, 0)])])
    assert probe.max_depth(probe.read_book([f], DAY)) == (4, 3)


def test_trade_book_lags_matches_a_trade_with_the_decrease_it_causes(tmp_path):
    book = probe.read_book([book_file(tmp_path, "d", [
        snapshot(T0 + 4, n=2),
        msg(T0 + 14, asks=[lv(0.20001, 9.5)]),           # the 0.50 bought at 0.20001 shows 4 ms later
        msg(T0 + 24, bids=[lv(0.20000, 8.0)]),           # 2.00 sold at 0.20000, 14 ms later
    ])], DAY)
    tr = probe.read_trades([trade_file(tmp_path, "t", [trade(1, T0 + 10), trade(2, T0 + 10, "sell", "0.20000", "2.00"),
                                                       trade(3, END + 1)])], DAY)
    out = probe.trade_book_lags(book, tr)
    assert out["matched_share"] == 1.0 and out["lag_ms_median"] == 9.0
    assert out["lag_within_10ms_share"] == 0.5 and out["lag_within_20ms_share"] == 1.0
    assert out["decrease_equals_size_share"] == 1.0 and out["decrease_over_size_median"] == 1.0
