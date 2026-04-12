"""
根据在职技师与 appointments 表已有预约，计算「最早能开始上钟」的技师与时间。
区间规则与预约重叠逻辑一致：半开区间 [起, 起+duration)；首尾相接不算重叠。
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Any

from langchain.tools import tool

from config.db import connection

# 默认营业窗口（可按门店调整）；新预约须在 [work_start, work_end - service_duration] 内开始
DEFAULT_WORK_START = time(9, 0)
DEFAULT_WORK_END = time(22, 0)
SLOT_STEP_MINUTES = 15


def _to_time(val: Any) -> time:
    if isinstance(val, time):
        return val
    if isinstance(val, timedelta):
        secs = int(val.total_seconds()) % (24 * 3600)
        return time(secs // 3600, (secs % 3600) // 60, secs % 60)
    if isinstance(val, str):
        parts = val.strip().split(":")
        h, m = int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
        s = int(parts[2]) if len(parts) > 2 else 0
        return time(h, m, s)
    raise TypeError(f"unsupported time type: {type(val)}")


def _to_date(val: Any) -> date:
    if isinstance(val, date) and not isinstance(val, datetime):
        return val
    if isinstance(val, datetime):
        return val.date()
    if isinstance(val, str):
        return datetime.strptime(val[:10], "%Y-%m-%d").date()
    raise TypeError(f"unsupported date type: {type(val)}")


def _combine(d: date, t: time) -> datetime:
    return datetime.combine(d, t)


def _parse_search_start(s: str | None) -> datetime:
    if not s or not str(s).strip():
        return datetime.now()
    raw = str(s).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    raise ValueError(f"无法解析时间: {s}")


def _fetch_active_therapists(cursor) -> list[dict[str, Any]]:
    try:
        cursor.execute(
            """
            SELECT id, COALESCE(real_name, name, '') AS display_name
            FROM therapists
            WHERE `status` = '在职'
            """
        )
        rows = cursor.fetchall()
    except Exception:
        cursor.execute("SELECT id FROM therapists WHERE `status` = '在职'")
        rows = cursor.fetchall()
        return [
            {"id": r["id"], "display_name": f"技师ID {r['id']}"} for r in rows
        ]
    return list(rows)


def _fetch_appointments_for_day(cursor, therapist_id: int, d: date) -> list[tuple[datetime, datetime]]:
    cursor.execute(
        """
        SELECT appointment_date, appointment_time, COALESCE(duration, 90) AS dur, status
        FROM appointments
        WHERE therapist_id = %s
          AND appointment_date = %s
          AND COALESCE(status, '') NOT IN ('已取消')
        ORDER BY appointment_time
        """,
        (therapist_id, d),
    )
    rows = cursor.fetchall()
    blocks: list[tuple[datetime, datetime]] = []
    for r in rows:
        dd = _to_date(r["appointment_date"])
        tt = _to_time(r["appointment_time"])
        dur = int(r["dur"] or 90)
        start = _combine(dd, tt)
        end = start + timedelta(minutes=dur)
        blocks.append((start, end))
    return _merge_intervals(blocks)


def _merge_intervals(intervals: list[tuple[datetime, datetime]]) -> list[tuple[datetime, datetime]]:
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    out: list[tuple[datetime, datetime]] = [intervals[0]]
    for s, e in intervals[1:]:
        ps, pe = out[-1]
        if s <= pe:
            out[-1] = (ps, max(pe, e))
        else:
            out.append((s, e))
    return out


def _intervals_overlap(a0: datetime, a1: datetime, b0: datetime, b1: datetime) -> bool:
    return a0 < b1 and b0 < a1


def _first_slot_in_day(
    day: date,
    blocks: list[tuple[datetime, datetime]],
    service_minutes: int,
    work_start: time,
    work_end: time,
    earliest_start: datetime,
) -> datetime | None:
    """在指定自然日内，从 earliest_start 起按步长找第一个可插入的 [t, t+service)。"""
    day_start = _combine(day, work_start)
    day_close = _combine(day, work_end)
    latest_start = day_close - timedelta(minutes=service_minutes)
    if latest_start < day_start:
        return None

    t = max(day_start, earliest_start.replace(second=0, microsecond=0))
    if t.date() != day:
        return None
    # 对齐到步长分钟
    if t.minute % SLOT_STEP_MINUTES or t.second:
        add = (SLOT_STEP_MINUTES - (t.minute % SLOT_STEP_MINUTES)) % SLOT_STEP_MINUTES
        if add == 0:
            add = SLOT_STEP_MINUTES
        t = t + timedelta(minutes=add)
        t = t.replace(second=0, microsecond=0)

    while t <= latest_start:
        te = t + timedelta(minutes=service_minutes)
        if te > day_close:
            break
        ok = True
        for bs, be in blocks:
            if _intervals_overlap(t, te, bs, be):
                ok = False
                break
        if ok:
            return t
        t += timedelta(minutes=SLOT_STEP_MINUTES)
    return None


def _find_earliest_for_therapist(
    cursor,
    therapist_id: int,
    display_name: str,
    service_minutes: int,
    search_from: datetime,
    horizon_days: int,
    work_start: time,
    work_end: time,
) -> tuple[datetime, int, str] | None:
    d0 = search_from.date()
    for offset in range(horizon_days):
        d = d0 + timedelta(days=offset)
        blocks = _fetch_appointments_for_day(cursor, therapist_id, d)
        earliest = search_from if offset == 0 else _combine(d, work_start)
        slot = _first_slot_in_day(
            d, blocks, service_minutes, work_start, work_end, earliest
        )
        if slot:
            return (slot, therapist_id, display_name or f"技师ID {therapist_id}")
    return None


@tool
def get_earliest_available_therapist(
    service_duration_minutes: int = 90,
    start_search_from: str | None = None,
    horizon_days: int = 7,
    work_start_hour: int = 12,
    work_end_hour: int = 23,
) -> str:
    """
    查询在当前规则下「最快可以上钟」的在职技师是谁、以及可开始的具体日期时间。

    规则概要：仅考虑 status 为「在职」的技师；在默认营业时间内按固定步长（15 分钟）搜索；
    已有预约占用 [开始, 开始+duration) 区间，新单不得与该区间重叠（首尾相接允许）。

    参数：
        service_duration_minutes: 本次服务持续分钟数（与预约占用时长一致，默认 90）
        start_search_from: 从何时开始找空档，格式 YYYY-MM-DD 或 YYYY-MM-DD HH:MM:SS；不传则使用当前时间
        horizon_days: 从搜索起点日期起向后最多看多少天（默认 7）
        work_start_hour / work_end_hour: 每日营业开始/结束整点（默认 9–22），结束时刻前须能完整排下整段服务
    返回：
        可读字符串：最早可上钟的技师与时间点；若无则说明原因（已排满、无在职技师等）。
        上层应答时须直接引用本工具结果；未调用本工具前不得向顾客编造可约时间。
    """
    print(
        "get_earliest_available_therapist: "
        f"duration={service_duration_minutes}, from={start_search_from}, days={horizon_days}"
    )
    try:
        search_from = _parse_search_start(start_search_from)
    except ValueError as e:
        return f"参数错误：{e}"

    ws = time(work_start_hour, 0)
    we = time(work_end_hour, 0)
    if service_duration_minutes <= 0:
        return "服务时长须为正整数分钟。"
    if horizon_days < 1:
        return "horizon_days 至少为 1。"

    try:
        with connection.cursor() as cursor:
            therapists = _fetch_active_therapists(cursor)
            if not therapists:
                return "当前没有在职技师，无法安排上钟。"

            best: tuple[datetime, int, str] | None = None
            for row in therapists:
                tid = int(row["id"])
                name = (row.get("display_name") or "").strip() or f"技师ID {tid}"
                got = _find_earliest_for_therapist(
                    cursor,
                    tid,
                    name,
                    service_duration_minutes,
                    search_from,
                    horizon_days,
                    ws,
                    we,
                )
                if got is None:
                    continue
                if best is None or got[0] < best[0]:
                    best = got

            if best is None:
                return (
                    f"在未来 {horizon_days} 天、每日 {work_start_hour}:00–{work_end_hour}:00、"
                    f"服务时长 {service_duration_minutes} 分钟的条件下，未找到可排班的在职技师（可能均已排满）。"
                )

            slot_dt, tid, label = best
            return (
                f"当前最快可上钟：{label}（技师ID {tid}）。"
                f"最早可开始时间：{slot_dt.strftime('%Y-%m-%d %H:%M')}（按 {service_duration_minutes} 分钟服务、"
                f"营业 {work_start_hour}:00–{work_end_hour}:00 测算）。"
            )
    except Exception as e:
        return f"查询失败：{e}"
