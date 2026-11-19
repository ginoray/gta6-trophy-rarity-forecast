"""Forecast GTA VI trophy earn rates on Exophase at a case's target time.

Entry for the make.fun bounty "GTA VI trophy rarity forecast 01"
(gta6-trophy-rarity-forecast-01). The program is run once per case file found
in the input directory and writes one forecast per case: for every trophy,
five quantiles (5/25/50/75/95%) of its earn rate -- the percent of tracked
players who will hold it -- at the case's target. Scoring is a pinball loss
on the logit scale, so all modelling happens in logit space.

How the forecast is built
-------------------------
A printed rate is earners / players_tracked, and the two parts move
differently over a launch week: the tracked crowd rushes in early and then
slows, while a trophy's earners only accumulate as players reach it. The
launch's first hours are a dilution crash -- the crowd grows faster than
anyone can finish anything -- and after it the regime flips: the crowd's
growth fades while earners keep climbing. A single trend fitted over the
whole history mixes the two regimes and extrapolates the crash, so every
fit below uses only the recent window of distinct page states (repeated
reads of an unrefreshed page are collapsed first), the last two days before
the checkpoint, widening only when that holds too few states.

Inside the window, three fits are made, all weighted towards the big later
states whose printed rates carry the least rounding noise:

* players_tracked: a straight line of log(players) against log(1 + hours
  since release), read at the target hour and kept within sane bounds of
  the last count.
* earners, per trophy: the same line for log(earners + 0.5); divided by the
  player forecast it gives one candidate rate.
* the rate itself: a line of logit(rate) against the same clock gives a
  second candidate.

Each slope is shrunk part-way to the median slope across trophies, which
keeps one noisy trophy from sprinting or stalling on its own. The median
forecast blends the two candidates in logit space and is then damped
towards the last printed rate, because over a few days the honest move is
usually a fraction of what a raw trend line promises. The spread around
the median combines a floor, a term growing with the square root of the
horizon, the rate fit's residuals and the player count's uncertainty, so
the 5%-95% band aims to cover the true rate about nine times in ten.

When a case's history holds fewer than three distinct states there is no
trend to fit: the forecast falls back to the last printed rate, tilted
down in logit space when the tracked crowd is still small (an early
refresh mostly adds beginners, which dilutes every rate), with a wide
spread.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone

LEVELS = (0.05, 0.25, 0.50, 0.75, 0.95)
Z = (-1.644854, -0.674490, 0.0, 0.674490, 1.644854)

RATE_LO = 0.01    # the scorer clamps rates into [0.01, 99.99] before logit
RATE_HI = 99.99
WINDOW = timedelta(hours=48)


def clamp_rate(x):
    return min(RATE_HI, max(RATE_LO, x))


def logit(rate):
    p = clamp_rate(rate) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(y):
    if y >= 0:
        return 100.0 / (1.0 + math.exp(-y))
    e = math.exp(y)
    return 100.0 * e / (1.0 + e)


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def wls_line(xs, ys, ws):
    """Weighted least squares y ~ a + b*x; returns (a, b, residual_sd)."""
    sw = sum(ws)
    if sw <= 0:
        return ys[0], 0.0, 0.0
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    my = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        return my, 0.0, 0.0
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws))
    b = sxy / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    dof = max(1, len(xs) - 2)
    sd = math.sqrt(sum(w * r * r for r, w in zip(resid, ws)) / max(sw, 1e-9) * len(xs) / dof)
    return a, b, sd


def median(values):
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def distinct_states(history):
    """Collapse runs of reads that show the identical page into one state."""
    states = []
    for read in history:
        key = (read.get("players_tracked"), tuple(read.get("rates") or ()))
        if states and states[-1][0] == key:
            continue
        states.append((key, read))
    return [read for _, read in states]


def forecast_case(case):
    trophies = case["trophies"]
    history = case.get("history") or []
    checkpoint = parse_time(case["at"])
    target = parse_time(case["target"])
    horizon_days = max(0.0, (target - checkpoint).total_seconds()) / 86400.0

    release = parse_time(case["game"]["release_date"] + "T00:00:00Z") \
        if case.get("game", {}).get("release_date") else None
    if release is None:
        release = parse_time(history[0]["at"]) if history else checkpoint

    def log_clock(moment):
        hours = max(0.0, (moment - release).total_seconds()) / 3600.0
        return math.log1p(hours)

    states = distinct_states(history)
    last = states[-1] if states else None
    last_rates = (last.get("rates") if last else None) or []
    last_players = (last.get("players_tracked") if last else None) or 0

    # the recent window of distinct states; widen if it holds too few
    win = [s for s in states if checkpoint - parse_time(s["at"]) <= WINDOW]
    if len(win) < 4:
        win = states
    win_x = [log_clock(parse_time(s["at"])) for s in win]
    target_x = log_clock(target)

    # --- the denominator: players tracked at the target -------------------
    players_med = float(last_players) if last_players else 1000.0
    players_sd = 0.35
    n_pts = [(x, s["players_tracked"]) for x, s in zip(win_x, win)
             if s.get("players_tracked")]
    if len(n_pts) >= 3:
        xs = [p[0] for p in n_pts]
        ys = [math.log(p[1]) for p in n_pts]
        ws = [math.sqrt(p[1]) for p in n_pts]
        a, b, sd = wls_line(xs, ys, ws)
        fitted = math.exp(a + b * target_x)
        players_med = min(max(fitted, players_med * 1.02), players_med * 2.6)
        players_sd = min(0.55, max(0.12, 0.10 + sd
                                   + 0.08 * abs(b) * max(0.0, target_x - xs[-1])))

    # --- per-trophy points inside the window -------------------------------
    per_trophy = []
    for idx, trophy in enumerate(trophies):
        pts = []  # (x, earners, weight, rate)
        for x, s in zip(win_x, win):
            rates = s.get("rates") or []
            rate = rates[idx] if idx < len(rates) else None
            players = s.get("players_tracked") or 0
            if rate is None or players <= 0:
                continue
            pts.append((x, rate * players / 100.0, math.sqrt(players), rate))
        last_rate = None
        if idx < len(last_rates) and last_rates[idx] is not None:
            last_rate = last_rates[idx]
        elif pts:
            last_rate = pts[-1][3]
        per_trophy.append((trophy, pts, last_rate))

    def fit_slopes(kind):
        out = []
        for _, pts, _ in per_trophy:
            if len(pts) < 3:
                out.append(None)
                continue
            xs = [p[0] for p in pts]
            ws = [p[2] for p in pts]
            if kind == "earners":
                if not any(e > 0 for _, e, _, _ in pts):
                    out.append(None)
                    continue
                ys = [math.log(p[1] + 0.5) for p in pts]
            else:
                ys = [logit(p[3]) for p in pts]
            _, b, _ = wls_line(xs, ys, ws)
            out.append(b)
        return out

    e_slopes = fit_slopes("earners")
    r_slopes = fit_slopes("rate")
    e_pack = median([b for b in e_slopes if b is not None]) if any(b is not None for b in e_slopes) else 0.0
    r_pack = median([b for b in r_slopes if b is not None]) if any(b is not None for b in r_slopes) else 0.0

    rows = []
    for (trophy, pts, last_rate), e_b, r_b in zip(per_trophy, e_slopes, r_slopes):
        med_rate = None
        resid = 0.0
        if len(pts) >= 3 and last_rate is not None:
            xs = [p[0] for p in pts]
            ws = [p[2] for p in pts]
            cands = []
            if e_b is not None:
                ys = [math.log(p[1] + 0.5) for p in pts]
                a, _, _ = wls_line(xs, ys, ws)
                b = 0.7 * e_b + 0.3 * e_pack
                earners = max(0.0, math.exp(a + b * target_x) - 0.5)
                if players_med > 0:
                    cands.append(clamp_rate(100.0 * earners / players_med))
            if r_b is not None:
                ys = [logit(p[3]) for p in pts]
                a, _, resid = wls_line(xs, ys, ws)
                b = 0.7 * r_b + 0.3 * r_pack
                cands.append(clamp_rate(inv_logit(a + b * target_x)))
            if cands:
                blend = sum(logit(c) for c in cands) / len(cands)
                # damped towards the last rate, and damped harder upwards
                # than downwards: a fresh wave of syncs can dilute a rate
                # overnight, but earners only ever accumulate gradually
                raw_move = blend - logit(last_rate)
                damp = 0.72 if raw_move > 0 else 0.85
                centre = logit(last_rate) + damp * raw_move
                move = centre - logit(last_rate)
                if abs(move) > 1.5:
                    centre = logit(last_rate) + math.copysign(1.5, move)
                med_rate = inv_logit(centre)
        if med_rate is None:
            # frozen or empty history: last rate, diluted if the crowd is
            # still small enough that one refresh can swamp it
            base = last_rate if last_rate is not None else \
                {"platinum": 0.05, "gold": 0.2, "silver": 0.5}.get(trophy.get("tier"), 1.0)
            centre = logit(base)
            if last_players and last_players < 20000 and len(states) <= 2:
                centre -= 0.55
            med_rate = inv_logit(centre)
            resid = 0.55

        sigma = math.sqrt(0.22 ** 2 + (0.27 * math.sqrt(horizon_days)) ** 2
                          + (0.7 * resid) ** 2 + (0.8 * players_sd) ** 2)
        if med_rate > 99.0:
            sigma = min(sigma, 0.35)     # a rate pinned to the ceiling stays there
        sigma = min(1.6, max(0.40, sigma))
        centre = logit(med_rate)
        q = [clamp_rate(inv_logit(centre + z * sigma)) for z in Z]
        q = [round(v, 4) for v in q]
        for i in range(1, len(q)):
            q[i] = max(q[i], q[i - 1])
        rows.append({"id": trophy["id"], "q": q})

    rows.sort(key=lambda r: r["id"])
    return {"trophies": rows}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    if not names:
        print(f"no case files in {in_dir}", file=sys.stderr)
        return 1
    for name in names:
        with open(os.path.join(in_dir, name), encoding="utf-8") as fh:
            case = json.load(fh)
        result = forecast_case(case)
        out_name = name[:-3] + ".out"
        with open(os.path.join(out_dir, out_name), "w", encoding="utf-8") as fh:
            json.dump(result, fh, separators=(",", ":"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
