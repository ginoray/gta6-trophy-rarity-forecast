# GTA VI trophy rarity forecast — gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/ginoray/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/ginoray/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A forecast, for every PlayStation 5 trophy of Grand Theft Auto VI, of the
percent of Exophase-tracked players who will have earned it one week after
launch (2026-11-26T00:00:00Z), as five quantiles per trophy, for the
[make.fun bounty](https://make.fun/bounty/gta6-trophy-rarity-forecast-01).

`main.py` reads each case file in the input directory — the house's recorded
history of the game's Exophase trophy page — and writes one forecast file per
case. A printed rate is earners / players_tracked, so the two are forecast
separately over the recent window of distinct page states and divided: the
tracked crowd's growth fades over the week while earners keep accumulating,
and fitting the launch's early dilution crash as if it were the trend is how
a forecast goes wrong. Slopes are shrunk towards the pack, the median is
damped towards the last printed rate, and the spread is calibrated to cover
the true rate about nine times in ten.

Run it the way the grader does:

    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py

Python 3 standard library only, deterministic, no network.
