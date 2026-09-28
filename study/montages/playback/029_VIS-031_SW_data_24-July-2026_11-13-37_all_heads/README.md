# C000000029, 2nd SW acquisition: buffer 3 with every possible head (2026-09-28)

Made by `study/analysis/all_heads_gifs.py` to check the unwrap visually. `head_XX.gif` plays stored
slots XX, XX+1, ..., XX-1 (40 ms per frame, ~real time); `all_heads_grid.gif` shows all 32 at once.
The VERSION-2 unwrap chose head 11 (combined margin 4.7; continuity, buffer-1 and expected-motion
each pick 11 on their own).

`links.csv`: weakest frame-to-frame link inside each sequence (low-pass de-meaned log envelope,
relative to the median link; the GIF loop point excluded). Head 11: +0.60 (ordinary motion); every
other head contains the stored 10 -> 11 link at -0.81 (the wrap: newest to oldest frame).

The last 32 loop triggers are exactly 39.4 ms apart (no dropped or stalled frames). The clip spans
1.26 s = ~1.5 beats (RR 865 ms): chronological frames run from R+374 ms through the next R-peak to
R+631 ms, so every GIF loop jumps back ~257 ms of the cardiac cycle at the loop point. That jump is
inherent to looping a clip that is not a whole number of beats, not an unwrap error.
