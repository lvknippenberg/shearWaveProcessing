"""Manual passive-SWE study reading: M-lines on buffers 1/3/4, hand slopes on five views.

``store``   files, hashes and the per-folder state machine (resumable, one writer per file)
``frames``  ECG-synchronised B-mode frames of buffers 1, 3 and 4
``worker``  unattended stages: burst detection and the five space-times per window
``line_gui`` / ``slope_gui``  the two interactive editors

Driver: ``scripts/passive_manual.py``; workflow and file layout: ``docs/passive_manual.md``.
"""
