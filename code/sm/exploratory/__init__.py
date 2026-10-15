"""Exploratory analyses that decide no registered hypothesis.

``sm.exploratory.cpu_lines`` holds the pure functions of the zero-GPU lines E1, E2, N3, N5 and N8
(docs/EXPLORATORY_CPU_LINES.md, rules committed before any statistic was computed); ``sm.exploratory.run_cpu``
reads the accepted artifacts on the laptop and writes results/exploratory/<base>/cpu/. Nothing here edits or
replaces the frozen analysis module (code/sm/analysis), which is imported only.
"""
