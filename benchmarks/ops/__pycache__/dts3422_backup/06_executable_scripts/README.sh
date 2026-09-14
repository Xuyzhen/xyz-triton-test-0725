#!/bin/bash
# 06_executable_scripts/README — 脚本清单
# 脚本文件已在 aisbench_kit/ 目录中，使用时从那里复制
#
# 核心脚本:
#   serve_main_allon.sh      — main 容器 B 组 serve (三开关全开)
#   serve_main_nomc2.sh     — main 容器 C 组 (关 MC2)
#   serve_main_nobalance.sh — main 容器 D 组 (关 balance)
#   serve_main_nomlapo.sh   — main 容器 E 组 (关 MLAPO)
#   serve_v023_allon.sh     — v023 容器 A 组 serve (基准)
#   ver_runner.sh           — 15 轮交叉验证 runner (断点续跑)
#   xround_runner.sh        — xround 马拉松 runner
#   resume_xround.sh        — xround 断点续跑
#   bisect_mlapo_mc2.sh     — bisect 定位 runner (草稿，需按方案重写)
#   deploy_ver.sh           — 部署脚本到远端容器
#   stop_all.sh             — 停止所有 serve
#   preflight.sh            — 预检
#   check_env.sh            — 环境检查
#   probe_positions.sh     — commit 位置探测
#   askpass.bat             — SSH 免密辅助
#
# 图表脚本:
#   gen_ver_chart.py        — 版本/开关交叉验证折线图
#   gen_xround_chart.py     — xround 折线图
#   gen_xround_chart_ci.py  — xround + CI 标记折线图
#
# CI 数据提取:
#   fetch_ci_tps.py         — 从 GitHub API 提取 CI TPS
#   fetch_ci_runs.py        — 获取 nightly runs 列表
#   parse_nightly_runs3.py — 解析 run JSON
#   locate_nightly_heads.py — 定位每日 nightly HEAD
#
# 部署方法:
#   scp -r aisbench_kit/*.sh root@<IP>:/tmp/
#   docker cp /tmp/serve_*.sh <container>:/tmp/
#   ssh root@<IP> "nohup bash /tmp/ver_runner.sh > /tmp/ver.log 2>&1 &"
