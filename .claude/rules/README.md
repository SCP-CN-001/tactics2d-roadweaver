# Tactics2D 代码规范

本目录记录 Tactics2D 项目的代码规范（源自 `tactics2d/.claude/`）。本仓库是 Tactics2D 的衍生仓库，同一作者团队，沿用同一套规范。

本目录下的 `.md` 规则文件由 Claude Code 自动加载（`.claude/rules/` 无需在 settings 中注册）。tactics2d 中的 JSON 规则已翻译为 markdown 形式，便于直接作为编码指引。

## 规则清单

| 文件 | 主题 | 生效范围 |
|------|------|----------|
| `python.md` | Python 文件头（版权头 + SPDX + docstring） | `**/*.py` |

## 说明

- **作者身份**：`Tactics2D Authors`，GPLv3 许可（`SPDX-License-Identifier: GPL-3.0-or-later`）。
- **不做 pytest / CI**：本仓库是研究型 repo，不写 pytest 测试、不配置 CI；tactics2d 的 pytest 风格规范与 CI 流程（PR 描述生成、GitHub Actions 同步）不适用，故不移植。
- **强制方式**：tactics2d 侧由 PostToolUse hook（`check_python_header.py` 等）强制校验；本仓库以规则文件作为引导，未部署 hook。
- **规则载体**：tactics2d 用 JSON 规则 + hook 脚本；此处为 markdown 规则（Claude Code 自动加载）。

## 变更记录

- [1.0] - 2026-07-31：首次记录，从 tactics2d/.claude 移植（含 python 文件头、pytest、changelog 三块）。
- [1.1] - 2026-07-31：按用户要求移除 pytest 规范（本仓库不需要 pytest / CI），保留 python 文件头与 changelog。
- [1.2] - 2026-07-31：按用户要求移除 changelog 规范（本仓库不需要 CHANGELOG.md），保留 python 文件头。
