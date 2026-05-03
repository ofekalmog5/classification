# Documentation Index

Topic-specific deep-dives for the classification project. Start at
[../README.md](../README.md) for the high-level overview.

| Doc | What's inside |
|-----|---------------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | God nodes, process diagram, two-step pipeline, batch mode, AI extraction flow, web-app state, extension points |
| [API_REFERENCE.md](API_REFERENCE.md) | Every FastAPI endpoint (main + MEA calibration tool) |
| [AI_FEATURE_EXTRACTION.md](AI_FEATURE_EXTRACTION.md) | OWLv2 + SAM 2/3, prompts, pre-filters, output organisation, model fallback chain |
| [MEA_CALIBRATION_TOOL.md](MEA_CALIBRATION_TOOL.md) | Profile schema (v2), workflow, profile path, merge rules |
| [GPU_ACCELERATION.md](GPU_ACCELERATION.md) | Engine probe order, CuPy install, FAISS, frozen-EXE notes |
| [TILING.md](TILING.md) | Tile mode, `suggest_tile_size`, worker pool, memory math |

Operational / deployment docs live at the project root:

- [../README.md](../README.md) — project overview
- [../RUNNING_GUIDE.md](../RUNNING_GUIDE.md) — running, troubleshooting
- [../STANDALONE_DEPLOYMENT.md](../STANDALONE_DEPLOYMENT.md) — three deployment methods
- [../SHADOW_DETECTION_FEATURE.md](../SHADOW_DETECTION_FEATURE.md) — shadow → adjacent material inference

Per-component READMEs:

- [../backend/README.md](../backend/README.md)
- [../web_app/README.md](../web_app/README.md)
- [../mea_calibration_tool/README.md](../mea_calibration_tool/README.md)
- [../offline_installer/README.md](../offline_installer/README.md)

Historical (kept for context, marked as such at the top of each):

- [../REFACTORING_SUMMARY.md](../REFACTORING_SUMMARY.md) — original two-step split
- [../FIXES_APPLIED.md](../FIXES_APPLIED.md) — March 2026 PROJ / CRS / dead-code fixes
- [../RASTERIZE_DEBUG.md](../RASTERIZE_DEBUG.md) — Hebrew checklist for "0 pixels rasterized"
