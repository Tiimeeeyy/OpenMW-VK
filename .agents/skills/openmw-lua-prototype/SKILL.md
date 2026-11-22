---
name: openmw-lua-prototype
description: Use when prototyping new gameplay mechanics, HUD tweaks, or player abilities in OpenMW using Lua scripts or exposing new C++ bindings in mwlua.
---

# OpenMW Lua-First Workflow

1. Check `files/data/scripts/` and `apps/openmw/mwlua/` to see if the needed data or event is already exposed to OpenMW's Lua API.
2. If it is already exposed, implement the feature as a self-contained Lua script rather than modifying core C++ in `mwmechanics/`.
3. If a new C++ binding is required:
   - Keep the binding minimal and sandbox-safe inside `apps/openmw/mwlua/`.
   - Expose read-only state first; only add mutators if explicitly needed.
   - Rebuild with `cmake --build build --target openmw -j$(nproc)` and verify no Lua API regressions.
