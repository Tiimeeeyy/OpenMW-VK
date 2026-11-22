# OpenMW Engine Development (CachyOS / Antigravity CLI)

## Environment & Tooling
- **OS / Toolchain:** CachyOS (Arch-based), GCC/Ninja, CMake out-of-source build in `build/`.
- **LSP / Indexing:** `compile_commands.json` is symlinked at the repo root—prefer symbol definitions over blind text grep when available.
- **Test Data:** OpenMW `example-suite` (configured via `~/.config/openmw/openmw.cfg`).

## Repository Architecture
- `apps/openmw/`: Main game engine executable.
  - `mwrender/`: OpenSceneGraph (OSG) rendering, shaders, shadows, water, post-processing.
  - `mwphysics/`: Bullet physics (double precision), collision, raycasting, character movement.
  - `mwmechanics/`: Combat, AI, magic, stats, actor state.
  - `mwlua/`: C++ to LuaJIT bindings and sandboxed Lua API.
  - `mwgui/`: MyGUI windows, HUD, menus.
  - `mwscript/`: Legacy vanilla MWScript interpreter.
- `components/`: Shared C++ libraries (`esm3`/`esm4` parsers, `nif`/`nifosg` loaders, `sceneutil`, `shader`, `lua`, `settings`, `vfs`).
- `files/`: GLSL shaders (`shaders/`), built-in Lua scripts (`data/scripts/`), and MyGUI layouts (`data/mygui/`).
- **Off-limits:** Never read, search, or modify `extern/` or `build/` unless explicitly instructed.

## Build, Test & Launch Commands
- **Incremental engine build (default):**
  `cmake --build build --target openmw -j$(nproc)`
- **Build & run unit tests:**
  `cmake --build build --target openmw_test_suite -j$(nproc) && ctest --test-dir build --output-on-failure`
- **Format modified C++ files:**
  `clang-format -i <file>`
- **Launch engine:**
  `./build/openmw`

*Rules:*
- Always target `openmw` rather than a full repo build so edits in `components/` don't trigger relinking `openmw-cs` or `openmw-launcher`.
- Every C++ change must compile cleanly before you report completion.

## Engineering Standards
- **C++20:** Follow `.clang-format` (4 spaces, no tabs) and `.clang-tidy`. Keep diffs surgical—do not refactor unrelated code.
- **Lua-First Prototyping:** For gameplay, UI, or mechanics experiments, prefer implementing in Lua (`files/data/scripts/`) or exposing a clean binding in `apps/openmw/mwlua/` before hardcoding logic into C++.
- **OSG Thread Safety:** Never mutate OpenSceneGraph scenegraph nodes during the draw traversal. Keep state updates in the update traversal/callback and flag any rendering changes that touch shared `osg::StateSet` or `osg::Node` objects.
- **Physics Precision:** OpenMW uses **double-precision Bullet** (`bullet-dp`). Match existing vector/scalar types (`osg::Vec3f` vs `osg::Vec3d` / `btVector3`) to avoid silent precision loss.
- **Plan First:** For changes touching `mwrender/`, `mwphysics/`, or cross-cutting headers in `components/`, outline the plan first before editing files.
- **Performance:** Build code optimized for runtime, so better and more demanding visual effects can be run. For this consider CPU and GPU runtie and build optimized code. Use bindless descriptors, compute-shader culling, and Multi-Draw Indirect and other Vulkan specific optimizations where you see fit.

## Subagent Delegation Policy (`agy`)
Use `agy` subagents (monitored via `/agents`) to keep the main conversation context clean and parallelize heavy work. If there is only one task to do, do not spawn subagents. Only spawn when tasks can be done in parallel. If there is two tasks to do for example, only spawn one subagent, if theres 3, 2, since you can do the other part:
- **Delegate to a subagent:**
  1. **Deep Codebase Research:** Tracing how a mechanic flows across `components/` and `apps/openmw/` (e.g., "Trace how hit-chance or shadow cascades are computed and return a concise file/line summary").
  2. **Build & Test Loops:** Running `cmake --build` and `ctest`, then returning only the failing compiler/test diagnostics rather than full build logs.
  3. **Independent Diff Review:** After a non-trivial C++ edit, spin up a clean-context subagent to audit the `git diff` for OSG thread-safety, memory ownership, and `.clang-format` issues.
- **Keep in the main session:**
  - Architectural decisions, multi-file edits that depend on each other, and quick 1–2 file fixes where delegation overhead isn't worth it.
