---
name: openmw-osg-audit
description: Use when modifying OpenMW rendering, lighting, shadows, water, or GLSL shaders in apps/openmw/mwrender/, components/sceneutil/, or files/shaders/.
---

# OpenSceneGraph & Shader Safety Checklist

Before finalizing any rendering or shader edit:
1. **Traversal Safety:** Verify scenegraph mutations happen only during the update phase, never during cull/draw.
2. **StateSet Hygiene:** Avoid mutating shared `osg::StateSet` objects in place if other nodes reference them; clone or attach per-node where appropriate.
3. **Shader Compatibility:** Check that GLSL edits in `files/shaders/` respect OpenMW's `#include` / `@define` preprocessor syntax and don't break fallback paths (e.g., shadows disabled or clustered lighting toggled).
4. **Verification:** Compile `--target openmw` and note any visual settings in `settings.cfg` required to test the change.
