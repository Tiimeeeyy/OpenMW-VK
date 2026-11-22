---
name: openmw-verify
description: Use after editing C++ files in OpenMW to format, incrementally compile, and run targeted unit tests.
---

# OpenMW Verification Loop

1. Run `git diff --name-only` to list modified `.cpp`/`.hpp` files.
2. Run `clang-format -i` on those files only.
3. Compile the engine target:
   `cmake --build build --target openmw -j$(nproc)`
4. If files in `components/` were modified, also build and run the test suite:
   `cmake --build build --target openmw_test_suite -j$(nproc) && ctest --test-dir build --output-on-failure`
5. Report a brief summary of the diff and confirm build/test status.
