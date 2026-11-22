find_program(OPENMW_GLSLC glslc)
if (NOT OPENMW_GLSLC)
    find_program(OPENMW_GLSLANG_VALIDATOR glslangValidator)
endif()

if (NOT OPENMW_GLSLC AND NOT OPENMW_GLSLANG_VALIDATOR)
    message(FATAL_ERROR "OPENMW_EXPERIMENTAL_VULKAN requires either 'glslc' or 'glslangValidator' to compile SPIR-V shaders.")
endif()

find_package(PNG REQUIRED)
find_package(JPEG REQUIRED)

set(VKRENDER_SHADER_OUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")
file(MAKE_DIRECTORY "${VKRENDER_SHADER_OUT_DIR}")

set(VKRENDER_SHADERS
    shaders/cull.comp
    shaders/light_cull.comp
    shaders/depth_resolve.comp
    shaders/gtao.comp
    shaders/bloom.comp
    shaders/shadow.vert
    shaders/sky.vert
    shaders/sky.frag
    shaders/mesh.vert
    shaders/mesh.frag
    shaders/post.vert
    shaders/post.frag
)

set(VKRENDER_SPV_FILES)
foreach(SHADER ${VKRENDER_SHADERS})
    get_filename_component(SHADER_NAME "${SHADER}" NAME)
    set(SPV_OUT "${VKRENDER_SHADER_OUT_DIR}/${SHADER_NAME}.spv")
    if (OPENMW_GLSLC)
        add_custom_command(
            OUTPUT "${SPV_OUT}"
            COMMAND "${OPENMW_GLSLC}" --target-env=vulkan1.2 "${CMAKE_CURRENT_SOURCE_DIR}/${SHADER}" -o "${SPV_OUT}"
            DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${SHADER}"
            COMMENT "Compiling Vulkan shader ${SHADER_NAME} with glslc"
            VERBATIM
        )
    else()
        add_custom_command(
            OUTPUT "${SPV_OUT}"
            COMMAND "${OPENMW_GLSLANG_VALIDATOR}" --target-env vulkan1.2 -V "${CMAKE_CURRENT_SOURCE_DIR}/${SHADER}" -o "${SPV_OUT}"
            DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${SHADER}"
            COMMENT "Compiling Vulkan shader ${SHADER_NAME} with glslangValidator"
            VERBATIM
        )
    endif()
    list(APPEND VKRENDER_SPV_FILES "${SPV_OUT}")
endforeach()

add_custom_target(vkrender_shaders DEPENDS ${VKRENDER_SPV_FILES})

add_library(openmw_vkrender STATIC
    mesh.hpp
    mesh.cpp
    scene.hpp
    texture.hpp
    texture.cpp
    renderer.hpp
    renderer.cpp
)

add_dependencies(openmw_vkrender vkrender_shaders)

target_compile_definitions(openmw_vkrender PUBLIC
    OPENMW_VK_SHADER_DIR="${VKRENDER_SHADER_OUT_DIR}"
)

target_link_libraries(openmw_vkrender
    PUBLIC
        Vulkan::Vulkan
        SDL2::SDL2
        components
    PRIVATE
        PNG::PNG
        JPEG::JPEG
)

option(OPENMW_EXPERIMENTAL_VULKAN_ASSIMP "Enable Assimp mesh loading (.dae/.gltf/.obj) in experimental Vulkan renderer" ON)
if (OPENMW_EXPERIMENTAL_VULKAN_ASSIMP)
    find_package(assimp CONFIG REQUIRED)
    target_compile_definitions(openmw_vkrender PUBLIC OPENMW_VKRENDER_USE_ASSIMP=1)
    target_link_libraries(openmw_vkrender PRIVATE assimp::assimp)
endif()
