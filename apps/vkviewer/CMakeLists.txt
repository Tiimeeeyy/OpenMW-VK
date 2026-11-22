openmw_add_executable(vkviewer
    main.cpp
)

target_link_libraries(vkviewer
    openmw_vkrender
    components
    Boost::program_options
    SDL2::SDL2
    Vulkan::Vulkan
)

if (WIN32)
    target_sources(vkviewer PRIVATE ${CMAKE_SOURCE_DIR}/files/windows/other-apps.manifest)
endif()
