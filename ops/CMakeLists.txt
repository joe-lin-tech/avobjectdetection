# Get all project dir
FILE(GLOB ops-metal ${CMAKE_CURRENT_SOURCE_DIR}/ops-metal/*)

# For each project dir, build a target
FOREACH(project ${ops-metal})
    IF(IS_DIRECTORY ${project})
        # Get project name and all sources
        get_filename_component(project-name ${project} NAME)
        FILE(GLOB ${project}-src ${project}/*.cpp)

        # Create executable and link target
        add_executable(${project-name} ${${project}-src})
        target_link_libraries(${project-name} METAL_CPP)

        message(STATUS "Adding ${project-name}")
    ENDIF()
ENDFOREACH()