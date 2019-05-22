project(Demo LANGUAGES CXX CUDA)

include_directories("../include")

add_executable(Demo
    main.cu
)

target_link_libraries(Demo cuNSearch)
target_compile_definitions(Demo PUBLIC $<$<CONFIG:DEBUG>:DEBUG>)
