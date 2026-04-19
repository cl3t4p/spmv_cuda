add_rules("mode.debug", "mode.release")

target("spmv")
    set_kind("binary")
    add_files("src/*.cpp")
    add_files("src/*.cu")
    -- Force Ampere 8.0
    add_cugencodes("sm_80")
    add_includedirs("inc/")
    add_headerfiles("inc/*.h", "inc/*.cuh")
    add_cxflags("-fopenmp")
    add_ldflags("-fopenmp")
