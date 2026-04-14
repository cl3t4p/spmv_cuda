add_rules("mode.debug", "mode.release")

target("spmv")
    set_kind("binary")
    add_files("src/*.cu")
    add_cugencodes("native")
    add_includedirs("inc/")
    add_cxflags("-fopenmp")
    add_ldflags("-fopenmp")
