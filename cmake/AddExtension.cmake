macro (add_extension source_dir output_dir)

    # Get fortran and signature files
    file(GLOB fortran_files ${source_dir}/*.F)
    file(GLOB signature_files ${source_dir}/*.pyf)

    # Ensure that there is exactly one signature file
    list(LENGTH signature_files n_signature)
    if (NOT ${n_signature} EQUAL 1)
    message(
        FATAL_ERROR
        "Detected more than one signature file for ${name} extension ${n_signature}"
        )
    endif()

    # Ensure that there is at least one fortran source
    list(LENGTH fortran_files n_fortran)
    if (NOT ${n_fortran} GREATER_EQUAL 1)
        message(
            FATAL_ERROR
            "Missing fortran source files for ${name} extension ${n_fortran}"
        )
    endif()

    # Compile and install extension
    execute_process(
        COMMAND
            ${Python_EXECUTABLE} -m numpy.f2py
            -c ${signature_files} ${fortran_files}
            --backend meson
        WORKING_DIRECTORY
            ${output_dir}
    )

endmacro()
