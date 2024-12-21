cmake_minimum_required(VERSION 3.17.2...3.29)
project(${SKBUILD_PROJECT_NAME} LANGUAGES C Fortran)

find_package(
  Python
  COMPONENTS Interpreter Development.Module NumPy
  REQUIRED
)

# Alias for source directiories
set(IERS_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/iers")
set(PRIDE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/pride")

# Import add_extension macro
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(AddExtension)

# Add extensions
add_extension(${IERS_SOURCE_DIR}/dehanttideinel ${PRIDE_SOURCE_DIR}/external/iers)
