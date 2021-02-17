cmake_minimum_required(VERSION 3.4)
project(testMySyclKokkos)

set(Kokkos_DIR /home/u62432/kokkos/swKokkos/lib/cmake/Kokkos)

find_package(Kokkos REQUIRED)

add_executable(testMySyclKokkos usm_allocate.cpp)

target_link_libraries(testMySyclKokkos Kokkos::kokkos)