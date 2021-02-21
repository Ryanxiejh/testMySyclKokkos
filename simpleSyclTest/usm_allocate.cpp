//
// Created by Ryanxiejh on 2021/2/17.
//

#include <Kokkos_Core.hpp>
#include <SYCL/Kokkos_SYCL_Instance.hpp>
#include <SYCL/Kokkos_SyclSpace.hpp>
#include <SYCL/Kokkos_SYCL.hpp>

#include <cstdio>
#include <iostream>

typedef Kokkos::View<long * [3],Kokkos::LayoutRight,Kokkos::SyclSpace> view_type;
typedef view_type::HostMirror host_view_type;

struct InitView {
    view_type a;

    InitView(view_type a_) : a(a_) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const int i) const {
        a(i, 0) =  i;
        a(i, 1) =  i + 2;
        a(i, 2) = 2 * i;
    }

};

int main(int argc, char* argv[]){
    Kokkos::initialize(argc, argv);
    {
        const int N = 10;
        view_type a("A", N);
        host_view_type host_a = create_mirror(a);
        std::cout << typeid(Kokkos::DefaultExecutionSpace).name() << std::endl;
//        Kokkos::parallel_for(N, InitView(a));
//
//        Kokkos::deep_copy(host_a, a);
//
//        for(int i=0;i<N;i++){
//            printf("value %d: %ld %ld %ld\n",i,host_a(i,0),host_a(i,1),host_a(i,2));
//        }
    }
    Kokkos::finalize();
    //Kokkos::SYCL::in_parallel();
    return 0;
}
