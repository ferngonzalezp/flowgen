#!/bin/bash
module purge 
module load python/anaconda3.10.9
module load tools/cmake/3.31.0 lib/hdf5/1.10.8_gcc112
module load mpi/openmpi/4.1.1_gcc112 nvidia/cuda/12.4
export PSM2_CUDA=0
export CC=mpicc
export CXX=mpiCC
export FC=mpif90
export CUDACXX=/softs/nvidia/cuda-12.4/bin/nvcc
export CMAKE_CUDA_ARCHITECTURE=all
cd ../
source pyenvs/flowgen_rk8/bin/activate
git clone https://github.com/ornladios/ADIOS2
mkdir adios2-build
cd adios2-build
cmake  -DADIOS2_BUILD_EXAMPLES=ON \
    -DBUILD_TESTING=ON \
    -DADIOS2_USE_CUDA=ON \
    -DADIOS2_USE_Python=ON \
    -DADIOS2_USE_MPI=ON \
    -DPython_FIND_STRATEGY=LOCATION \
    -DPython_ROOT=/scratch/cfd/gonzalez/pyenvs/flowgen_rk8/bin/python3 \
    -DCMAKE_CUDA_COMPILER=$CUDACXX \
    -DCMAKE_INSTALL_PREFIX=../adios2_flowgen ../ADIOS2
make -j 4
make install
cd ../
rm -r ADIOS2 adios2-build