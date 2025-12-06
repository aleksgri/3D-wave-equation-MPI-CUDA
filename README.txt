1. OpenMP:

Компиляция:
xlc++_r openmp_sol.cpp -o omp -qsmp=omp -std=c++11

Запуск:
mpisubmit.pl -p 1 -t Np omp N Np Lx Ly Lz T timesteps
, где Np - кол-во нитей, N - кол-во узлов сетки, Lx, Ly, Lz, T - стороны куба и время (из задания), timesteps - количество временных щагов.

tau будет равным T / timesteps. В начале запуска выводится число C, по которому можно понять, устойчива ли схема
Выводится файл output_N{N}_Np{Np}.txt с погрешностями и временем вычисления.

Если число потоков превышает 8, можно воспользоваться скриптом OpenMP_job.lsf.

2. MPI

Компиляция:
module load SpectrumMPI
mpixlC mpi_sol.cpp -o mpi_sol -std=c++11

Запуск:
mpisubmit.pl -p Np mpi_sol N 1 Lx Ly Lz T timesteps
, где Np - число процессов.

3. MPI+OpenMP

Компиляция:
module load SpectrumMPI
mpixlC hybrid_sol.cpp -o hyb_sol -std=c++11 -fopenmp

Запуск:
mpisubmit.pl -p Np -t Nt hyb_sol N Nt Lx Ly Lz T timesteps
, где Np - число процессов, Nt - число потоков.

4. MPI+CUDA

Компиляция:
make

В makefile имеются переменные ARCH (архитектура), и HOST_COMP (хост компилятор).

Запуск:
bsub -n Np -R span[hosts=1] -gpu num=Ng:mode=shared:mps=yes -o "cuda.%J.out" -e "cuda.%J.err" mpirun -n Np cuda_sol N 1 Lx Ly Lz T timesteps
, где Np - количество процессов, Ng - количество видеокарт.
В результате Np процессов запускаются на Ng видеокартах, принадлежащих одному из узлов