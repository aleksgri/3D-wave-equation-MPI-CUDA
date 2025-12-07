#include <iostream>
#include <cmath>
#include <vector>
//#include <omp.h>
#include <ctime>
#include <string>
#include <cstring>
#include <fstream>
#include <chrono>
#include <mpi.h>
#include <unistd.h>
#include <cuda_runtime.h>

//using namespace std;

#define CUDA_RT_CALL(call)                                                                  \
    {                                                                                       \
        cudaError_t cudaStatus = call;                                                      \
        if (cudaSuccess != cudaStatus) {                                                    \
            fprintf(stderr,                                                                 \
                    "ERROR: CUDA RT call \"%s\" in line %d of file %s failed "              \
                    "with "                                                                 \
                    "%s (%d).\n",                                                           \
                    #call, __LINE__, __FILE__, cudaGetErrorString(cudaStatus), cudaStatus); \
            exit( cudaStatus );                                                             \
        }                                                                                   \
    }

void call_copy_to_matrices(double * grid, double * matrix1, double * matrix2, int x, int y, int z,
                        int dim, int start, int end, cudaStream_t stream);

void call_copy_to_grid(double * grid, double * matrix, int x, int y, int z,
                        int dim, int index, cudaStream_t stream);
void init_device_constants(double a2_, double a_t_, double tau_, double hx_, double hy_, double hz_, double Lx_, double Ly_, double Lz_,  int x0, int y0, int z0, cudaStream_t stream);

void call_calculate_start(double * grid0, double * grid1, int x, int y, int z, int n,
                        int i1, int j1, int k1, int i2, int j2, int k2,
                        double * abs_errs, double * rel_errs, cudaStream_t stream);

void calculate_errors(double * abs_errs, double * rel_errs, int x, int y, int z, double * result, cudaStream_t stream);

void call_calculate_layer(double * grid0, double * grid1, double * grid2, int x, int y, int z, int n,
                        int i1, int j1, int k1, int i2, int j2, int k2,
                        double * abs_errs, double * rel_errs, cudaStream_t stream);

void call_prepare_layer(double * grid0, double * grid1, double * grid2, int n, int x, int y, int z,
                        int j1, int k1, int j2, int k2, unsigned int b, cudaStream_t stream);

double Lx, Ly, Lz, a_t, a2, T;
double tau, hx, hy, hz;

unsigned long Np;

const double PI =  3.14159265358979323846;

float total_exchange_time = 0.0;
float total_mpi_exchange_time = 0.0;
float total_loop_time = 0.0;
float total_error_time = 0.0;

int N = 32;
int X, Y, Z;
int x_0, y_0, z_0;
int timesteps;

cudaStream_t stream;
cudaStream_t exchange_stream;
cudaEvent_t started, finished,
            exchange_started, exchange_finished,
            loop_started, loop_finished,
            error_calculation_started, error_calculation_finished,
            initialization_started, initialization_finished;

class Matrix {
private:
    int x, y;
public:
    double * array_h;
    double * array_d;
    Matrix() {}

    void init_device(int x_, int y_) {
        x = x_;
        y = y_;
        cudaMalloc(&array_d, x*y*sizeof(double));
        cudaMemsetAsync(array_d, 0, x*y*sizeof(double), stream);
    }
    
    void init_host(int x_, int y_) {
        x = x_;
        y = y_;
        cudaMallocHost(&array_h, x*y*sizeof(double));
        for(unsigned i = 0; i < x*y; i++) array_h[i] = 0.0;
    }

    void init(int x_, int y_) {
        x = x_;
        y = y_;
        cudaMallocHost(&array_h, x*y*sizeof(double));
        memset(array_h, 0, x*y*sizeof(double));
        cudaMalloc(&array_d, x*y*sizeof(double));
        cudaMemsetAsync(array_d, 0, x*y*sizeof(double), stream);
    }

    void copy_to_host() {
        cudaMemcpyAsync(array_h, array_d, x*y*sizeof(double), cudaMemcpyDeviceToHost, stream);
    }

    void copy_to_device() {
        cudaMemcpyAsync(array_d, array_h, x*y*sizeof(double), cudaMemcpyHostToDevice, stream);
    }

    double * get_ptr_host() {
        return array_h;
    }
    double * get_ptr_device() {
        return array_d;
    }

    ~Matrix() {
        cudaFreeHost(array_h);
        cudaFree(array_d);
    }
};

class Grid { // 3D grid
private:
    double * array_h;
    double * array_d;
    int x, y, z;
public:
    Grid() {
        
    }

    void init_grid_device(int x_, int y_, int z_) {
        x = x_;
        y = y_;
        z = z_;
        cudaMalloc(&array_d, x*y*z*sizeof(double));
        cudaMemsetAsync(array_d, 0.0, x*y*z*sizeof(double), stream);
    }

    double * get_ptr_device() {
        return array_d;
    }

    void memset_device(double s) {
        cudaMemsetAsync(array_d, s, x*y*z*sizeof(double), stream);
    }

    void init_grid(int x_, int y_, int z_) {
        x = x_;
        y = y_;
        z = z_;
        cudaMallocHost(&array_h, x * y * z * sizeof(double));
        memset(array_h, 0, x*y*z*sizeof(double));
    }

    double get(int i, int j, int k) {
        return array_h[i*y*z + j*z + k];
    }

    void set(int i, int j, int k, double val) {
        array_h[i*y*z + j*z + k] = val;
    }

    // host only
    double laplace(int i, int j, int k, int x1, int y1, int z1, int x2, int y2, int z2) {
        double ans = 0.0;
        ans += (array_h[x1*y*z + j*z + k] - 2*array_h[i*y*z + j*z + k] + array_h[x2*y*z + j*z + k])/(hx*hx);
        ans += (array_h[i*y*z + y1*z + k] - 2*array_h[i*y*z + j*z + k] + array_h[i*y*z + y2*z + k])/(hy*hy);
        ans += (array_h[i*y*z + j*z + z1] - 2*array_h[i*y*z + j*z + k] + array_h[i*y*z + j*z + z2])/(hz*hz);
        return ans;
    }

    void copy_to_host() { // not used
        cudaMemcpy(array_h, array_d, x*y*z*sizeof(double), cudaMemcpyDeviceToHost);
    }

    void print_layer(int t) {
        std::cout << "Layer " << t << ":\n";
        for(int i = 0; i < x; i++) {
            for(int j = 0; j < y; j++) {
                for(int k = 0; k < z; k++) {
                    std::cout << array_h[i*y*z + j*z + k] << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
    }

    ~Grid() {
        cudaFreeHost(array_h);
        cudaFree(array_d);
    }
};

Grid grids[3];
Grid abs_errors;
Grid rel_errors;

std::vector<double> max_abs_errors;
std::vector<double> max_rel_errors;
double errors[2] = {0.0, 0.0};

MPI_Comm cart_comm;
MPI_Comm local_comm;
Matrix up, down, front, back, right, left; // Matrices that are to be received from neighbors and copied into grid (0, X + 1)
Matrix up1, down1, front1, back1, right1, left1; // Matrices that are to be copied from grid and sent to neighbors (1, X)
int up_rank, down_rank, front_rank, back_rank, right_rank, left_rank;
int dim[3] = {0, 0, 0};
int coords[3];
int nprocs, rank;

std::ofstream out;

void prepare_layer(int n) { // граничные значения
    unsigned int b = (back_rank < 0) * 1 + (front_rank < 0) * 2 + (left_rank < 0) * 4 + (right_rank < 0) * 8 + (coords[0] == dim[0]-1) * 16 + (coords[0] == 0) * 32;
    cudaEventRecord(loop_started, stream);
    call_prepare_layer(grids[(n+1)%3].get_ptr_device(), grids[(n+2)%3].get_ptr_device(), grids[n%3].get_ptr_device(), 1, X+2, Y+2, Z+2, coords[1] == 0 ? 2 : 1, coords[2] == 0 ? 2 : 1, coords[1] == dim[1]-1 ? Y-1 : Y, coords[2] == dim[2]-1 ? Z-1 : Z, b, stream);
    cudaEventRecord(loop_finished, stream);
    
}

void exchange(int n) {
    MPI_Status status;
    call_copy_to_matrices(grids[n%3].get_ptr_device(), down1.array_d, up1.array_d, X+2, Y+2, Z+2, 1, coords[0] == 0 ? 2 : 1, coords[0] == dim[0]-1 ? X-1 : X, stream);
    call_copy_to_matrices(grids[n%3].get_ptr_device(), back1.array_d, front1.array_d, X+2, Y+2, Z+2, 3, 1, Z, stream);
    call_copy_to_matrices(grids[n%3].get_ptr_device(), left1.array_d, right1.array_d, X+2, Y+2, Z+2, 2, 1, Y, stream);
    cudaEventRecord(exchange_started, stream);
    down1.copy_to_host(); up1.copy_to_host();
    front1.copy_to_host(); back1.copy_to_host();
    right1.copy_to_host(); left1.copy_to_host();
    cudaEventRecord(exchange_finished, stream);
    cudaEventSynchronize(exchange_finished);
    float t = 0.0;
    cudaEventElapsedTime(&t, exchange_started, exchange_finished);
    total_exchange_time += t;

    double start = MPI_Wtime();
    if(up_rank >= 0 && down_rank >= 0) {
        MPI_Sendrecv(up1.array_h, (Z+2)*(Y+2), MPI_DOUBLE, up_rank, 1, 
                     down.array_h, (Z+2)*(Y+2), MPI_DOUBLE, down_rank, 1, cart_comm, &status);
        MPI_Sendrecv(down1.array_h, (Z+2)*(Y+2), MPI_DOUBLE, down_rank, 2, 
                     up.array_h, (Z+2)*(Y+2), MPI_DOUBLE, up_rank, 2, cart_comm, &status);
    }
    // left-right
    if(right_rank >= 0 && left_rank >= 0) {
        MPI_Sendrecv(right1.array_h, (Z+2)*(X+2), MPI_DOUBLE, right_rank, 3, 
                     ::left.array_h, (Z+2)*(X+2), MPI_DOUBLE, left_rank, 3, cart_comm, &status);
    } else if (right_rank >= 0) {
        MPI_Send(right1.array_h, (Z+2)*(X+2), MPI_DOUBLE, right_rank, 3, cart_comm);
    } else if(left_rank >= 0) {
        MPI_Recv(::left.array_h, (Z+2)*(X+2), MPI_DOUBLE, left_rank, 3, cart_comm, &status);
    }
    //std::cout << "proc " << rank << " ex " << n << " left-right exchange complete\n";
    // right-left
    if(right_rank >= 0 && left_rank >= 0) {
        MPI_Sendrecv(left1.array_h, (Z+2)*(X+2), MPI_DOUBLE, left_rank, 4, 
                     ::right.array_h, (Z+2)*(X+2), MPI_DOUBLE, right_rank, 4, cart_comm, &status);
    } else if (left_rank >= 0) {
        MPI_Send(left1.array_h, (Z+2)*(X+2), MPI_DOUBLE, left_rank, 4, cart_comm);
    } else if(right_rank >= 0) {
        MPI_Recv(::right.array_h, (Z+2)*(X+2), MPI_DOUBLE, right_rank, 4, cart_comm, &status);
    }
    //std::cout << "proc " << rank << " ex " << n << " right-left exchange complete\n";
    // back-front
    if(front_rank >= 0 && back_rank >= 0) {
        MPI_Sendrecv(front1.array_h, (Y+2)*(X+2), MPI_DOUBLE, front_rank, 5, 
                     back.array_h, (Y+2)*(X+2), MPI_DOUBLE, back_rank, 5, cart_comm, &status);
    } else if (front_rank >= 0) {
        MPI_Send(front1.array_h, (Y+2)*(X+2), MPI_DOUBLE, front_rank, 5, cart_comm);
    } else if(back_rank >= 0) {
        MPI_Recv(back.array_h, (Y+2)*(X+2), MPI_DOUBLE, back_rank, 5, cart_comm, &status);
    }
    //std::cout << "proc " << rank << " ex " << n << " back-front exchange complete\n";
    // front-back
    if(front_rank >= 0 && back_rank >= 0) {
        MPI_Sendrecv(back1.array_h, (Y+2)*(X+2), MPI_DOUBLE, back_rank, 6, 
                     front.array_h, (Y+2)*(X+2), MPI_DOUBLE, front_rank, 6, cart_comm, &status);
    } else if (back_rank >= 0) {
        MPI_Send(back1.array_h, (Y+2)*(X+2), MPI_DOUBLE, back_rank, 6, cart_comm);
    } else if(front_rank >= 0) {
        MPI_Recv(front.array_h, (Y+2)*(X+2), MPI_DOUBLE, front_rank, 6, cart_comm, &status);
    }
    double end = MPI_Wtime();
    total_mpi_exchange_time += (end - start)*1000;
    //std::cout << "proc " << rank << " ex " << n << " front-back exchange complete\n";
    //std::cout << "proc " << rank << " ex " << n << " side exchanges complete\n";
    // copy matrices
    cudaEventRecord(exchange_started, stream);
    down.copy_to_device(); up.copy_to_device();
    front.copy_to_device(); back.copy_to_device();
    right.copy_to_device(); left.copy_to_device();
    cudaEventRecord(exchange_finished, stream);
    cudaEventSynchronize(exchange_finished);
    t = 0.0;
    cudaEventElapsedTime(&t, exchange_started, exchange_finished);
    total_exchange_time += t;
    if(left_rank >= 0) call_copy_to_grid(grids[n%3].get_ptr_device(), left.array_d, X+2, Y+2, Z+2, 2, 0, stream);
    if(right_rank >= 0) call_copy_to_grid(grids[n%3].get_ptr_device(), right.array_d, X+2, Y+2, Z+2, 2, Y+1, stream);
    if(back_rank >= 0) call_copy_to_grid(grids[n%3].get_ptr_device(), back.array_d, X+2, Y+2, Z+2, 3, 0, stream);
    if(front_rank >= 0) call_copy_to_grid(grids[n%3].get_ptr_device(), front.array_d, X+2, Y+2, Z+2, 3, Z+1, stream);
    if(up_rank >= 0) call_copy_to_grid(grids[n%3].get_ptr_device(), up.array_d, X+2, Y+2, Z+2, 1, X+1, stream);
    if(down_rank >= 0) call_copy_to_grid(grids[n%3].get_ptr_device(), down.array_d, X+2, Y+2, Z+2, 1, 0, stream);

}

void calculate_start() { // начальные значения
    // n = 0
    cudaEventRecord(loop_started, stream);
    call_calculate_layer(grids[0].get_ptr_device(), 0, 0, X+2, Y+2, Z+2, 0, 1, 1, 1, X, Y, Z, abs_errors.get_ptr_device(), rel_errors.get_ptr_device(), stream);
    cudaEventRecord(loop_finished, stream);
    cudaStreamSynchronize(stream);
    float t = 0.0;
    cudaEventElapsedTime(&t, loop_started, loop_finished);
    total_loop_time += t;

    cudaEventRecord(error_calculation_started, stream);
    calculate_errors(abs_errors.get_ptr_device(), rel_errors.get_ptr_device(), X+2, Y+2, Z+2, errors, stream);
    cudaEventRecord(error_calculation_finished, stream);
    cudaEventSynchronize(error_calculation_finished);
    cudaEventElapsedTime(&t, error_calculation_started, error_calculation_finished);
    total_error_time += t;

    exchange(0);
    max_abs_errors.push_back(errors[0]);
    max_rel_errors.push_back(errors[1]);
    abs_errors.memset_device(0.0);
    rel_errors.memset_device(0.0);
    cudaStreamSynchronize(stream);

    if(rank == 0) std::cout << "calculating layer " << 1 << std::endl;
    // n = 1
    prepare_layer(1);
    //#pragma omp parallel for num_threads(Np) shared(num_sol)
    int x = coords[0] == dim[0]-1 ? X-1 : X,
        y = coords[1] == dim[1]-1 ? Y-1 : Y,
        z = coords[2] == dim[2]-1 ? Z-1 : Z,
        x1 = coords[0] == 0 ? 2 : 1,
        y1 = coords[1] == 0 ? 2 : 1,
        z1 = coords[2] == 0 ? 2 : 1;
    cudaEventSynchronize(loop_finished); // prepare_layer loop
    cudaEventElapsedTime(&t, loop_started, loop_finished);
    total_loop_time += t;
    cudaEventRecord(loop_started, stream);
    call_calculate_layer(grids[0].get_ptr_device(), grids[1].get_ptr_device(), 0, X+2, Y+2, Z+2, 1, x1, y1, z1, x, y, z, abs_errors.get_ptr_device(), rel_errors.get_ptr_device(), stream);
    cudaEventRecord(loop_finished, stream);
    cudaStreamSynchronize(stream);

    cudaEventRecord(error_calculation_started, stream);
    calculate_errors(abs_errors.get_ptr_device(), rel_errors.get_ptr_device(), X+2, Y+2, Z+2, errors, stream);
    cudaEventRecord(error_calculation_finished, stream);
    cudaEventSynchronize(error_calculation_finished);
    cudaEventElapsedTime(&t, error_calculation_started, error_calculation_finished);
    total_error_time += t;
    
    exchange(1);
    cudaEventElapsedTime(&t, loop_started, loop_finished);
    total_loop_time += t;
    max_abs_errors.push_back(errors[0]);
    max_rel_errors.push_back(errors[1]);
    abs_errors.memset_device(0.0);
    rel_errors.memset_device(0.0);
    cudaStreamSynchronize(stream);
    cudaDeviceSynchronize();
    //std::cout << rank << " " << max_abs_errors.size() << " " << errors[0] << std::endl;
    
    
    cudaStreamSynchronize(stream);
}




void calculate_num_sol() {
    cudaEventRecord(started, stream);
    calculate_start();
    for(int n = 2; n <= timesteps; n++) {
        if(rank == 0) std::cout << "calculating layer " << n << std::endl;
        prepare_layer(n);

        int x = coords[0] == dim[0]-1 ? X-1 : X,
            y = coords[1] == dim[1]-1 ? Y-1 : Y,
            z = coords[2] == dim[2]-1 ? Z-1 : Z,
            x1 = coords[0] == 0 ? 2 : 1,
            y1 = coords[1] == 0 ? 2 : 1,
            z1 = coords[2] == 0 ? 2 : 1;
        
        cudaEventSynchronize(loop_finished); // prepare_layer loop
        float t;
        cudaEventElapsedTime(&t, loop_started, loop_finished);
        total_loop_time += t;
        cudaEventRecord(loop_started, stream);
        call_calculate_layer(grids[(n+1) % 3].get_ptr_device(), grids[(n+2) % 3].get_ptr_device(), grids[n % 3].get_ptr_device(), X+2, Y+2, Z+2, n, x1, y1, z1, x, y, z, abs_errors.get_ptr_device(), rel_errors.get_ptr_device(), stream);
        cudaEventRecord(loop_finished, stream);
        cudaStreamSynchronize(stream); // must wait before calculating errors and exchanging
        calculate_errors(abs_errors.get_ptr_device(), rel_errors.get_ptr_device(), X+2, Y+2, Z+2, errors, stream);
        if(n < timesteps) exchange(n);
        cudaEventElapsedTime(&t, loop_started, loop_finished);
        total_loop_time += t;
        //std::cout << t << "\n";
        cudaStreamSynchronize(stream);
        // std::cout << rank << " " << max_abs_errors.size() << " " << errors[0] << std::endl;
        max_abs_errors.push_back(errors[0]);
        max_rel_errors.push_back(errors[1]);
        abs_errors.memset_device(0.0);
        rel_errors.memset_device(0.0);
        
        cudaStreamSynchronize(stream);
    }
    cudaEventRecord(finished, stream);
    cudaEventSynchronize(finished);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, started, finished);
    if(rank == 0) {
        out << "numerical solution calculated in " << milliseconds << "ms" << std::endl;
    }
    std::vector<double> global_max_abs_errors(timesteps + 1);
    std::vector<double> global_max_rel_errors(timesteps + 1);
    MPI_Reduce(max_abs_errors.data(), global_max_abs_errors.data(), timesteps + 1, MPI_DOUBLE, MPI_MAX, 0, cart_comm);
    MPI_Reduce(max_rel_errors.data(), global_max_rel_errors.data(), timesteps + 1, MPI_DOUBLE, MPI_MAX, 0, cart_comm);
    if(rank == 0) {
        for(int n = 0; n <= timesteps; n++) {
            out << "max abs and rel errors on layer " << n << ": " << global_max_abs_errors[n] << " " << global_max_rel_errors[n] << std::endl;
        }
        //std::cout << "errors calculated\n";
        out << "total host-device exchange time: " << total_exchange_time << " ms\n";
        out << "total loop time: " << total_loop_time << " ms\n";
        out << "total MPI exchange time: " << total_mpi_exchange_time << " ms\n";
    }
}



int main(int argc, char *argv[]) { // command line args: N, Np, Lx, Ly, Lz, T, timesteps
    MPI_Init(&argc, &argv);
    
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    N = std::stoi(argv[1]);
    Np = std::stoi(argv[2]);
    if(strcmp(argv[3], "pi") == 0) Lx = PI;
    else Lx = std::stod(argv[3]);
    if(strcmp(argv[4], "pi") == 0) Ly = PI;
    else Ly = std::stod(argv[4]);
    if(strcmp(argv[5], "pi") == 0) Lz = PI;
    else Lz = std::stod(argv[5]);
    if(argc >= 7) T = std::stod(argv[6]);
    else T = 1;
    if(argc >= 8) timesteps = std::stoi(argv[7]);
    else timesteps = 20;


    a2 = 1/(4*PI*PI);
    a_t = 0.5 * sqrt(4/(Lx*Lx) + 1/(Ly*Ly) + 1/(Lz*Lz));
    
    tau = T / timesteps;
    hx = Lx / N; hy = Ly / N; hz = Lz / N;

    if(rank == 0) {
        std::cout << "C = " << sqrt(a2)*tau/std::min(hx, std::min(hy, hz)) << std::endl;
    }
    
    // creating virtual topology
    MPI_Dims_create(nprocs, 3, dim);
    int periods[3] = {true, false, false};
    MPI_Cart_create(MPI_COMM_WORLD, 3, dim, periods, false, &cart_comm);

    
    MPI_Cart_coords(cart_comm, rank, 3, coords);
    //std::cout << "process " << rank << " coords: " << coords[0] << " " << coords[1] << " " << coords[2] << std::endl;
    X = (N+1) / dim[0], Y = (N+1) / dim[1], Z = (N+1) / dim[2];
    x_0 = coords[0]*X, y_0 = coords[1]*Y, z_0 = coords[2]*Z; // grid starting points
    if(coords[0] == dim[0]-1) X += (N+1) % dim[0];
    if(coords[1] == dim[1]-1) Y += (N+1) % dim[1];
    if(coords[2] == dim[2]-1) Z += (N+1) % dim[2];

    
    // grid coords: ([x_0, x_0 + X - 1], [y_0, y_0 + Y - 1], [z_0, z_0 + Z - 1])
    // + 2 side coords for neighbour's values
    MPI_Cart_shift(cart_comm, 0, 1, &down_rank, &up_rank); // down to top
    MPI_Cart_shift(cart_comm, 1, 1, &left_rank, &right_rank);
    MPI_Cart_shift(cart_comm, 2, 1, &back_rank, &front_rank);
    // std::cout << rank << " down and up: " << down_rank << " " << up_rank << std::endl;
    // std::cout << rank << " left and right: " << left_rank << " " << right_rank << std::endl;
    // std::cout << rank << " back and front: " << back_rank << " " << front_rank << std::endl;

    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                                     &local_comm);
    int local_rank, local_size;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_size(local_comm, &local_size);

    MPI_Comm_free(&local_comm);

    char hostname[256];
    gethostname(hostname, sizeof(hostname));

    std::cout << "Process " << rank << " local rank = " << local_rank << " local size = " << local_size << " hostname = " << hostname << std::endl;
    

    // CUDA
    int num_devices;
    CUDA_RT_CALL(cudaGetDeviceCount(&num_devices));
    CUDA_RT_CALL(cudaSetDevice(local_rank%num_devices));
    //CUDA_RT_CALL(cudaFree(0)); // initialize CUDA runtime context
    CUDA_RT_CALL(cudaStreamCreate(&stream));
    CUDA_RT_CALL(cudaStreamCreate(&exchange_stream));
    CUDA_RT_CALL(cudaEventCreate(&started));
    CUDA_RT_CALL(cudaEventCreate(&finished));
    CUDA_RT_CALL(cudaEventCreate(&exchange_started));
    CUDA_RT_CALL(cudaEventCreate(&exchange_finished));
    CUDA_RT_CALL(cudaEventCreate(&loop_started));
    CUDA_RT_CALL(cudaEventCreate(&loop_finished));
    CUDA_RT_CALL(cudaEventCreate(&initialization_started));
    CUDA_RT_CALL(cudaEventCreate(&initialization_finished));
    if(local_rank == 0) std::cout << "number of CUDA devices on node " << hostname << ": " << num_devices << std::endl;

    if(rank == 0) {
        out.clear();
        out.open(std::string("output_N") + std::to_string(N) + "_Np" + std::to_string(nprocs) + "_Ng" + std::to_string(num_devices) + std::string("_cuda.txt"));
    }

    cudaEventRecord(initialization_started, stream);

    grids[0].init_grid_device(X + 2, Y + 2, Z + 2); // init only on device
    grids[1].init_grid_device(X + 2, Y + 2, Z + 2);
    grids[2].init_grid_device(X + 2, Y + 2, Z + 2);

    // grids[0].init_grid(X + 2, Y + 2, Z + 2); // init only on host
    // grids[1].init_grid(X + 2, Y + 2, Z + 2);
    // grids[2].init_grid(X + 2, Y + 2, Z + 2);

    abs_errors.init_grid_device(X + 2, Y + 2, Z + 2);
    rel_errors.init_grid_device(X + 2, Y + 2, Z + 2);
    // abs_errors.init_grid(X + 2, Y + 2, Z + 2);
    // rel_errors.init_grid(X + 2, Y + 2, Z + 2);
    up.init(Y+2, Z+2); up1.init(Y+2, Z+2); // init on both device and host
    down.init(Y+2, Z+2); down1.init(Y+2, Z+2);
    ::right.init(X+2, Z+2); right1.init(X+2, Z+2);
    ::left.init(X+2, Z+2); left1.init(X+2, Z+2);
    front.init(X+2, Y+2); front1.init(X+2, Y+2);
    back.init(X+2, Y+2); back1.init(X+2, Y+2);

    init_device_constants(a2, a_t, tau, hx, hy, hz, Lx, Ly, Lz, x_0, y_0, z_0, stream);

    cudaEventRecord(initialization_finished, stream);
    cudaStreamSynchronize(stream);
    float init_milliseconds = 0;
    cudaEventElapsedTime(&init_milliseconds, initialization_started, initialization_finished);
    if(rank == 0) {
        out << "initialization done in " << (unsigned) init_milliseconds << "ms" << std::endl;
    }

    MPI_Barrier(cart_comm);
    
    
    calculate_num_sol();


    if(rank == 0) out.close();
    MPI_Finalize();
    cudaEventDestroy(loop_started);
    cudaEventDestroy(loop_finished);
    cudaEventDestroy(started);
    cudaEventDestroy(finished);
    cudaEventDestroy(exchange_started);
    cudaEventDestroy(exchange_finished);
    cudaEventDestroy(initialization_started);
    cudaEventDestroy(initialization_finished);
    cudaStreamDestroy(stream);
    cudaStreamDestroy(exchange_stream);

    return 0;
}