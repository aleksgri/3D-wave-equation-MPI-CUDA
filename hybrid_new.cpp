#include <iostream>
#include <cmath>
#include <vector>
#include <omp.h>
#include <ctime>
#include <string>
#include <cstring>
#include <fstream>
#include <chrono>
#include <mpi.h>
#include <unistd.h>
//#include <cuda_runtime.h>

//using namespace std;



float Lx, Ly, Lz, a_t, a2, T;
float tau, hx, hy, hz;

unsigned long Np;

const float PI =  3.1415926535;

float max_abs_error = -100.0;
float max_rel_error = -100.0;

int N = 32;
int X, Y, Z;
int x_0, y_0, z_0;
int timesteps;

float total_exchange_time = 0.0;
float total_loop_time = 0.0;

class Matrix {
private:
    
    int x, y;
public:
    float * array;
    Matrix() {}
    Matrix(int x_, int y_) {
        x = x_;
        y = y_;
        array = new float[x*y];
        for(unsigned i = 0; i < x*y; i++) array[i] = 0.0;
    }
    void init(int x_, int y_) {
        x = x_;
        y = y_;
        array = new float[x*y];
        for(unsigned i = 0; i < x*y; i++) array[i] = 0.0;
    }
    float * get_ptr() {
        return array;
    }
};

class Grid { // 3D grid
private:
    float * array;
    int x, y, z;
public:
    Grid() {
        
    }

    Grid(int x_, int y_, int z_) {
        x = x_;
        y = y_;
        z = z_;
        array = new float[x*y*z];
        for(unsigned i = 0; i < x*y*z; i++) array[i] = 0.0;
    }

    void init_grid_cuda(int x_, int y_, int z_) {
        // TODO
    }

    float * get_ptr_cuda() {
        // TODO
        return nullptr;
    }

    void init_grid(int x_, int y_, int z_) {
        x = x_;
        y = y_;
        z = z_;
        array = new float[x*y*z];
        for(unsigned i = 0; i < x*y*z; i++) array[i] = 0.0;
    }

    float get(int i, int j, int k) {
        return array[i*y*z + j*z + k];
    }

    void set(int i, int j, int k, float val) {
        //cout << "set " << t << i << j << k << endl;
        array[i*y*z + j*z + k] = val;
    }


    float laplace(int i, int j, int k, int x1, int y1, int z1, int x2, int y2, int z2) {
        float ans = 0.0;
        // array[n*size*size*size + i*size*size + j*size + k]
        ans += (array[x1*y*z + j*z + k] - 2*array[i*y*z + j*z + k] + array[x2*y*z + j*z + k])/(hx*hx);
        ans += (array[i*y*z + y1*z + k] - 2*array[i*y*z + j*z + k] + array[i*y*z + y2*z + k])/(hy*hy);
        ans += (array[i*y*z + j*z + z1] - 2*array[i*y*z + j*z + k] + array[i*y*z + j*z + z2])/(hz*hz);
        return ans;
    }

    void print_layer(int t) {
        std::cout << "Layer " << t << ":\n";
        for(int i = 0; i < x; i++) {
            for(int j = 0; j < y; j++) {
                for(int k = 0; k < z; k++) {
                    std::cout << array[i*y*z + j*z + k] << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
    }

    ~Grid() {
        delete[] array;
    }
};

Grid grids[3]; 

Grid cur_prec_sol; // sol[t][x][y][z]
Grid cur_abs_error;
Grid cur_rel_error;
std::vector<float> max_abs_errors;
std::vector<float> max_rel_errors;

MPI_Comm cart_comm;
MPI_Comm local_comm;
Matrix up, down, front, back, right, left; // Matrices that are to be received from neighbors and copied into grid (0, X + 1)
Matrix up1, down1, front1, back1, right1, left1; // Matrices that are to be copied from grid and sent to neighbors (1, X)
int up_rank, down_rank, front_rank, back_rank, right_rank, left_rank;
int dim[3] = {0, 0, 0};
int coords[3];
int nprocs, rank;

std::ofstream out;

float an_sol(float t, float x, float y, float z) {
    return sin(2*PI*x/Lx)*sin(PI*y/Ly)*sin(PI*z/Lz)*cos(a_t*t + 2*PI);
}




void prepare_layer(int n) { // граничные значения
    // y, z
    float loop_start = MPI_Wtime();
    #pragma omp parallel for num_threads(Np) shared(grids, front_rank, back_rank, left_rank, right_rank, n, X, Y, Z)
    for(int i = 1; i <= X; i++) {
        for(int j = 1; j <= Y; j++) {
            if(back_rank < 0) grids[n % 3].set(i, j, 1, 0.0);
            if(front_rank < 0) grids[n % 3].set(i, j, Z, 0.0);
        }
        for(int k = 1; k <= Z; k++) {
            if(left_rank < 0) grids[n % 3].set(i, 1, k, 0.0);
            if(right_rank < 0) grids[n % 3].set(i, Y, k, 0.0);
        }
    }
    int y = coords[1] == dim[1]-1 ? Y-1 : Y, z = coords[2] == dim[2]-1 ? Z-1 : Z;
    int j1 = coords[1] == 0 ? 2 : 1, k1 = coords[2] == 0 ? 2 : 1;
    bool b1 = coords[0] == dim[0]-1, b2 = coords[0] == 0;
    if(b1 || b2) {
        #pragma omp parallel for num_threads(Np) shared(grids, coords, X, n, y, z, j1, k1, b1, b2)
        for(int j = j1; j <= y; j++) {
            for(int k = k1; k <= z; k++) {
                if(b1) grids[n % 3].set(X, j, k, (n > 1 ? 2 : 1)*grids[(n + 2) % 3].get(X, j, k) - (n > 1 ? grids[(n + 1) % 3].get(X, j, k) : 0) + (n > 1 ? 1.0 : 0.5)*a2*tau*tau*grids[(n+2) % 3].laplace(X, j, k, X-1, j-1, k-1, X+1, j+1, k+1));
                if(b2) grids[n % 3].set(1, j, k, (n > 1 ? 2 : 1)*grids[(n + 2) % 3].get(1, j, k) - (n > 1 ? grids[(n + 1) % 3].get(1, j, k) : 0) + (n > 1 ? 1.0 : 0.5)*a2*tau*tau*grids[(n+2) % 3].laplace(1, j, k, 0, j-1, k-1, 2, j+1, k+1));
            }
        }
    }
    float loop_duration = MPI_Wtime() - loop_start;
    total_loop_time += loop_duration;
}

void exchange(int n) {
    MPI_Status status;
    // prepare matrices to send
    for(int j = 1; j <= Y; j++) {
        for(int k = 1; k <= Z; k++) {
            up1.array[j*(Z+2) + k] = grids[n % 3].get(coords[0] == dim[0]-1 ? X-1 : X, j, k);
            down1.array[j*(Z+2) + k] = grids[n % 3].get(coords[0] == 0 ? 2 : 1, j, k);
        }
    }
    for(int i = 1; i <= X; i++) {
        for(int j = 1; j <= Y; j++) {
            front1.array[i*(Y+2) + j] = grids[n % 3].get(i, j, Z);
            back1.array[i*(Y+2) + j] = grids[n % 3].get(i, j, 1);
        }
        for(int k = 1; k <= Z; k++) {
            right1.array[i*(Z+2) + k] = grids[n % 3].get(i, Y, k);
            left1.array[i*(Z+2) + k] = grids[n % 3].get(i, 1, k);
        }
    }
    float exchange_start = MPI_Wtime();
    if(up_rank >= 0 && down_rank >= 0) {
        MPI_Sendrecv(up1.array, (Z+2)*(Y+2), MPI_FLOAT, up_rank, 1, 
                     down.array, (Z+2)*(Y+2), MPI_FLOAT, down_rank, 1, cart_comm, &status);
        MPI_Sendrecv(down1.array, (Z+2)*(Y+2), MPI_FLOAT, down_rank, 2, 
                     up.array, (Z+2)*(Y+2), MPI_FLOAT, up_rank, 2, cart_comm, &status);
    }
    if(right_rank >= 0 && left_rank >= 0) {
        MPI_Sendrecv(right1.array, (Z+2)*(X+2), MPI_FLOAT, right_rank, 3, 
                     ::left.array, (Z+2)*(X+2), MPI_FLOAT, left_rank, 3, cart_comm, &status);
    } else if (right_rank >= 0) {
        MPI_Send(right1.array, (Z+2)*(X+2), MPI_FLOAT, right_rank, 3, cart_comm);
    } else if(left_rank >= 0) {
        MPI_Recv(::left.array, (Z+2)*(X+2), MPI_FLOAT, left_rank, 3, cart_comm, &status);
    }
    if(right_rank >= 0 && left_rank >= 0) {
        MPI_Sendrecv(left1.array, (Z+2)*(X+2), MPI_FLOAT, left_rank, 4, 
                     ::right.array, (Z+2)*(X+2), MPI_FLOAT, right_rank, 4, cart_comm, &status);
    } else if (left_rank >= 0) {
        MPI_Send(left1.array, (Z+2)*(X+2), MPI_FLOAT, left_rank, 4, cart_comm);
    } else if(right_rank >= 0) {
        MPI_Recv(::right.array, (Z+2)*(X+2), MPI_FLOAT, right_rank, 4, cart_comm, &status);
    }
    if(front_rank >= 0 && back_rank >= 0) {
        MPI_Sendrecv(front1.array, (Y+2)*(X+2), MPI_FLOAT, front_rank, 5, 
                     back.array, (Y+2)*(X+2), MPI_FLOAT, back_rank, 5, cart_comm, &status);
    } else if (front_rank >= 0) {
        MPI_Send(front1.array, (Y+2)*(X+2), MPI_FLOAT, front_rank, 5, cart_comm);
    } else if(back_rank >= 0) {
        MPI_Recv(back.array, (Y+2)*(X+2), MPI_FLOAT, back_rank, 5, cart_comm, &status);
    }
    if(front_rank >= 0 && back_rank >= 0) {
        MPI_Sendrecv(back1.array, (Y+2)*(X+2), MPI_FLOAT, back_rank, 6, 
                     front.array, (Y+2)*(X+2), MPI_FLOAT, front_rank, 6, cart_comm, &status);
    } else if (back_rank >= 0) {
        MPI_Send(back1.array, (Y+2)*(X+2), MPI_FLOAT, back_rank, 6, cart_comm);
    } else if(front_rank >= 0) {
        MPI_Recv(front.array, (Y+2)*(X+2), MPI_FLOAT, front_rank, 6, cart_comm, &status);
    }
    float exchange_duration = MPI_Wtime() - exchange_start;
    total_exchange_time += exchange_duration;

    // copy matrices
    if(left_rank >= 0) for(int i = 1; i <= X; i++) for(int k = 1; k <= Z; k++) {
        grids[n % 3].set(i, 0, k, ::left.array[i*(Z+2) + k]);
    }
    if(right_rank >= 0) for(int i = 1; i <= X; i++) for(int k = 1; k <= Z; k++) {
        grids[n % 3].set(i, Y+1, k, ::right.array[i*(Z+2) + k]);
    }
    if(front_rank >= 0) for(int i = 1; i <= X; i++) for(int j = 1; j <= Y; j++) {
        grids[n % 3].set(i, j, Z+1, ::front.array[i*(Y+2) + j]);
    }
    if(back_rank >= 0) for(int i = 1; i <= X; i++) for(int j = 1; j <= Y; j++) {
        grids[n % 3].set(i, j, 0, ::back.array[i*(Y+2) + j]);
    }

    if(up_rank >= 0) {
        for(int j = 1; j <= Y; j++) {
            for(int k = 1; k <= Z; k++) {
                grids[n % 3].set(X+1, j, k, up.array[j*(Z+2) + k]);
            }
        }
    }
    if(down_rank >= 0) {
        for(int j = 1; j <= Y; j++) {
            for(int k = 1; k <= Z; k++) {
                grids[n % 3].set(0, j, k, ::down.array[j*(Z+2) + k]);
            }
        }
    }
}

void calculate_start() {
    // n = 0
    float loop_start = MPI_Wtime();
    #pragma omp parallel for num_threads(Np) shared(grids, X, Y, Z)
    for(int i = 0; i <= X-1; i++) {
        for(int j = 0; j <= Y-1; j++) {
            for(int k = 0; k <= Z-1; k++) {
                float u = an_sol(0, (i+x_0)*hx, (j+y_0)*hy, (k+z_0)*hz);
                grids[0].set(i+1, j+1, k+1, u);
                float f = an_sol(0, hx*(i+x_0), hy*(j+y_0), hz*(k+z_0));
                float abs_error = fabs(u - f);
                float rel_error = fabs((u - f) / f);
                if(abs_error > max_abs_error) max_abs_error = abs_error;
                if(rel_error > max_rel_error) max_rel_error = rel_error;
            }
        }
    }
    float loop_duration = MPI_Wtime() - loop_start;
    total_loop_time += loop_duration;
    max_abs_errors.push_back(max_abs_error);
    max_rel_errors.push_back(max_rel_error);
    exchange(0);
    // n = 1
    max_abs_error = -100.0;
    max_rel_error = -100.0;
    prepare_layer(1);
    int x = coords[0] == dim[0]-1 ? X-1 : X,
        y = coords[1] == dim[1]-1 ? Y-1 : Y,
        z = coords[2] == dim[2]-1 ? Z-1 : Z;
    loop_start = MPI_Wtime();
    int i1 = coords[0] == 0 ? 2 : 1,
        j1 = coords[1] == 0 ? 2 : 1,
        k1 = coords[2] == 0 ? 2 : 1;
    #pragma omp parallel for num_threads(Np) shared(grids, x, y, z, i1, j1, k1)
    for(int i = i1; i <= x; i++) {
        for(int j = j1; j <= y; j++) {
            for(int k = k1; k <= z; k++) {
                float u = grids[0].get(i, j, k) + a2*tau*tau*0.5*grids[0].laplace(i, j, k, i-1, j-1, k-1, i+1, j+1, k+1);
                grids[1].set(i, j, k, u);
                float f = an_sol(tau, hx*(i-1+x_0), hy*(j-1+y_0), hz*(k-1+z_0));
                float abs_error = fabs(u - f);
                float rel_error = fabs((u - f) / f);
                if(abs_error > max_abs_error) max_abs_error = abs_error;
                if(rel_error > max_rel_error) max_rel_error = rel_error;
            }
        }
    }
    loop_duration = MPI_Wtime() - loop_start;
    total_loop_time += loop_duration;
    max_abs_errors.push_back(max_abs_error);
    max_rel_errors.push_back(max_rel_error);
    exchange(1);
}




void calculate_num_sol() {
    //std::cout << "calculating num sol\n";
    float start = MPI_Wtime();
    calculate_start();
    for(int n = 2; n <= timesteps; n++) {
        if(rank == 0) std::cout << "calculating layer " << n << std::endl;
        max_abs_error = -100.0;
        max_rel_error = -100.0;
        prepare_layer(n);
        //cout << "prepared\n";
        //cout << num_sol.size() << " " << num_sol[0].size() << " " << num_sol[0][0].size() << " " << num_sol[0][0][0].size() << endl;
        //#pragma omp parallel for num_threads(Np) shared(num_sol, n)
        int x = coords[0] == dim[0]-1 ? X-1 : X,
            y = coords[1] == dim[1]-1 ? Y-1 : Y,
            z = coords[2] == dim[2]-1 ? Z-1 : Z;
            int i1 = coords[0] == 0 ? 2 : 1,
            j1 = coords[1] == 0 ? 2 : 1,
            k1 = coords[2] == 0 ? 2 : 1;
        float loop_start = MPI_Wtime();
        #pragma omp parallel for num_threads(Np) shared(grids, x, y, z, i1, j1, k1, n) collapse(3)
        for(int i = i1; i <= x; i++) {
            for(int j = j1; j <= y; j++) {
                for(int k = k1; k <= z; k++) {
                    float u = 2*grids[(n+2) % 3].get(i, j, k) - grids[(n+1) % 3].get(i, j, k) + a2*tau*tau*grids[(n+2) % 3].laplace(i, j, k, i-1, j-1, k-1, i+1, j+1, k+1);
                    grids[n % 3].set(i, j, k, u);
                    float f = an_sol(tau*n, hx*(i-1+x_0), hy*(j-1+y_0), hz*(k-1+z_0));
                    float abs_error = fabs(u - f);
                    float rel_error = fabs((u - f) / f);
                    if(abs_error > max_abs_error) max_abs_error = abs_error;
                    if(rel_error > max_rel_error) max_rel_error = rel_error;
                }
            }
        }
        float loop_duration = MPI_Wtime() - loop_start;
        total_loop_time += loop_duration;
        max_abs_errors.push_back(max_abs_error);
        max_rel_errors.push_back(max_rel_error);
        if(n < timesteps) exchange(n);
    }
    float duration = MPI_Wtime() - start;
    if(rank == 0) {
        out << "numerical solution calculated in " << (unsigned) (duration*1000) << "ms" << std::endl;
    }
    std::vector<float> global_max_abs_errors(timesteps + 1);
    std::vector<float> global_max_rel_errors(timesteps + 1);
    MPI_Reduce(max_abs_errors.data(), global_max_abs_errors.data(), timesteps + 1, MPI_FLOAT, MPI_MAX, 0, cart_comm);
    MPI_Reduce(max_rel_errors.data(), global_max_rel_errors.data(), timesteps + 1, MPI_FLOAT, MPI_MAX, 0, cart_comm);
    if(rank == 0) {
        for(int n = 0; n <= timesteps; n++) {
            out << "max abs and rel errors on layer " << n << ": " << global_max_abs_errors[n] << " " << global_max_rel_errors[n] << std::endl;
        }
        //std::cout << "errors calculated\n";
    }
    if(rank == 0) {
        out << "total MPI exchange time: " << (unsigned) (total_exchange_time*1000) << "ms\n";
        out << "total loop time: " << (unsigned) (total_loop_time*1000) << "ms\n";
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
        //std::cout << "a_t = " << a_t << std::endl;
        std::cout << "C = " << sqrt(a2)*tau/std::min(hx, std::min(hy, hz)) << std::endl;
    }
    
    // creating virtual topology
    
    MPI_Dims_create(nprocs, 3, dim);
    
    int periods[3] = {true, false, false};
    MPI_Cart_create(MPI_COMM_WORLD, 3, dim, periods, false, &cart_comm);
    //if(rank == 0) std::cout << "virtual topology dims: " << dim[0] << " " << dim[1] << " " << dim[2] << std::endl;
    //MPI_Barrier(MPI_COMM_WORLD);

    
    MPI_Cart_coords(cart_comm, rank, 3, coords);
    //std::cout << "process " << rank << " coords: " << coords[0] << " " << coords[1] << " " << coords[2] << std::endl;
    X = (N+1) / dim[0], Y = (N+1) / dim[1], Z = (N+1) / dim[2];
    x_0 = coords[0]*X, y_0 = coords[1]*Y, z_0 = coords[2]*Z; // grid starting points
    if(coords[0] == dim[0]-1) X += (N+1) % dim[0];
    if(coords[1] == dim[1]-1) Y += (N+1) % dim[1];
    if(coords[2] == dim[2]-1) Z += (N+1) % dim[2];
    

    //std::cout << rank << " " << coords[0] << " " << coords[1] << " " << coords[2] << " " << X << " " << Y << " " << Z << std::endl;
    //std::cout << rank << " " << x_0 << " " << y_0 << " " << z_0 << std::endl;

    // grid coords: ([x_0, x_0 + X - 1], [y_0, y_0 + Y - 1], [z_0, z_0 + Z - 1])
    // + 2 side coords for neighbour's values
    MPI_Cart_shift(cart_comm, 0, 1, &down_rank, &up_rank); // down to top
    MPI_Cart_shift(cart_comm, 1, 1, &left_rank, &right_rank);
    MPI_Cart_shift(cart_comm, 2, 1, &back_rank, &front_rank);
    //std::cout << rank << " down and up: " << down_rank << " " << up_rank << std::endl;
    //std::cout << rank << " left and right: " << left_rank << " " << right_rank << std::endl;
    //std::cout << rank << " back and front: " << back_rank << " " << front_rank << std::endl;

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

    if(rank == 0) {
        out.clear();
        out.open(std::string("output_N") + std::to_string(N) + "_Np" + std::to_string(nprocs) + "_Nt" + std::to_string(Np) + std::string("_hyb.txt"));
    }
    
    float init_start = MPI_Wtime();
    grids[0].init_grid(X + 2, Y + 2, Z + 2);
    grids[1].init_grid(X + 2, Y + 2, Z + 2);
    grids[2].init_grid(X + 2, Y + 2, Z + 2);
    // cur_prec_sol.init_grid(X + 2, Y + 2, Z + 2);
    // num_sol.init_grid(timesteps+2, X + 2, Y + 2, Z + 2);
    cur_abs_error.init_grid(X, Y, Z);
    cur_rel_error.init_grid(X, Y, Z);
    up.init(Y+2, Z+2); up1.init(Y+2, Z+2);
    down.init(Y+2, Z+2); down1.init(Y+2, Z+2);
    ::right.init(X+2, Z+2); right1.init(X+2, Z+2);
    ::left.init(X+2, Z+2); left1.init(X+2, Z+2);
    front.init(X+2, Y+2); front1.init(X+2, Y+2);
    back.init(X+2, Y+2); back1.init(X+2, Y+2);
    //std::cout << "grids initialized\n";
    float init_duration = MPI_Wtime() - init_start;
    if(rank == 0) {
        out << "grids initialized in " << (unsigned) (init_duration*1000) << "ms" << std::endl;
    }

    
    omp_set_num_threads(Np);
    
    calculate_num_sol();


    if(rank == 0) out.close();


    MPI_Finalize();

    return 0;
}