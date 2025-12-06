#include <iostream>
#include <cmath>
#include <vector>
#include <ctime>
#include <string>
#include <cstring>
#include <fstream>
#include <chrono>
#include <mpi.h>
#include <omp.h>

//using namespace std;



double Lx, Ly, Lz, a_t, a2, T;
double tau, hx, hy, hz;

unsigned long Np;

const double PI =  3.1415926535;

int N = 32;
int X, Y, Z;
int x_0, y_0, z_0;
int timesteps;

class Matrix {
private:
    
    int x, y;
public:
    double * array;
    Matrix() {}
    Matrix(int x_, int y_) {
        x = x_;
        y = y_;
        array = new double[x*y];
        for(unsigned i = 0; i < x*y; i++) array[i] = 0.0;
    }
    void init(int x_, int y_) {
        x = x_;
        y = y_;
        array = new double[x*y];
        for(unsigned i = 0; i < x*y; i++) array[i] = 0.0;
    }
    double * get_ptr() {
        return array;
    }
};

class Grid {
private:
    double * array;
    int n, x, y, z;
public:
    Grid() {
        
    }

    Grid(int n_, int x_, int y_, int z_) {
        n = n_;
        x = x_;
        y = y_;
        z = z_;
        array = new double[n*x*y*z];
        for(unsigned i = 0; i < n*x*y*z; i++) array[i] = 0.0;
    }

    void init_grid(int n_, int x_, int y_, int z_) {
        n = n_;
        x = x_;
        y = y_;
        z = z_;
        array = new double[n*x*y*z];
        for(unsigned i = 0; i < n*x*y*z; i++) array[i] = 0.0;
    }

    double get(int t, int i, int j, int k) {
        return array[t*x*y*z + i*y*z + j*z + k];
    }

    void set(int t, int i, int j, int k, double val) {
        array[t*x*y*z + i*y*z + j*z + k] = val;
    }

    double * get_layer(int t) {
        return array + t*x*y*z;
    }

    void copy_from_matrices(int t, Matrix & front, Matrix & back, Matrix & up, Matrix & down, Matrix & right, Matrix & left) {
        for(int i = 0; i < x; i++) {
            for(int j = 0; j < y; j++) { // front and back
                array[t*x*y*z + i*y*z + j*z + z-1] = front.array[i*y + j];
                array[t*x*y*z + i*y*z + j*z + 0] = back.array[i*y + j];
            }
            for(int k = 0; k < z; k++) { // right and left
                array[t*x*y*z + i*y*z + (y-1)*z + k] = right.array[i*z + k];
                array[t*x*y*z + i*y*z + 0*z + k] = left.array[i*z + k];
            }
        }
        for(int j = 0; j < y; j++) {
            for(int k = 0; k < z; k++) { // up and down
                array[t*x*y*z + (x-1)*y*z + j*z + k] = up.array[j*z + k];
                array[t*x*y*z + 0*y*z + j*z + k] = down.array[j*z + k];
            }
        }
    }

    double laplace(int n, int i, int j, int k, int x1, int y1, int z1, int x2, int y2, int z2) {
        double ans = 0.0;
        ans += (array[n*x*y*z + x1*y*z + j*z + k] - 2*array[n*x*y*z + i*y*z + j*z + k] + array[n*x*y*z + x2*y*z + j*z + k])/(hx*hx);
        ans += (array[n*x*y*z + i*y*z + y1*z + k] - 2*array[n*x*y*z + i*y*z + j*z + k] + array[n*x*y*z + i*y*z + y2*z + k])/(hy*hy);
        ans += (array[n*x*y*z + i*y*z + j*z + z1] - 2*array[n*x*y*z + i*y*z + j*z + k] + array[n*x*y*z + i*y*z + j*z + z2])/(hz*hz);
        return ans;
    }

    void print_layer(int t) {
        std::cout << "Layer " << t << ":\n";
        for(int i = 0; i < x; i++) {
            for(int j = 0; j < y; j++) {
                for(int k = 0; k < z; k++) {
                    std::cout << array[t*x*y*z + i*y*z + j*z + k] << " ";
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

Grid prec_sol; // sol[t][x][y][z]
Grid num_sol; // sol[t][x][y][z]
std::vector<double> max_abs_errors;
std::vector<double> max_rel_errors;

MPI_Comm cart_comm;
Matrix up, down, front, back, right, left; // Matrices that are to be received from neighbors and copied into grid (0, X + 1)
Matrix up1, down1, front1, back1, right1, left1; // Matrices that are to be copied from grid and sent to neighbors (1, X)
int up_rank, down_rank, front_rank, back_rank, right_rank, left_rank;
int dim[3] = {0, 0, 0};
int coords[3];
int nprocs, rank;

std::ofstream out;

double an_sol(double t, double x, double y, double z) {
    return sin(2*PI*x/Lx)*sin(PI*y/Ly)*sin(PI*z/Lz)*cos(a_t*t + 2*PI);
}



void calculate_an_sol() {
    auto start = std::chrono::high_resolution_clock::now();
    for(int n = 0; n <= timesteps; n++) {
        for(int i = 0; i < X; i++) { // if nproc = 1 then X == N + 1
            for(int j = 0; j < Y; j++) {
                for(int k = 0; k < Z; k++) {
                    double f = an_sol(tau*n, hx*(i+x_0), hy*(j+y_0), hz*(k+z_0));
                    prec_sol.set(n, i+1, j+1, k+1, f);
                }
            }
        }
    }
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();
}

void prepare_layer(int n) {
    // y, z
    //#pragma omp parallel for num_threads(Np) shared(num_sol, front_rank, back_rank, left_rank, right_rank, n, X, Y, Z)
    for(int i = 1; i <= X; i++) {
        for(int j = 1; j <= Y; j++) {
            if(back_rank < 0) num_sol.set(n, i, j, 1, 0.0);
            if(front_rank < 0) num_sol.set(n, i, j, Z, 0.0);
        }
        for(int k = 1; k <= Z; k++) {
            if(left_rank < 0) num_sol.set(n, i, 1, k, 0.0);
            if(right_rank < 0) num_sol.set(n, i, Y, k, 0.0);
        }
    }
    int y = coords[1] == dim[1]-1 ? Y-1 : Y, z = coords[2] == dim[2]-1 ? Z-1 : Z;
    int j1 = coords[1] == 0 ? 2 : 1, k1 = coords[2] == 0 ? 2 : 1;
    bool b1 = coords[0] == dim[0]-1, b2 = coords[0] == 0;
    if(b1 || b2) {
    //#pragma omp parallel for num_threads(Np) shared(num_sol, coords, X, n, y, z, j1, k1, b1, b2)
        for(int j = j1; j <= y; j++) {
            for(int k = k1; k <= z; k++) {
                if(b1) num_sol.set(n, X, j, k, (n > 1 ? 2 : 1)*num_sol.get(n-1, X, j, k) - (n > 1 ? num_sol.get(n-2, X, j, k) : 0) + (n > 1 ? 1.0 : 0.5)*a2*tau*tau*num_sol.laplace(n-1, X, j, k, X-1, j-1, k-1, X+1, j+1, k+1));
                if(b2) num_sol.set(n, 1, j, k, (n > 1 ? 2 : 1)*num_sol.get(n-1, 1, j, k) - (n > 1 ? num_sol.get(n-2, 1, j, k) : 0) + (n > 1 ? 1.0 : 0.5)*a2*tau*tau*num_sol.laplace(n-1, 1, j, k, 0, j-1, k-1, 2, j+1, k+1));
            }
        }
    }
}

void exchange(int n) {
    //std::cout << "exchange " << n << std::endl;
    MPI_Status status;
    // prepare matrices to send
    #pragma omp parallel for num_threads(Np) shared(up1, down1, num_sol, coords, dim, X, Y, Z, n)
    for(int j = 1; j <= Y; j++) {
        for(int k = 1; k <= Z; k++) {
            up1.array[j*(Z+2) + k] = num_sol.get(n, coords[0] == dim[0]-1 ? X-1 : X, j, k);
            down1.array[j*(Z+2) + k] = num_sol.get(n, coords[0] == 0 ? 2 : 1, j, k);
        }
    }
    #pragma omp parallel for num_threads(Np) shared(front1, back1, right1, left1, num_sol, n, Y, Z, X)
    for(int i = 1; i <= X; i++) {
        for(int j = 1; j <= Y; j++) {
            front1.array[i*(Y+2) + j] = num_sol.get(n, i, j, Z);
            back1.array[i*(Y+2) + j] = num_sol.get(n, i, j, 1);
        }
        for(int k = 1; k <= Z; k++) {
            right1.array[i*(Z+2) + k] = num_sol.get(n, i, Y, k);
            left1.array[i*(Z+2) + k] = num_sol.get(n, i, 1, k);
        }
    }
    //cout << "buf matrices prepared\n";
    if(up_rank >= 0 && down_rank >= 0) {
        MPI_Sendrecv(up1.array, (Z+2)*(Y+2), MPI_DOUBLE, up_rank, 1, 
                     down.array, (Z+2)*(Y+2), MPI_DOUBLE, down_rank, 1, cart_comm, &status);
        MPI_Sendrecv(down1.array, (Z+2)*(Y+2), MPI_DOUBLE, down_rank, 2, 
                     up.array, (Z+2)*(Y+2), MPI_DOUBLE, up_rank, 2, cart_comm, &status);
    }
    //std::cout << "proc " << rank << " ex " << n << " up-down exchange complete\n";
    // left-right
    if(right_rank >= 0 && left_rank >= 0) {
        MPI_Sendrecv(right1.array, (Z+2)*(X+2), MPI_DOUBLE, right_rank, 3, 
                     ::left.array, (Z+2)*(X+2), MPI_DOUBLE, left_rank, 3, cart_comm, &status);
    } else if (right_rank >= 0) {
        MPI_Send(right1.array, (Z+2)*(X+2), MPI_DOUBLE, right_rank, 3, cart_comm);
    } else if(left_rank >= 0) {
        MPI_Recv(::left.array, (Z+2)*(X+2), MPI_DOUBLE, left_rank, 3, cart_comm, &status);
    }
    //std::cout << "proc " << rank << " ex " << n << " left-right exchange complete\n";
    // right-left
    if(right_rank >= 0 && left_rank >= 0) {
        MPI_Sendrecv(left1.array, (Z+2)*(X+2), MPI_DOUBLE, left_rank, 4, 
                     ::right.array, (Z+2)*(X+2), MPI_DOUBLE, right_rank, 4, cart_comm, &status);
    } else if (left_rank >= 0) {
        MPI_Send(left1.array, (Z+2)*(X+2), MPI_DOUBLE, left_rank, 4, cart_comm);
    } else if(right_rank >= 0) {
        MPI_Recv(::right.array, (Z+2)*(X+2), MPI_DOUBLE, right_rank, 4, cart_comm, &status);
    }
    //std::cout << "proc " << rank << " ex " << n << " right-left exchange complete\n";
    // back-front
    if(front_rank >= 0 && back_rank >= 0) {
        MPI_Sendrecv(front1.array, (Y+2)*(X+2), MPI_DOUBLE, front_rank, 5, 
                     back.array, (Y+2)*(X+2), MPI_DOUBLE, back_rank, 5, cart_comm, &status);
    } else if (front_rank >= 0) {
        MPI_Send(front1.array, (Y+2)*(X+2), MPI_DOUBLE, front_rank, 5, cart_comm);
    } else if(back_rank >= 0) {
        MPI_Recv(back.array, (Y+2)*(X+2), MPI_DOUBLE, back_rank, 5, cart_comm, &status);
    }
    //std::cout << "proc " << rank << " ex " << n << " back-front exchange complete\n";
    // front-back
    if(front_rank >= 0 && back_rank >= 0) {
        MPI_Sendrecv(back1.array, (Y+2)*(X+2), MPI_DOUBLE, back_rank, 6, 
                     front.array, (Y+2)*(X+2), MPI_DOUBLE, front_rank, 6, cart_comm, &status);
    } else if (back_rank >= 0) {
        MPI_Send(back1.array, (Y+2)*(X+2), MPI_DOUBLE, back_rank, 6, cart_comm);
    } else if(front_rank >= 0) {
        MPI_Recv(front.array, (Y+2)*(X+2), MPI_DOUBLE, front_rank, 6, cart_comm, &status);
    }
    //std::cout << "proc " << rank << " ex " << n << " front-back exchange complete\n";
    //std::cout << "proc " << rank << " ex " << n << " side exchanges complete\n";
    // copy matrices
    if(left_rank >= 0) {
        //#pragma omp parallel for num_threads(Np) shared(num_sol, n, left, X, Z, Y)
        for(int i = 1; i <= X; i++) for(int k = 1; k <= Z; k++) {
            num_sol.set(n, i, 0, k, ::left.array[i*(Z+2) + k]);
        }
    }
    if(right_rank >= 0) {
        //#pragma omp parallel for num_threads(Np) shared(num_sol, n, right, Y, Z, X)
        for(int i = 1; i <= X; i++) for(int k = 1; k <= Z; k++) {
            num_sol.set(n, i, Y+1, k, ::right.array[i*(Z+2) + k]);
        }
    }
    if(front_rank >= 0) {
        //#pragma omp parallel for num_threads(Np) shared(num_sol, n, front, Y, Z, X)
        for(int i = 1; i <= X; i++) for(int j = 1; j <= Y; j++) {
            num_sol.set(n, i, j, Z+1, ::front.array[i*(Y+2) + j]);
        }
    }
    if(back_rank >= 0) {
        //#pragma omp parallel for num_threads(Np) shared(num_sol, n, back, Y, Z, X)
        for(int i = 1; i <= X; i++) for(int j = 1; j <= Y; j++) {
            num_sol.set(n, i, j, 0, ::back.array[i*(Y+2) + j]);
        }
    }
    //cout << "yo\n";
    if(up_rank >= 0) {
        //#pragma omp parallel for num_threads(Np) shared(num_sol, n, up, Y, Z, X)
        for(int j = 1; j <= Y; j++) {
            for(int k = 1; k <= Z; k++) {
                num_sol.set(n, X+1, j, k, up.array[j*(Z+2) + k]);
            }
        }
    }
    if(down_rank >= 0) {
        //#pragma omp parallel for num_threads(Np) shared(num_sol, n, down, Y, Z, X)
        for(int j = 1; j <= Y; j++) {
            for(int k = 1; k <= Z; k++) {
                num_sol.set(n, 0, j, k, ::down.array[j*(Z+2) + k]);
            }
        }
    }
    //cout << "matrices copied to num_sol\n";
    //std::cout << "exchange " << n << " complete\n";
}

void calculate_start() { // начальные значения
    // n = 0
    //if(rank == 0) std::cout << "calculating layer " << 0 << std::endl;
    #pragma omp parallel for num_threads(Np) shared(num_sol, Y, Z, X) collapse(3)
    for(int i = 0; i <= X-1; i++) {
        for(int j = 0; j <= Y-1; j++) {
            for(int k = 0; k <= Z-1; k++) {
                num_sol.set(0, i+1, j+1, k+1, an_sol(0, (i+x_0)*hx, (j+y_0)*hy, (k+z_0)*hz));
            }
        }
    }
    exchange(0);
    //num_sol.print_layer(0);
    // n = 1
    //if(rank == 0) std::cout << "calculating layer " << 1 << std::endl;
    prepare_layer(1);
    int x = coords[0] == dim[0]-1 ? X-1 : X,
        y = coords[1] == dim[1]-1 ? Y-1 : Y,
        z = coords[2] == dim[2]-1 ? Z-1 : Z;
    int i1 = coords[0] == 0 ? 2 : 1,
        j1 = coords[1] == 0 ? 2 : 1,
        k1 = coords[2] == 0 ? 2 : 1;
    #pragma omp parallel for num_threads(Np) shared(num_sol, x, y, z, i1, j1, k1) collapse(3)
    for(int i = i1; i <= x; i++) {
        for(int j = j1; j <= y; j++) {
            for(int k = k1; k <= z; k++) {
                num_sol.set(1, i, j, k, num_sol.get(0, i, j, k) + a2*tau*tau*0.5*num_sol.laplace(0, i, j, k, i-1, j-1, k-1, i+1, j+1, k+1));
            }
        }
    }
    exchange(1);
    //num_sol.print_layer(1);
}




void calculate_num_sol() {
    //std::cout << "calculating num sol\n";
    double start = MPI_Wtime();
    double start1 = omp_get_wtime();
    calculate_start();
    for(int n = 2; n <= timesteps; n++) {
        //if(rank == 0) std::cout << "calculating layer " << n << std::endl;
        prepare_layer(n);
        //std::cout << "prepared\n";
        //cout << num_sol.size() << " " << num_sol[0].size() << " " << num_sol[0][0].size() << " " << num_sol[0][0][0].size() << endl;
        int x = coords[0] == dim[0]-1 ? X-1 : X,
            y = coords[1] == dim[1]-1 ? Y-1 : Y,
            z = coords[2] == dim[2]-1 ? Z-1 : Z;
        int i1 = coords[0] == 0 ? 2 : 1,
            j1 = coords[1] == 0 ? 2 : 1,
            k1 = coords[2] == 0 ? 2 : 1;
        #pragma omp parallel for num_threads(Np) shared(num_sol, x, y, z, i1, j1, k1, n) collapse(3)
        for(int i = i1; i <= x; i++) {
            for(int j = j1; j <= y; j++) {
                for(int k = k1; k <= z; k++) {
                    num_sol.set(n, i, j, k, 2*num_sol.get(n-1, i, j, k) - num_sol.get(n-2, i, j, k) + a2*tau*tau*num_sol.laplace(n-1, i, j, k, i-1, j-1, k-1, i+1, j+1, k+1));
                }
            }
        }
        if(n < timesteps) exchange(n);
    }
    MPI_Barrier(cart_comm);
    double duration = MPI_Wtime() - start;
    double duration1 = omp_get_wtime() - start;
    if(rank == 0) {
        out << "numerical solution calculated in " << (unsigned) (duration*1000) << "ms" << std::endl;
    }
}

void calculate_error() {
    max_abs_errors.clear();
    max_rel_errors.clear();
    for(int n = 0; n <= timesteps; n++) {
        double max_rel = 0.0, max_abs = 0.0;
        int x = coords[0] == dim[0]-1 ? X-1 : X,
            y = coords[1] == dim[1]-1 ? Y-1 : Y,
            z = coords[2] == dim[2]-1 ? Z-1 : Z;
        for(int i = coords[0] == 0 ? 2 : 1; i <= x; i++) {
            for(int j = coords[1] == 0 ? 2 : 1; j <= y; j++) {
                for(int k = coords[2] == 0 ? 2 : 1; k <= z; k++) {
                    double a = fabs(prec_sol.get(n, i, j, k) - num_sol.get(n, i, j, k));
                    double r = fabs(prec_sol.get(n, i, j, k) - num_sol.get(n, i, j, k)) / fabs(prec_sol.get(n, i, j, k));
                    max_rel = fmax(max_rel, r);
                    max_abs = fmax(max_abs, a);
                }
            }
        }
        max_abs_errors.push_back(max_abs);
        max_rel_errors.push_back(max_rel);
        //std::cout << rank << " max abs and rel errors on layer " << n << ": " << max_abs << " " << max_rel << std::endl;
    }
    std::vector<double> global_max_abs_errors(timesteps + 1);
    std::vector<double> global_max_rel_errors(timesteps + 1);
    MPI_Reduce(max_abs_errors.data(), global_max_abs_errors.data(), timesteps + 1, MPI_DOUBLE, MPI_MAX, 0, cart_comm);
    MPI_Reduce(max_rel_errors.data(), global_max_rel_errors.data(), timesteps + 1, MPI_DOUBLE, MPI_MAX, 0, cart_comm);
    if(rank == 0) {
        for(int n = 0; n <= timesteps; n++) {
            out << "max abs and rel errors on layer " << n << ": " << global_max_abs_errors[n] << " " << global_max_rel_errors[n] << std::endl;
        }
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

    prec_sol.init_grid(timesteps+2, X + 2, Y + 2, Z + 2);
    num_sol.init_grid(timesteps+2, X + 2, Y + 2, Z + 2);
    up.init(Y+2, Z+2); up1.init(Y+2, Z+2);
    down.init(Y+2, Z+2); down1.init(Y+2, Z+2);
    ::right.init(X+2, Z+2); right1.init(X+2, Z+2);
    ::left.init(X+2, Z+2); left1.init(X+2, Z+2);
    front.init(X+2, Y+2); front1.init(X+2, Y+2);
    back.init(X+2, Y+2); back1.init(X+2, Y+2);
    //std::cout << "grids initialized\n";

    //Np = 1;
    calculate_an_sol();
    //std::cout << "an sol calculated" << std::endl;

    if(rank == 0) {
        out.clear();
        out.open(std::string("output_N") + std::to_string(N) + "_Np" + std::to_string(nprocs) + std::string("_Nt") + std::to_string(Np) + std::string("_hyb.txt"));
    }

    omp_set_num_threads(Np);
    
    calculate_num_sol();
    calculate_error();

    if(rank == 0) out.close();

    MPI_Finalize();

    return 0;
}