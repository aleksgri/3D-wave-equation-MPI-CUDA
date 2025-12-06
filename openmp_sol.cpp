#include <iostream>
#include <cmath>
#include <vector>
#include <omp.h>
#include <ctime>
#include <string>
#include <cstring>
#include <fstream>
#include <chrono>

using namespace std;



double Lx, Ly, Lz, a_t, a2, T;
double tau, hx, hy, hz;

unsigned long Np;

const double PI =  3.1415926535;

int N = 32;
int timesteps;

class Grid {
private:
    double * array;
    int size;
public:
    Grid() {
        
    }

    Grid(int n, int size_) {
        size = size_;
        array = new double[n*size*size*size];
    }

    void init_grid(int n, int size_) {
        size = size_;
        array = new double[n*size*size*size];
    }

    double get(int t, int x, int y, int z) {
        return array[t*size*size*size + x*size*size + y*size + z];
    }

    void set(int t, int x, int y, int z, double val) {
        array[t*size*size*size + x*size*size + y*size + z] = val;
    }

    double * get_layer(int t) {
        return array + t*size*size*size;
    }

    double laplace(int n, int i, int j, int k, int x1, int y1, int z1, int x2, int y2, int z2) {
        double ans = 0.0;
        // array[n*size*size*size + i*size*size + j*size + k]
        ans += (array[n*size*size*size + x1*size*size + j*size + k] - 2*array[n*size*size*size + i*size*size + j*size + k] + array[n*size*size*size + x2*size*size + j*size + k])/(hx*hx);
        ans += (array[n*size*size*size + i*size*size + y1*size + k] - 2*array[n*size*size*size + i*size*size + j*size + k] + array[n*size*size*size + i*size*size + y2*size + k])/(hy*hy);
        ans += (array[n*size*size*size + i*size*size + j*size + z1] - 2*array[n*size*size*size + i*size*size + j*size + k] + array[n*size*size*size + i*size*size + j*size + z2])/(hz*hz);
        return ans;
    }

    ~Grid() {
        delete[] array;
    }
};

Grid prec_sol; // sol[t][x][y][z]
Grid num_sol; // sol[t][x][y][z]
Grid abs_error;
Grid rel_error;
vector<double> max_abs_errors;
vector<double> max_rel_errors;

ofstream out;

double an_sol(double t, double x, double y, double z) {
    return sin(2*PI*x/Lx)*sin(PI*y/Ly)*sin(PI*z/Lz)*cos(a_t*t + 2*PI);
}



void calculate_an_sol() {
    auto start = chrono::high_resolution_clock::now();
    for(int n = 0; n <= timesteps; n++) {
        for(int i = 0; i <= N; i++) {
            for(int j = 0; j <= N; j++) {
                for(int k = 0; k <= N; k++) {
                    double f = an_sol(tau*n, hx*i, hy*j, hz*k);
                    prec_sol.set(n, i, j, k, f);

                }
            }
        }
    }
    auto duration = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - start).count();
    out << "analytical solution calculated in " << duration << "ms" << endl;
}

void prepare_layer(int n) { // граничные значения
    // y, z
    #pragma omp parallel for num_threads(Np) shared(num_sol)
    for(int i = 0; i <= N; i++) {
        for(int h = 0; h <= N; h++) {
            num_sol.set(n, i, h, 0, 0.0);
            num_sol.set(n, i, h, N, 0.0);
            num_sol.set(n, i, 0, h, 0.0);
            num_sol.set(n, i, N, h, 0.0);
        }
    }
    #pragma omp parallel for num_threads(Np) shared(num_sol)
    for(int j = 1; j <= N-1; j++) {
        for(int k = 1; k <= N-1; k++) {

            num_sol.set(n, N, j, k, (n > 1 ? 2 : 1)*num_sol.get(n-1, N, j, k) - (n > 1 ? num_sol.get(n-2, N, j, k) : 0) + (n > 1 ? 1.0 : 0.5)*a2*tau*tau*num_sol.laplace(n-1, N, j, k, N-1, j-1, k-1, 1, j+1, k+1));
            num_sol.set(n, 0, j, k, num_sol.get(n, N, j, k));
        }
    }
}

void calculate_start() { // начальные значения
    // n = 0
    cout << "calculating layer " << 0 << endl;
    #pragma omp parallel for num_threads(Np) shared(num_sol)
    for(int i = 0; i <= N; i++) {
        for(int j = 0; j <= N; j++) {
            for(int k = 0; k <= N; k++) {
                num_sol.set(0, i, j, k, an_sol(0, i*hx, j*hy, k*hz));
            }
        }
    }
    // n = 1
    cout << "calculating layer " << 1 << endl;
    prepare_layer(1);
    #pragma omp parallel for num_threads(Np) shared(num_sol)
    for(int i = 1; i <= N-1; i++) {
        for(int j = 1; j <= N-1; j++) {
            for(int k = 1; k <= N-1; k++) {
                num_sol.set(1, i, j, k, num_sol.get(0, i, j, k) + a2*tau*tau*0.5*num_sol.laplace(0, i, j, k, i-1, j-1, k-1, i+1, j+1, k+1));
            }
        }
    }
}




void calculate_num_sol() {
    double start = omp_get_wtime();
    calculate_start();
    for(int n = 2; n <= timesteps; n++) {
        cout << "calculating layer " << n << endl;
        prepare_layer(n);
        #pragma omp parallel for num_threads(Np) shared(num_sol, n)
        for(int i = 1; i <= N-1; i++) {
            for(int j = 1; j <= N-1; j++) {
                for(int k = 1; k <= N-1; k++) {
                    num_sol.set(n, i, j, k, 2*num_sol.get(n-1, i, j, k) - num_sol.get(n-2, i, j, k) + a2*tau*tau*num_sol.laplace(n-1, i, j, k, i-1, j-1, k-1, i+1, j+1, k+1));
                }
            }
        }
    }
    double duration = omp_get_wtime() - start;
    out << "numerical solution calculated in " << (unsigned) (duration*1000) << "ms" << endl;
}

void calculate_error() {
    max_abs_errors.clear();
    max_rel_errors.clear();
    for(int n = 0; n <= timesteps; n++) {
        double max_rel = 0.0, max_abs = 0.0;
        for(int i = 1; i <= N-1; i++) {
            for(int j = 1; j <= N-1; j++) {
                for(int k = 1; k <= N-1; k++) {
                    double a = fabs(prec_sol.get(n, i, j, k) - num_sol.get(n, i, j, k));
                    double r = fabs(prec_sol.get(n, i, j, k) - num_sol.get(n, i, j, k)) / fabs(prec_sol.get(n, i, j, k));
                    abs_error.set(n, i, j, k, a);
                    rel_error.set(n, i, j, k, r);
                    max_rel = fmax(max_rel, r);
                    max_abs = fmax(max_abs, a);
                }
            }
        }
        max_abs_errors.push_back(max_abs);
        max_rel_errors.push_back(max_rel);
        out << "max abs and rel errors on layer " << n << ": " << max_abs << " " << max_rel << endl;
    }
}

int main(int argc, char *argv[]) { // command line args: N, Np, Lx, Ly, Lz, T, timesteps
    N = stoi(argv[1]);
    Np = stoi(argv[2]);
    if(strcmp(argv[3], "pi") == 0) Lx = PI;
    else Lx = stod(argv[3]);
    if(strcmp(argv[4], "pi") == 0) Ly = PI;
    else Ly = stod(argv[4]);
    if(strcmp(argv[5], "pi") == 0) Lz = PI;
    else Lz = stod(argv[5]);
    if(argc >= 7) T = stod(argv[6]);
    else T = 1;
    if(argc >= 8) timesteps = stoi(argv[7]);
    else timesteps = 20;


    a2 = 1/(4*PI*PI);
    a_t = 0.5 * sqrt(4/(Lx*Lx) + 1/(Ly*Ly) + 1/(Lz*Lz));
    
    tau = T / timesteps;
    hx = Lx / N; hy = Ly / N; hz = Lz / N;

    cout << "a_t = " << a_t << endl;
    cout << "C = " << sqrt(a2)*tau/min(hx, min(hy, hz)) << endl;

    prec_sol.init_grid(timesteps+2, N);
    num_sol.init_grid(timesteps+2, N);
    abs_error.init_grid(timesteps+2, N);
    rel_error.init_grid(timesteps+2, N);
    cout << "grids initialized\n";

    //Np = 1;
    calculate_an_sol();
    cout << "an sol calculated" << endl;

    for(int i = 0; i <= 0; i++) {

        out.clear();
        out.open(string("output_N") + to_string(N) + "_Np" + to_string(Np) + string(".txt"));

        omp_set_num_threads(Np);

        
        calculate_num_sol();
        calculate_error();


        out.close();
        Np *= 2;
    }


    

    return 0;
}