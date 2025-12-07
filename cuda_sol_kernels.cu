#include <iostream>

__device__ __constant__ double PI = 3.14159265358979323846;

__device__ __constant__ double a2, tau, hx, hy, hz, a_t, Lx, Ly, Lz;
__device__ __constant__ int x_0, y_0, z_0;

void init_device_constants(double a2_, double a_t_, double tau_, double hx_, double hy_, double hz_, double Lx_, double Ly_, double Lz_,  int x0, int y0, int z0, cudaStream_t stream) {
    cudaMemcpyToSymbol(a2, &a2_, sizeof(double));
    cudaMemcpyToSymbol(a_t, &a_t_, sizeof(double));
    cudaMemcpyToSymbol(tau, &tau_, sizeof(double));
    cudaMemcpyToSymbol(hx, &hx_, sizeof(double));
    cudaMemcpyToSymbol(hy, &hy_, sizeof(double));
    cudaMemcpyToSymbol(hz, &hz_, sizeof(double));
    cudaMemcpyToSymbol(x_0, &x0, sizeof(int));
    cudaMemcpyToSymbol(y_0, &y0, sizeof(int));
    cudaMemcpyToSymbol(z_0, &z0, sizeof(int));
    cudaMemcpyToSymbol(Lx, &Lx_, sizeof(double));
    cudaMemcpyToSymbol(Ly, &Ly_, sizeof(double));
    cudaMemcpyToSymbol(Lz, &Lz_, sizeof(double));
    //std::cout << "Device constants initialized\n";
}

__global__ void calculate_layer(double * grid0, double * grid1, double * grid2, int x, int y, int z, int n,
                                int i1, int j1, int k1, int i2, int j2, int k2, double * abs_errs, double * rel_errs) {
    // x, y, z - dimensions of the grid
    // i1, ..., i2, ... - start and end indices
    int i, j, k;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    k = idx % z;
    j = (idx / z) % y;
    i = idx / (y * z);
    if(i >= i1 && i <= i2 && j >= j1 && j <= j2 && k >= k1 && k <= k2) {

        double u = 2*grid1[i*y*z + j*z + k] - grid0[i*y*z + j*z + k] + a2 * tau * tau * 
                    (
                        (grid1[(i+1)*y*z + j*z + k] + grid1[(i-1)*y*z + j*z + k] - 2*grid1[i*y*z + j*z + k])/(hx*hx) +
                        (grid1[i*y*z + (j+1)*z + k] + grid1[i*y*z + (j-1)*z + k] - 2*grid1[i*y*z + j*z + k])/(hy*hy) +
                        (grid1[i*y*z + j*z + (k+1)] + grid1[i*y*z + j*z + (k-1)] - 2*grid1[i*y*z + j*z + k])/(hz*hz)
                    );
        grid2[i*y*z + j*z + k] = u;
        double f = cos(a_t*n*tau + 2*PI) * sin(2*PI*hx*(i-1+x_0)/Lx) * sin(PI*hy*(j-1+y_0)/Ly) * sin(PI*hz*(k-1+z_0)/Lz);
        double abs_error = fabs(u - f);
        double rel_error = abs_error / fabs(f);
        abs_errs[i*y*z + j*z + k] = abs_error;
        rel_errs[i*y*z + j*z + k] = rel_error;
    }
}


__global__ void calculate_max_errors(double * abs_errors, double * rel_errors, int x, int y, int z,
                                    double * block_abs_errors, double * block_rel_errors)
{
    // calculates max errors in the block ([blockIdx.x * blockDim.x, (blockIdx.x + 1) * blockDim.x)))
    int total_size = x*y*z; // 343
    int size = min(blockDim.x, total_size - blockIdx.x * blockDim.x); // 343
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= total_size) return;
    for(int stride = 1; stride < size; stride *= 2) {
        if(threadIdx.x % (2*stride) == 0 && idx + stride < total_size) {
            abs_errors[idx] = fmax(abs_errors[idx], abs_errors[idx + stride]);
            rel_errors[idx] = fmax(rel_errors[idx], rel_errors[idx + stride]);
        }
        __syncthreads();
    }
    if(threadIdx.x == 0) {
        block_abs_errors[blockIdx.x] = abs_errors[idx];
        block_rel_errors[blockIdx.x] = rel_errors[idx];
    }
}

void calculate_errors(double * abs_errs, double * rel_errs, int x, int y, int z, double * result, cudaStream_t stream) {
    double * blocks_max_abs;
    double * blocks_max_rel;
    double * abs_res;
    double * rel_res;
    int block_size = 512;
    int size = x*y*z;
    int blocks = (size + block_size - 1) / block_size;
    dim3 gridDim(blocks);
    dim3 blockDim(block_size);
    //std::cout << blocks << " " << block_size << " " << size << "\n";
    cudaMalloc(&blocks_max_abs, blocks*sizeof(double));
    cudaMalloc(&blocks_max_rel, blocks*sizeof(double));
    cudaMemset(blocks_max_abs, 0.0, blocks*sizeof(double));
    cudaMemset(blocks_max_rel, 0.0, blocks*sizeof(double));
    cudaMallocHost(&abs_res, blocks*sizeof(double));
    cudaMallocHost(&rel_res, blocks*sizeof(double));
    calculate_max_errors<<<gridDim, blockDim, 0, stream>>>(abs_errs, rel_errs, x, y, z, blocks_max_abs, blocks_max_rel);
    cudaStreamSynchronize(stream);
    cudaMemcpy(abs_res, blocks_max_abs, blocks*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(rel_res, blocks_max_rel, blocks*sizeof(double), cudaMemcpyDeviceToHost);
    double abs_err = 0.0, rel_err = 0.0;
    for(int i = 0; i < blocks; i++) {
        if(abs_res[i] > abs_err) abs_err = abs_res[i];
        if(rel_res[i] > rel_err) rel_err = rel_res[i];
    }
    result[0] = abs_err; result[1] = rel_err;
    cudaFreeHost(abs_res); cudaFreeHost(rel_res);
    cudaFree(blocks_max_abs); cudaFree(blocks_max_rel);
}

__global__ void copy_to_matrices(double * grid, double * matrix1, double * matrix2, int x, int y, int z,
                                    int dim, int start, int end) {
    // x, y, z = X+2, Y+2, Z+2
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(dim == 1) {
        int i1 = start, i2 = end;
        int j = idx / z;
        int k = idx % z;
        if(j >= 1 && k >= 1 && j <= y-2 && k <= z-2) {
            matrix1[j*z + k] = grid[i1*y*z + j*z + k];
            matrix2[j*z + k] = grid[i2*y*z + j*z + k];
        }
    }
    if(dim == 2) {
        int j1 = start, j2 = end;
        int i = idx / z;
        int k = idx % z;
        if(i >= 1 && k >= 1 && i <= x-2 && k <= z-2) {
            matrix1[i*z + k] = grid[i*y*z + j1*z + k];
            matrix2[i*z + k] = grid[i*y*z + j2*z + k];
        }
    }
    if (dim == 3) {
        int k1 = start, k2 = end;
        int i = idx / y;
        int j = idx % y;
        if(i >= 1 && j >= 1 && i <= x-2 && j <= y-2) {
            matrix1[i*y + j] = grid[i*y*z + j*z + k1];
            matrix2[i*y + j] = grid[i*y*z + j*z + k2];
        }
    }
}

void call_copy_to_matrices(double * grid, double * matrix1, double * matrix2, int x, int y, int z,
                            int dim, int start, int end, cudaStream_t stream) {
    int block_size = 512;
    int size;
    if(dim == 1) size = y*z;
    else if(dim == 2) size = x*z;
    else size = x*y;
    int grid_size = (size + block_size - 1) / block_size;
    dim3 gridDim(grid_size);
    dim3 blockDim(block_size);
    copy_to_matrices<<<gridDim, blockDim, 0, stream>>>(grid, matrix1, matrix2, x, y, z, dim, start, end);
}

__global__ void copy_to_grid(double * grid, double * matrix, int x, int y, int z,
                            int dim, int index) {
    // x, y, z = X+2, Y+2, Z+2
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i, j, k, m;
    if(dim == 1) {
        i = index;
        j = idx / z;
        k = idx % z;
        m = j*z + k;
        if(j >= 1 && k >= 1 && j <= y-2 && k <= z-2) grid[i*y*z + j*z + k] = matrix[m];
    } else if (dim == 2) {
        j = index;
        i = idx / z;
        k = idx % z;
        m = i*z + k;
        if(i >= 1 && k >= 1 && i <= x-2 && k <= z-2) grid[i*y*z + j*z + k] = matrix[m];
    } else {
        k = index;
        i = idx / y;
        j = idx % y;
        m = i*y + j;
        if(i >= 1 && j >= 1 && i <= x-2 && j <= y-2) grid[i*y*z + j*z + k] = matrix[m];
    }
}

void call_copy_to_grid(double * grid, double * matrix, int x, int y, int z,
                        int dim, int index, cudaStream_t stream) {
    int block_size = 512;
    int size;
    if(dim == 1) size = y*z;
    else if(dim == 2) size = x*z;
    else size = x*y;
    int grid_size = (size + block_size - 1) / block_size;
    dim3 gridDim(grid_size);
    dim3 blockDim(block_size);
    copy_to_grid<<<gridDim, blockDim, 0, stream>>>(grid, matrix, x, y, z, dim, index);
}

__global__ void layer0(double * grid0, int x, int y, int z, double * abs_errs, double * rel_errs) {
    int i, j, k;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    k = idx % z;
    j = (idx / z) % y;
    i = idx / (y * z);
    if(i >= 1 && i <= x-2 && j >= 1 && j <= y-2 && k >= 1 && k <= z-2) {
        double u = cos(0.0 + 2*PI) * sin(2*PI*hx*(i-1+x_0)/Lx) * sin(PI*hy*(j-1+y_0)/Ly) * sin(PI*hz*(k-1+z_0)/Lz);
        grid0[i*y*z + j*z + k] = u;
        abs_errs[i*y*z + j*z + k] = 0.0;
        rel_errs[i*y*z + j*z + k] = 0.0;
    }
}

__global__ void layer1(double * grid0, double * grid1, int x, int y, int z,
                        int i1, int j1, int k1, int i2, int j2, int k2,
                        double * abs_errs, double * rel_errs)
{
    int i, j, k;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    k = idx % z;
    j = (idx / z) % y;
    i = idx / (y * z);
    if(i >= i1 && i <= i2 && j >= j1 && j <= j2 && k >= k1 && k <= k2) {
        double u = grid0[i*y*z + j*z + k] + a2*tau*tau*0.5*(
            (grid0[(i+1)*y*z + j*z + k] + grid0[(i-1)*y*z + j*z + k] - 2*grid0[i*y*z + j*z + k])/(hx*hx) +
            (grid0[i*y*z + (j+1)*z + k] + grid0[i*y*z + (j-1)*z + k] - 2*grid0[i*y*z + j*z + k])/(hy*hy) +
            (grid0[i*y*z + j*z + (k+1)] + grid0[i*y*z + j*z + (k-1)] - 2*grid0[i*y*z + j*z + k])/(hz*hz)
        );
        grid1[i*y*z + j*z + k] = u;
        double f = cos(a_t*1*tau + 2*PI) * sin(2*PI*hx*(i-1+x_0)/Lx) * sin(PI*hy*(j-1+y_0)/Ly) * sin(PI*hz*(k-1+z_0)/Lz);
        double abs_error = fabs(u - f);
        double rel_error = abs_error / fabs(f);
        abs_errs[i*y*z + j*z + k] = abs_error;
        rel_errs[i*y*z + j*z + k] = rel_error;
    }
}

void call_calculate_layer(double * grid0, double * grid1, double * grid2, int x, int y, int z, int n,
                        int i1, int j1, int k1, int i2, int j2, int k2,
                        double * abs_errs, double * rel_errs, cudaStream_t stream)
{
    int block_size = 512;
    int size = x*y*z;
    int grid_size = (size + block_size - 1) / block_size;
    dim3 gridDim(grid_size);
    dim3 blockDim(block_size);
    if(n == 0) {
        layer0<<<gridDim, blockDim, 0, stream>>>(grid0, x, y, z, abs_errs, rel_errs);
    } else if(n == 1) {
        layer1<<<gridDim, blockDim, 0, stream>>>(grid0, grid1, x, y, z, i1, j1, k1, i2, j2, k2, abs_errs, rel_errs);
    } else {
        calculate_layer<<<gridDim, blockDim, 0, stream>>>(grid0, grid1, grid2, x, y, z, n, i1, j1, k1, i2, j2, k2, abs_errs, rel_errs);
    }
}

__global__ void prepare_layer(double * grid0, double * grid1, double * grid2, int n, int x, int y, int z,
                                int j1, int k1, int j2, int k2, unsigned int b)
{
    int j, k;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int length = max(x, max(y, z));
    k = idx % length;
    j = (idx / length);
    if(b & 1 != 0 && j >= 1 && j <= x-2 && k >= 1 && k <= y-2) grid2[j*y*z + k*z + 1] = 0.0;
    if(b & 2 != 0 && j >= 1 && j <= x-2 && k >= 1 && k <= y-2) grid2[j*y*z + k*z + z-2] = 0.0;
    if(b & 4 != 0 && j >= 1 && j <= x-2 && k >= 1 && k <= z-2) grid2[j*y*z + 1*z + k] = 0.0;
    if(b & 8 != 0 && j >= 1 && j <= x-2 && k >= 1 && k <= z-2) grid2[j*y*z + (y-2)*z + k] = 0.0;
    if(b & 16 != 0 && j >= j1 && j <= j2 && k >= k1 && k <= k2) {
        // grid2[(x-2)*y*z + j*z + k]
        double u = (n > 1 ? 2 : 1)*grid1[(x-2)*y*z + j*z + k] - (n > 1 ? grid0[(x-2)*y*z + j*z + k] : 0) + 
                (n > 1 ? 1.0 : 0.5)*a2*tau*tau*(
                    (grid1[(x-1)*y*z + j*z + k] + grid1[(x-3)*y*z + j*z + k] - 2*grid1[(x-2)*y*z + j*z + k])/(hx*hx) +
                    (grid1[(x-2)*y*z + (j+1)*z + k] + grid1[(x-2)*y*z + (j-1)*z + k] - 2*grid1[(x-2)*y*z + j*z + k])/(hy*hy) +
                    (grid1[(x-2)*y*z + j*z + (k+1)] + grid1[(x-2)*y*z + j*z + (k-1)] - 2*grid1[(x-2)*y*z + j*z + k])/(hz*hz)
                );
        grid2[(x-2)*y*z + j*z + k] = u;
    }
    if(b & 32 != 0 && j >= j1 && j <= j2 && k >= k1 && k <= k2) {
        double u = (n > 1 ? 2 : 1)*grid1[1*y*z + j*z + k] - (n > 1 ? grid0[1*y*z + j*z + k] : 0) + 
                (n > 1 ? 1.0 : 0.5)*a2*tau*tau*(
                    (grid1[2*y*z + j*z + k] + grid1[0*y*z + j*z + k] - 2*grid1[1*y*z + j*z + k])/(hx*hx) +
                    (grid1[1*y*z + (j+1)*z + k] + grid1[1*y*z + (j-1)*z + k] - 2*grid1[1*y*z + j*z + k])/(hy*hy) +
                    (grid1[1*y*z + j*z + (k+1)] + grid1[1*y*z + j*z + (k-1)] - 2*grid1[1*y*z + j*z + k])/(hz*hz)
                );
        grid2[1*y*z + j*z + k] = u;
    }
}

void call_prepare_layer(double * grid0, double * grid1, double * grid2, int n, int x, int y, int z,
                                int j1, int k1, int j2, int k2, unsigned int b, cudaStream_t stream)
{
    int block_size = 512;
    int length = max(x, max(y, z));
    int size = length*length;
    int grid_size = (size + block_size - 1) / block_size;
    dim3 gridDim(grid_size);
    dim3 blockDim(block_size);
    prepare_layer<<<gridDim, blockDim, 0, stream>>>(grid0, grid1, grid2, n, x, y, z, j1, k1, j2, k2, b);
}