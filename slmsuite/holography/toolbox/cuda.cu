// This file is loaded into slmsuite.holography.toolbox.phase at runtime.
// cupy.RawKernels call functions inside it.
#include <cupy/complex.cuh>
#include <math.h>

extern "C"  __device__ void warp_reduce(
    complex<float>* sdata, // volatile
    unsigned int tid
) {
    sdata[tid] += sdata[tid + 32]; __syncwarp();
    sdata[tid] += sdata[tid + 16]; __syncwarp();
    sdata[tid] += sdata[tid + 8];  __syncwarp();
    sdata[tid] += sdata[tid + 4];  __syncwarp();
    sdata[tid] += sdata[tid + 2];  __syncwarp();
    sdata[tid] += sdata[tid + 1];  __syncwarp();
}

// Version 1 compressed kernel

extern "C" __global__ void compressed_farfield2nearfield(
    const complex<float>* farfield, // Input (N)
    const unsigned int W,           // Width of nearfield
    const unsigned int H,           // Height of nearfield
    const unsigned int N,           // Size of farfield
    const unsigned int D,           // Dimension of spots (2 or 3)
    const float* kxyz,              // Spot parameters (D*N)
    const float cx,                 // Grid offset x
    const float cy,                 // Grid offset y
    const float dx,                 // X grid pitch
    const float dy,                 // Y grid pitch
    complex<float>* nearfield       // Output (W*H)
) {
    // nf is each pixel in the nearfield.
    int nf = blockDim.x * blockIdx.x + threadIdx.x;

    if (nf < W * H) {
        // Make a local result variable to avoid talking with global memory.
        float exponent = 0;
        complex<float> result = 0;

        // Figure out which pixel we are at.
        float x = dx * (nf % W) - cx;
        float y = dy * (nf / W) - cy;

        if (D == 3) {
            // Additional copy for 3D spots.
            float r = x * x + y * y;

            // Loop over all the spots (compiler should handle optimizing the trinary).
            for (int i = 0; i < N; i++) {
                exponent = x * kxyz[i] + y * kxyz[i + N] + r * kxyz[i + N + N];
                result += farfield[i] * exp(1j * exponent);
            }
        } else {
            // Loop over all the spots (compiler should handle optimizing the trinary).
            for (int i = 0; i < N; i++) {
                exponent = x * kxyz[i] + y * kxyz[i + N];
                result += farfield[i] * exp(1j * exponent);
            }
        }

        // Export the result to global memory.
        nearfield[nf] = result;
    }
}

extern "C" __global__ void compressed_nearfield2farfield(
    const complex<float>* nearfield,        // Input (W*H)
    const unsigned int W,                   // Width of nearfield
    const unsigned int H,                   // Height of nearfield
    const unsigned int N,                   // Size of farfield
    const unsigned int D,                   // Dimension of spots (2 or 3)
    const float* kxyz,                      // Spot parameters (D*N)
    const float cx,                         // Grid offset x
    const float cy,                         // Grid offset y
    const float dx,                         // X grid pitch
    const float dy,                         // Y grid pitch
    complex<float>* farfield_intermediate   // Output (blockIdx.x*N)
) {
    // Allocate shared data which will store intermediate results.
    // (Hardcoded to 1024 block size).
    __shared__ complex<float> sdata[1024];

    // Make some IDs.
    int ff = blockIdx.y;                    // Farfield index  [0, N)
    int tid = threadIdx.x;                  // Thread ID
    int rid = blockIdx.x + ff * gridDim.x;  // Farfield result ID
    int nf = blockDim.x * blockIdx.x + tid; // Nearfield index [0, WH)

    float x = dx * (nf % W) - cx;
    float y = dy * (nf / W) - cy;

    float exponent = 0;

    if (nf < W * H) {
        exponent = x * kxyz[ff] + y * kxyz[ff + N];

        if (D == 3) {
            exponent += (x * x + y * y) * kxyz[ff + N + N];
        }

        // Do the overlap integrand for one nearfield-farfield mapping.
        sdata[tid] = conj(nearfield[nf]) * exp(1j * exponent);
    } else {
        sdata[tid] = 0;
    }

    // Now we want to integrate by summing these results.
    // Note that we assume 1024 block size and 32 warp size (change this?).
    __syncthreads();

    if (tid < 512) { sdata[tid] += sdata[tid + 512]; } __syncthreads();
    if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads();
    if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads();
    if (tid < 64) {  sdata[tid] += sdata[tid + 64];  } __syncthreads();
    if (tid < 32) {
        // The last 32 threads don't require __syncthreads() as they can be warped.
        warp_reduce(sdata, tid);

        // Save the summed results to global memory.
        if (tid == 0) { farfield_intermediate[rid] = sdata[0]; }
    }
}

// Version 2 compressed kernel

extern "C" __device__ void populate_basis(
    const float x,
    const float y,
    const unsigned int D,           // Dimension of Zernike basis (2 [xy] for normal spot arrays)
    const unsigned int M,           // Dimension of polynomial basis (2 [xy] for normal spot arrays)
    const float* c_md,              // Polynomial coefficients for Zernike (size M*D) [shared]
    const int* i_md,                // A key for where c_dm is nonzero (size M*D) [shared]
    const int* pxy_m,               // Monomial coefficients (size 2*M) [shared]
    float* basis,
) {
    int i, j, d, stride = 0;
    int nx, ny, nx0, ny0 = 0;
    float monomial = 0;

    for (i = 0; i < M; i++) {
        nx = pxy_m[i];
        ny = pxy_m[i+M];

        if (nx < 0) {   // Vortex phase plate case
            monomial = -nx * atan2(x, y);
            nx0 = ny0 = M;
        } else {        // Monomial case
            // Reset if we're starting a new path.
            if (nx - nx0 < 0 || ny - ny0 < 0) {
                nx0 = ny0 = 0;
                monomial = 1;
            }

            // Traverse the path in +x or +y.
            for (j = 0; j < nx - nx0; j++) {
                monomial *= x;
            }
            for (j = 0; j < ny - ny0; j++) {
                monomial *= y;
            }

            // Update the state of the monomial
            nx0 = nx;
            ny0 = ny;
        }

        // Now we need to add this monomial to all relevant basis states
        j = 0;
        stride = i*D;       // Only mulitply once and find the starting point.
        d = i_dm[stride];   //
        while (d >= 0) {
            // Add the monomial to the result.
            basis[d] += c_md[stride + d] * monomial;

            // Determine if there's another basis state which needs this monomial
            j++;
            d = i_md[stride + j];
        }
    }
}

extern "C" __global__ void compressed_farfield2nearfield_v2(
    const complex<float>* farfield, // Input (size N)
    const unsigned int W,           // Width of nearfield
    const unsigned int H,           // Height of nearfield
    const unsigned int N,           // Size of farfield
    const unsigned int D,           // Dimension of Zernike basis (2 [xy] for normal spot arrays)
    const unsigned int M,           // Dimension of polynomial basis (2 [xy] for normal spot arrays)
    const float* a_nd,              // Spot Zernike coefficients (size N*D) [shared]
    const float* c_md,              // Polynomial coefficients for Zernike (size M*D) [shared]
    const int* i_md,                // A key for where c_dm is nonzero (size M*D) [shared]
    const int* pxy_m,               // Monomial coefficients (size 2*M) [shared]
    const float cx,
    const float cy,
    const float dx,                 // X grid pitch
    const float dy,                 // Y grid pitch
    complex<float>* nearfield       // Output (size W*H)
) {
    // nf is the index of the pixel in the nearfield.
    int nf = blockDim.x * blockIdx.x + threadIdx.x;

    if (nf < W*H) {
        // Prepare local basis.
        // This is the phase for a given coefficient d at the current pixel.
        // It incurs an O(D) cost, but is amortized over O(N) points.
        float* basis = float[D];

        populate_basis(
            dx * (nf % W) - cx,
            dy * (nf / W) - cy,
            D, M, c_md, i_md, pxy_m,
            basis
        );

        // Local variables.
        complex<float> result = 0;
        float exponent = 0;
        int i = 0;
        int d = 0;

        // Loop over all the spots.
        for (i = 0; i < N; i++) {
            exponent = 0;
            // Loop over basis indices.
            for (d = 0; d < D; d++) {
                exponent += basis[d] * a_dn[d*N + i];
            }
            result += farfield[i] * exp(1j * exponent);
        }

        // Export the result to global memory.
        nearfield[nf] = result;
    }
}

extern "C" __global__ void compressed_nearfield2farfield_v2(
    const complex<float>* nearfield,        // Input (W * H)
    const unsigned int W,                   // Width of nearfield
    const unsigned int H,                   // Height of nearfield
    const unsigned int N,                   // Size of farfield
    const unsigned int D,                   // Dimension of Zernike basis (2 [xy] for normal spot arrays)
    const unsigned int M,                   // Dimension of polynomial basis (2 [xy] for normal spot arrays)
    const float* a_nd,                      // Spot Zernike coefficients (size N*D) [shared]
    const float* c_md,                      // Polynomial coefficients for Zernike (size M*D) [shared]
    const int* i_md,                        // A key for where c_dm is nonzero (size M*D) [shared]
    const int* pxy_m,                       // Monomial coefficients (size 2*M) [shared]
    const float cx,                         // Grid offset x
    const float cy,                         // Grid offset y
    const float dx,                         // X grid pitch
    const float dy,                         // Y grid pitch
    complex<float>* farfield_intermediate   // Output (blockIdx.x*N)
) {
    // Allocate shared data which will store intermediate results.
    // (Hardcoded to 1024 block size).
    __shared__ complex<float> sdata[1024];

    // Make some IDs.
    int tid = threadIdx.x;                  // Thread ID
    int nf = blockDim.x * blockIdx.x + tid; // Nearfield index [0, W*H)
    bool inrange = nf < W*H;

    // Local variables.
    complex<float> result = 0;
    float exponent = 0;
    int i = 0;
    int d = 0;

    // Prepare local basis.
    // This is the phase for a given coefficient d at the current pixel.
    // It incurs an O(D) cost, but is amortized over O(N) points.
    float* basis = float[D];

    if (inrange) {
        populate_basis(
            dx * (nf % W) - cx,
            dy * (nf / W) - cy,
            D, M, c_md, i_md, pxy_m,
            basis
        );
    }

    // Iterate through farfield points. Use our basis to find the results.
    for (i = 0; i < N; i++) {
        if (inrange) {
            // Loop over basis indices.
            for (d = 0; d < D; d++) {
                exponent += basis[d] * a_dn[d*N + ff];
            }

            // Do the overlap integrand for one nearfield-farfield mapping.
            sdata[tid] = conj(nearfield[nf]) * exp(1j * exponent);
        } else {
            sdata[tid] = 0;
        }

        // Now we want to integrate by summing these results.
        // Note that we assume 1024 block size and 32 warp size (change this?).
        __syncthreads();

        if (tid < 512) { sdata[tid] += sdata[tid + 512]; } __syncthreads();
        if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads();
        if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads();
        if (tid < 64) {  sdata[tid] += sdata[tid + 64];  } __syncthreads();
        if (tid < 32) {
            // The last 32 threads don't require __syncthreads() as they can be warped.
            warp_reduce(sdata, tid);

            // Save the summed results to global memory.
            if (tid == 0) { farfield_intermediate[blockIdx.x + i * gridDim.x] = sdata[0]; }
        }
    }
}

// Polynomial sum kernel

extern "C" __global__ void polynomial_sum(
    const unsigned int N,           // Number of coefficients
    const int* pathing,             // Path order (1*N)
    const float* coefficients,      // Spot parameters (1*N)
    const float* px,                // Spot parameters (1*N)
    const float* py,                // Spot parameters (1*N)
    const unsigned int WH,          // Size of nearfield
    const float* X,                 // X grid (WH)
    const float* Y,                 // Y grid (WH)
    float* out                      // Output (WH)
) {
    // g is each pixel in the grid.
    int g = blockDim.x * blockIdx.x + threadIdx.x;

    if (g < WH) {
        // Make a local result variable to avoid talking with global memory.
        float result = 0;

        // Copy data that will be used multiple times per thread into local memory (this might not matter though).
        float local_X = X[g];
        float local_Y = Y[g];
        float coefficient;

        // Local helper variables.
        float monomial = 1;
        int nx, ny, nx0, ny0 = 0;
        int j, k = 0

        // Loop over all the spots (compiler should handle optimizing the trinary).
        for (int i = 0; i < N; i++) {
            k = pathing[i];
            coefficient = coefficients[k];

            if (coefficient != 0) {
                nx = px[k];
                ny = py[k];

                // Reset if we're starting a new path.
                if (nx - nx0 < 0 || ny - ny0 < 0) {
                    nx0 = ny0 = 0;
                    monomial = 1;
                }

                // Traverse the path in +x or +y.
                for (j = 0; j < nx - nx0; j++) {
                    monomial *= local_X;
                }
                for (j = 0; j < ny - ny0; j++) {
                    monomial *= local_Y;
                }

                // Add the monomial to the result.
                result += coefficients[k] * monomial;

                // Update the state of the monomial
                nx0 = nx;
                ny0 = ny;
            }
        }

        // Export the result to global memory.
        out[g] = result;
    }
}

// Weighting
extern "C" __global__ void update_weights_generic(
    float* weight_amp,                  // Input (N)
    const float* feedback_amp,          // Measured amplitudes (N)
    const float* target_amp,            // Desired amplitudes (N)
    const unsigned int N,               // Size
    const unsigned int method,          // Indexed WGS method
    const float feedback_norm,          // cupy-computed norm of feedback_amp
    const float feedback_exponent,      // Method-specific
    const float feedback_factor,        // Method-specific

) {
    // i is each pixel in the weights.
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < N) {
        float feedback = feedback_amp[i];
        float target = 0;

        if (feedback_norm != 0) {
            feedback /= feedback_norm;
            target = target_amp[i]
        } else {
            target = 1 / sqrt(N);
        }

        if (method != 4 && method != 5) {
            // Handle feedback with a non-uniform target
            if (target != 0) {
                feedback /= target;
            } else {
                feedback = 1;           // Do nothing.
            }

            if (method == 1 || method == 2) {   // Leonardo, Kim
                feedback = pow(-feedback, feedback_exponent)
            } else if (method == 3) {           // Nogrette
                feedback = 1 / (1 - feedback_factor * (1 - feedback))
            }
        } else {
            if (method == 4) {                  // Wu
                feedback = exp(feedback_exponent * (target - feedback))
            } else if (method == 5) {           // tanh
                feedback = feedback_factor * tanh(feedback_exponent * (target - feedback))
            }
        }

        // Check nan, inf
        if (isinf(feedback) || isnan(feedback)) { feedback = 1 }

        // Export the result to global memory.
        if (feedback != 1) {
            weight_amp[i] *= feedback;
        }
    }
}