#include <stdio.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <math.h> 

#include "cuda.h"

const int NUM_ELEMENTS = 1 << 22; // Rozmiar instancji, potêga dwójki nie wiêksza ni¿ 2**30
const int THREADS_PER_BLOCK = 128; // Rozmiar bloku
const int NUM_OPERATIONS = 1; // Liczba operacji sumowañ: 1, 3, 7

// Kernel
__global__ void vectorAdd(const float* A, float* C) {
	__shared__ float sdata[THREADS_PER_BLOCK];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
	sdata[tid] = A[i] + A[i + blockDim.x];
	__syncthreads();
	// do reduction in shared mem
	for(unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid < s) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}
	// write result for this block to global mem
	if(tid == 0) C[blockIdx.x] = sdata[0];
	__syncthreads();
}

int calculateThreads(int iteration) {
	int result = NUM_ELEMENTS;
	for (int i = 1; i < iteration; i++) {
		result /= (THREADS_PER_BLOCK * 2);
	}
	printf("DUPA vector with %d elements\n", result);
	return result/2;
}


int main(void) {
	srand(time(NULL));
	cudaError_t err = cudaSuccess;

	size_t size = NUM_ELEMENTS * sizeof(float);
	printf("Summing vector with %d elements\n", NUM_ELEMENTS);

	// Allocate the host input vector A and C
	float* h_A = (float*)malloc(size);
	float* h_C = (float*)malloc(size); // todo: zmieniæ na blocks_per_grid ?

	// Verify that allocations succeeded
	if (h_A == NULL || h_C == NULL) {
		fprintf(stderr, "Failed to allocate host vectors!\n");
		exit(EXIT_FAILURE);
	}

	// Initialize the host input vectors
	for (int i = 0; i < NUM_ELEMENTS; ++i) {
		h_A[i] = rand() / (float)RAND_MAX;
		//h_A[i] = 1;
	}

	// Allocate the device input vector A
	float* d_A = NULL;
	err = cudaMalloc((void**)& d_A, size);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Allocate the device output vector C
	float* d_C = NULL;
	err = cudaMalloc((void**)& d_C, size);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	int kernel_execs = 0, b_base = 1;
	while (b_base < NUM_ELEMENTS) {
		b_base *= THREADS_PER_BLOCK * 2;
		kernel_execs++;
	}
	printf("Number of kernel executions: %d\n", kernel_execs);

	err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	int blocks = -1;
	int elements = NUM_ELEMENTS;
	for (int i = 0; i < kernel_execs; i++) {
		int blocks = ceil(elements / (THREADS_PER_BLOCK * 2.0));
		printf("NumBlocks: %d\n", blocks);
		vectorAdd << < blocks, THREADS_PER_BLOCK >> > (d_A, d_C);
		err = cudaGetLastError();
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}
		
		err = cudaMemcpy(d_A, d_C, size, cudaMemcpyDeviceToDevice);
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to copy vector C from device to A device (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}



		//////////////////////
		//err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
		/*if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}

		for (int i = 0; i < NUM_ELEMENTS; i++) {
			printf("%f, ", h_C[i]);
		}
		printf("\n");
		printf("WYNIK: %f", h_C[0]);*/
		/////////////////////////////



		if (i + 1 < kernel_execs) {
			err = cudaMemset(d_C, 0, size);
			if (err != cudaSuccess)
			{
				fprintf(stderr, "Failed to clear C device (error code %s)!\n", cudaGetErrorString(err));
				exit(EXIT_FAILURE);
			}
		}
		elements = blocks;
	}
	err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	//TO TESTS
	///////////////// Verify that the result vector is correct
	float checkSum = 0;
	for (int i = 0; i < NUM_ELEMENTS; ++i) {
		checkSum += h_A[i];
	}
	float maxError = checkSum * 0.0001;
	if (fabs(checkSum - h_C[0]) > maxError) {
		fprintf(stderr, "Result verification failed! CPU sum: %f, GPU sum: %f\n", checkSum, h_C[0]);
	}
	else {
		printf("Result verification success! CPU sum: %f, GPU sum: %f\n", checkSum, h_C[0]);
	}
	///////////////////////////////////////////////

	// Free device global memory
	err = cudaFree(d_A);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}


	err = cudaFree(d_C);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Free host memory
	free(h_A);
	free(h_C);

	printf("Done\n");
	return 0;
}