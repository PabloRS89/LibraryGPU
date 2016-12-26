/*
 *		r3d.c		
 *		See r3d.h for usage.
 *		Devon Powell
 *		31 August 2015
 *		This program was prepared by Los Alamos National Security, LLC at Los Alamos National
 *		Laboratory (LANL) under contract No. DE-AC52-06NA25396 with the U.S. Department of Energy (DOE). 
 *		All rights in the program are reserved by the DOE and Los Alamos National Security, LLC.  
 *		Permission is granted to the public to copy and use this software without charge, provided that 
 *		this Notice and any statement of authorship are reproduced on all copies.  Neither the U.S. 
 *		Government nor LANS makes any warranty, express or implied, or assumes any liability 
 *		or responsibility for the use of this software.
 */
#include "r3d.h"
 #include "cur3d.h"
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/random.h>
#include <thrust/generate.h>
#include <thrust/detail/type_traits.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <cufft.h>

// size of the grid
#define NGRID 23
// order of polynomial integration for all tests 
#define POLY_ORDER 2

// numerical tolerances for pass/warn/fail tests
#define TOL_WARN 1.0e-8
#define TOL_FAIL 1.0e-4
#define MAX_THREADS_BLOCK 512

// minimum volume allowed for test polyhedra 
#define MIN_VOL 1.0e-8

// useful macros
#define ONE_THIRD 0.333333333333333333333333333333333333333333333333333333
#define ONE_SIXTH 0.16666666666666666666666666666666666666666666666666666667
 #define CLIP_MASK 0x80
#define dot(va, vb) (va.x*vb.x + va.y*vb.y + va.z*vb.z)
#define wav(va, wa, vb, wb, vr) {			\
	vr.x = (wa*va.x + wb*vb.x)/(wa + wb);	\
	vr.y = (wa*va.y + wb*vb.y)/(wa + wb);	\
	vr.z = (wa*va.z + wb*vb.z)/(wa + wb);	\
}

#define norm(v) {					\
	r3d_real tmplen = sqrt(dot(v, v));	\
	v.x /= (tmplen + 1.0e-299);		\
	v.y /= (tmplen + 1.0e-299);		\
	v.z /= (tmplen + 1.0e-299);		\
}

#define dot3(va, vb) (va.x*vb.x + va.y*vb.y + va.z*vb.z)
#define norm3(v) {					\
	r3d_real tmplen = sqrt(dot3(v, v));	\
	v.x /= (tmplen + 1.0e-299);		\
	v.y /= (tmplen + 1.0e-299);		\
	v.z /= (tmplen + 1.0e-299);		\
}

// Timing variables
clock_t t_ini,t_fin;
float milliseconds = 0;
cudaEvent_t start, stop;	    		
cublasHandle_t handle;

__host__ void cur3d_err(cudaError_t err) {
	if (err != cudaSuccess) {
		printf("%s\n", cudaGetErrorString(err));
	}
}


//This function calculates the average of the data array over the CPU in a sequential programming
float avg_CPU(size_t size, float *pos){
	float avg = 0;
	unsigned int x;
	for(x = 0; x<size; x++){
		avg = avg + pos[x];
	}
	avg = avg / size;
	return avg;
}

//This function is a kind of helper to qsot to determine if the value is greater or lower than..
int compare(const void *a, const void *b) {
    return ( *(int*)a - *(int*)b );
}

//Function which calculates the medium value of the dataset in a sequential programming using qsort algorithm
float medium_CPU(size_t size, float *pos){
	float medium, medium1, medium2;
	int x = (int)floor(size/2);
	qsort(pos, size, sizeof(float), &compare);	
	if(size%2 == 0){
		medium1 = pos[x];
		medium2 = pos[x-1];
		medium = (medium1 + medium2) / 2;
	}else{
		medium = pos[x];
	}		
	return medium;
}
//Calculate the standard deviation in sequential programming over CPU
float StDev_CPU(size_t size, float *pos){
	float medium = 0, ed = 0;
	unsigned int x;
	for(x = 0; x<size; x++){
		medium = medium + pos[x];
	}
	medium = medium / size;		
	for(x = 0; x<size; x++){
		ed = ed + pow(pos[x]-medium, 2);
	}	
	ed = ed / size;
	ed = sqrt(ed);
	return ed;
}
//Calculate the standard deviation in sequential programming over GPU
__global__ void StDev_GPU(size_t size, float *pos, float *medium, float *ED){		
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;	
	float tmp;
	if (idx >= size) {
 		return;
   	}	
    	tmp = pow(pos[idx]-*medium,2);	
	__syncthreads();
	atomicAdd(ED, tmp);
	__syncthreads();
}

void medium_GPU(size_t size, float *pos){
	//FILE *f = fopen("times_Medium.csv", "a");
	thrust::host_vector<float> h_keys(size);		
	unsigned int i;
	for(i=0; i<size; i++){
		h_keys[i] = pos[i];
	}
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0); 

	thrust::device_vector<float> d_values = h_keys;
	thrust::sort(d_values.begin(), d_values.end());
	thrust::host_vector<float> h_values = d_values;

	cudaEventSynchronize(stop);
	cudaEventRecord(stop, 0);	    	    	
	cudaEventSynchronize(stop);	

	bool bTestResult = thrust::is_sorted(h_values.begin(), h_values.end());
	float m1, m2, medium;
    	int x = (int)floor(size/2);
    	if(size%2 == 0){
		m1 = h_values[x];
		m2 = h_values[x-1];
		medium = (m1 + m2) / 2;
	}else{
		medium = h_values[x];
	}
	if(bTestResult)
	{
		printf("Medium in GPU:%f\n",medium);
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("\tms: %f\n",milliseconds);
		//fprintf(f, "%f", milliseconds);
		//fclose(f);
	}
	else
		printf("No sorted\n");
}

__global__ void real2Complex(float *a, cufftComplex *c, size_t N){
	int idx = blockIdx.x*blockDim.x+threadIdx.x;	

	if(idx < N){
		int index = idx * N;
		c[index].x = a[index];		
	}
}

extern "C"{
	void calc_FFT(size_t size, float *pos){
		//printf("Calculando FFT\n");
		//FILE *f = fopen("FFT.csv", "a");
		cufftComplex *h_data = (float2 *) malloc(sizeof(float2) * size);
		cufftComplex *r_data = (float2 *) malloc(sizeof(float2) * size);

		for(unsigned int i=0; i<size; i++){
			h_data[i].x = i;
			h_data[i].y = pos[i];
		}

		cufftComplex *d_data;
		cudaMalloc((void **)&d_data, size*sizeof(cufftComplex));
		cudaMemcpy(d_data, h_data, size*sizeof(cufftComplex), cudaMemcpyHostToDevice);

		cufftHandle plan;
		cufftPlan1d(&plan, size, CUFFT_C2C, 1);

		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);

		cufftExecC2C(plan, (cufftComplex *)d_data, (cufftComplex *)d_data, CUFFT_FORWARD);

		cudaEventSynchronize(stop);
		cudaEventRecord(stop, 0);	    	    	
		cudaEventSynchronize(stop);		
		cudaEventElapsedTime(&milliseconds, start, stop);

		//cufftExecC2C(plan, (cufftComplex *)d_data, (cufftComplex *)d_data, CUFFT_INVERSE);

		cudaMemcpy(r_data, d_data, size*sizeof(cufftComplex), cudaMemcpyDeviceToHost);
		
		for(unsigned int i=0; i<10; i++){
			pos[i] = r_data[i].x / (float)size;		
			//pos[i] = r_data[i].x;
			//printf("%f, %f\n", r_data[i].x, r_data[i].y);
		}

		cufftDestroy(plan);
		free(h_data);		
		cudaFree(d_data);

		printf("CUFFT:\n");
		printf("\tms: %f\n",milliseconds);
		//printf(f, "%f\n", milliseconds);
		//fclose(f);		
	}
	
	//Calculate the average of the input data ´pos´
	float calc_avg(size_t size, float *pos){	
		float result, *d_pos, avg_gpu, average_cpu;
		//FILE *f = fopen("times_Average.csv", "a");
	
		cublasCreate(&handle);
		cudaMalloc((void **)&d_pos, size * sizeof(float));
		cudaMemcpy(d_pos, pos, size * sizeof(float), cudaMemcpyHostToDevice);

		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);

		cublasSasum(handle, size, d_pos, 1, &result);		
		avg_gpu = result/size;

		cudaEventSynchronize(stop);
		cudaEventRecord(stop, 0);	    	    	
		cudaEventSynchronize(stop);		
		cudaEventElapsedTime(&milliseconds, start, stop);

		cudaFree(d_pos);
		cublasDestroy(handle);
		printf("\n\n");
		printf("Average in GPU: %f\n", avg_gpu);
		printf("\tms: %f\n",milliseconds);

		t_ini=clock();	    
		average_cpu = avg_CPU(size, pos);	    
		t_fin=clock();
		printf("Average in CPU:%f\n", average_cpu);	    
		printf("\tms: %f\n\n",((double)(t_fin-t_ini)/CLOCKS_PER_SEC)*1000);		
		//fprintf(f, "%f %f\n", milliseconds, ((double)(t_fin-t_ini)/CLOCKS_PER_SEC)*1000);
		//fclose(f);      	   
		return average_cpu;
	}
	
   	//Calculate the medium value   	
	void calc_medium(size_t size, float *pos){
		medium_GPU(size, pos);
		//FILE *f = fopen("times_Medium.csv", "a");
		t_ini=clock();
		float M = medium_CPU(size, pos);
		t_fin=clock();
		printf("Medium in CPU:%f\n",M);
		printf("\tms: %f\n\n",((double)(t_fin-t_ini)/CLOCKS_PER_SEC)*1000);
		//fprintf(f, " %f\n", ((double)(t_fin-t_ini)/CLOCKS_PER_SEC)*1000);
		//fclose(f);
	}
	//Calculate the standard deviation value
	void calc_StDev(size_t size, float *pos){		
		float result;
		float *d_pos, *medium, *m, *SD, *sd;
		int block, thread;
		//FILE *f = fopen("times_StDev.csv", "a");

		block = ceil(size/MAX_THREADS_BLOCK)+1;
	   	thread = MAX_THREADS_BLOCK;

		dim3 BLOCK(block);
		dim3 THREAD(thread);

		cublasCreate(&handle);
		cudaMalloc((void **)&d_pos, size * sizeof(float));
		cudaMemcpy(d_pos, pos, size * sizeof(float), cudaMemcpyHostToDevice);

		cublasSasum(handle, size, d_pos,1,&result);

		cudaFree(d_pos);
		cublasDestroy(handle);					

		cudaMalloc((void **)&d_pos, size * sizeof(float));
		cudaMalloc((void **)&medium, sizeof(float));
		cudaMalloc((void **)&SD, sizeof(float));
		m = (float *)malloc(sizeof(float));
		sd = (float *)malloc(sizeof(float));	    
		*m = result/size;
		cudaMemcpy(d_pos, pos, size * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(medium, m, sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(SD, sd, sizeof(float), cudaMemcpyHostToDevice);
			  
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);   	    
		StDev_GPU <<< BLOCK, THREAD >>> (size, d_pos, medium, SD);
		cudaMemcpy(sd, SD, sizeof(float), cudaMemcpyDeviceToHost);
		float sd_result = *sd;	    	    
		sd_result = sd_result / size;
		sd_result = sqrt(sd_result);
		cudaEventSynchronize(stop);
		cudaEventRecord(stop, 0);	    	    	
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);	    	    	    	    
		printf("Standar Deviation in GPU:%f\n",sd_result);	    
		printf("\tms: %f\n",milliseconds);	    
		cudaEventDestroy(start);
		cudaEventDestroy(stop);	  
		cudaFree(d_pos);  
		cudaFree(medium);
		free(m);
		
		t_ini=clock();
		float d = StDev_CPU(size, pos);
		t_fin=clock();
		printf("Standar Deviation in CPU:%f\n",d);
		printf("\tms: %f\n\n",((double)(t_fin-t_ini)/CLOCKS_PER_SEC)*1000);
		//fprintf(f, "%f %f\n", milliseconds, ((double)(t_fin-t_ini)/CLOCKS_PER_SEC)*1000);
		//fclose(f);
	}
	//Calculate the minimum and maximum value existing in dataset
	void calc_MaxMin(size_t size, float *pos){		
		int d_max, d_min;
		float *d_pos;    
		//FILE *f = fopen("times_maxmin.csv", "a");

		cublasCreate(&handle);
		cudaMalloc((void **)&d_pos, size * sizeof(float));
		cudaMemcpy(d_pos, pos, size * sizeof(float), cudaMemcpyHostToDevice);

		cudaEventCreate(&start);
	   	cudaEventCreate(&stop);
	   	cudaEventRecord(start, 0);
		
		cublasIsamax(handle, size, d_pos,1,&d_max);
		cublasIsamin(handle, size, d_pos,1,&d_min);

		cudaEventSynchronize(stop);
	    	cudaEventRecord(stop, 0);	    	    	
	    	cudaEventSynchronize(stop);		
		cudaEventElapsedTime(&milliseconds, start, stop);		

		cudaFree(d_pos);
		cublasDestroy(handle);
		printf("GPU:\n");
		printf(" - Minimum: %f\n", pos[d_min-1]);
		printf(" - Maximun: %f\n", pos[d_max-1]);
		printf("\tms: %f\n",milliseconds);

		float min = 0;
		float max = 0;
		
		t_ini=clock();
		qsort(pos, size, sizeof(float), &compare);
		min = pos[0];
		max = pos[size-1];
		t_fin=clock();

		printf("CPU:\n");
		printf(" - Minimum: %f\n", min);
		printf(" - Maximun: %f\n", max);
		printf("\tms: %f\n\n",((double)(t_fin-t_ini)/CLOCKS_PER_SEC)*1000);
		//fprintf(f, "%f %f\n", milliseconds, ((double)(t_fin-t_ini)/CLOCKS_PER_SEC)*1000);
		//fclose(f);		
	}

	__host__ void voxelization() {
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start);
		setbuf(stdout, NULL);		

		// free device arrays
		cudaEventSynchronize(stop);
		float ms = 0;
		cudaEventElapsedTime(&ms, start, stop);
		printf("\n\n");
		printf("milliseconds: %f\n", ms);
		printf("Testing compilation\n");
		return;
	}
}