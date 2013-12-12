/* *
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <stdio.h>
#include <stdlib.h>
#include "point.h"
#include <cuda.h>
static const int WORK_SIZE = 256;

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }

__host__ __device__ unsigned int bitreverse(unsigned int number) {
	number = ((0xf0f0f0f0 & number) >> 4) | ((0x0f0f0f0f & number) << 4);
	number = ((0xcccccccc & number) >> 2) | ((0x33333333 & number) << 2);
	number = ((0xaaaaaaaa & number) >> 1) | ((0x55555555 & number) << 1);
	return number;
}

/**
 * CUDA kernel function that reverses the order of bits in each element of the array.
 */
__global__ void bitreverse(void *data) {
	unsigned int *idata = (unsigned int*) data;
	idata[threadIdx.x] = bitreverse(idata[threadIdx.x]);
}

__device__ int getPixel(const Point &p, int &i, int &j, int rows, int cols)
{
  i = ((int)p.coord[0] + rows)%rows;
  j = ((int)p.coord[1] + cols)%cols;

  if (i >= 0 && i < rows && j >=0 && j < cols)
    return 1;
  return 0;
}

__device__ Vector getVector(const Point &p,   Vector *vecdata, int rows, int cols)
{
  int i,j;

  if (getPixel(p,i,j, rows, cols)) {
    return vecdata[i * cols +j];
  }
  return(Vector(0,0,0));
}

__device__ void RK(Point &p, double h, Vector *vecdata, int rows, int cols)
{
  Vector v;
  Vector k1,k2,k3,k4;

  v = getVector(p, vecdata, rows, cols);
  if (!v.iszero())
    v = v.unit();
  //v.Print();

  k1 = v*h;
  v = (getVector(p+k1*.5, vecdata, rows, cols));
  if (!v.iszero())
    v = v.unit();
  //v.Print();

  k2 = v*h;
  v = (getVector(p+k2*.5, vecdata, rows, cols));
  if (!v.iszero())
    v = v.unit();
  //v.Print();

  k3 = v*h;
  v = (getVector(p+k3, vecdata, rows, cols));
  if (!v.iszero())
    v = v.unit();
  //v.Print();

  k4 = v*h;
  p += k1/6 + k2/3 + k3/3 + k4/6;
}

__device__ void GenStreamLine(int i, int j, Point* bwd, Point* fwd, Vector *vecdata, int rows, int cols, Point* origin)
{
  Point b,f;

  *origin = f = b = Point(i+.5,j+.5);
  for (int k=0; k<M+L-1; k++) {
    RK(f,Ht, vecdata, rows, cols);
    fwd[k] = f;
    RK(b,-Ht,vecdata, rows, cols);
    bwd[k] = b;
  }
}

__device__  int validpt(Point &p, int rows,int cols) {
  int i,j;

  if (getPixel(p,i,j, rows, cols))
    return 1;
  return 0;
}

__device__ Point getSLIndex(int m, Point* bwd, Point* fwd, Point origin) {
    if (m == 0)
      return origin;
    else if (m>0)
      return fwd[m-1];
    else
      return bwd[-m-1];
}

__device__ inline double getT(Point &p, int *texdata, int rows, int cols)
{
  int i,j;

  if (getPixel(p,i,j, rows, cols))
    return texdata[i * cols + j];
  return 0;
}

__device__ double ComputeI(Point* bwd, Point* fwd, Point origin, int &numvalid, int rows, int cols, double *Idata, int *hitdata, int *texdata)
{
  double T,k,I;
  int i,j;

  T=0;
  numvalid = 0;


  for(i=-L; i<= L; i++) {
    Point p = getSLIndex(i, bwd, fwd, origin);
    if (validpt(p, rows, cols)) {
      T += getT(p, texdata, rows, cols);
      numvalid++;
    }
  }
  if (getPixel(origin, i, j, rows, cols)) {
    k = 1./numvalid;
    // printf("GPU inside IF statement");
    Idata[i * cols + j] += I = T*k;
    // printf("[%d] = %lf", i * cols + j, T*k);
    hitdata[i * cols + j]++;
    return I;
  }
  return 0;
}

__device__ double ComputeIFwd(Point* bwd, Point* fwd, Point origin, double &I, int m, int &numvalid, int rows, int cols, double *Idata, int *hitdata, int *texdata) {
  int i,j;
  double k;

  Point p = getSLIndex(m, bwd, fwd, origin);
  if (getPixel(p,i,j,rows, cols)) {
    Point p1 = getSLIndex(m+L, bwd, fwd, origin);
    if (validpt(p1, rows, cols))
      numvalid++;
    Point p2 = getSLIndex(m-1-L, bwd, fwd, origin);
    if (validpt(p2, rows, cols))
      numvalid--;
    k = 1./numvalid;
    Idata[i * cols + j] += I += k*(getT(p1, texdata, rows, cols) - getT(p2, texdata, rows, cols));
    hitdata[i * cols +j]++;
    return I;
  }
  return 0;
}

__device__ double ComputeIBwd(Point* bwd, Point* fwd, Point origin, double &I, int m, int &numvalid, int rows, int cols, double *Idata, int *hitdata, int *texdata) {
  int i,j;
  double k;

  Point p = getSLIndex(m, bwd, fwd, origin);
  if (getPixel(p,i,j, rows, cols)) {
    Point p1 = getSLIndex(m-L, bwd, fwd, origin);
    if (validpt(p1, rows, cols))
      numvalid++;
    Point p2 = getSLIndex(m+1+L, bwd, fwd, origin);
    if (validpt(p2, rows, cols))
      numvalid--;
    k = 1./numvalid;
    Idata[i * cols + j] += I += k*(getT(p1, texdata, rows, cols) - getT(p2, texdata, rows, cols));
    hitdata[i * cols +j]++;
    return I;
  }
  return 0;
}

__global__ void Normalize(int rows, int cols, double *Idata, int *hitdata){
  for (int i=0; i<rows; i++)
    for (int j=0; j<cols; j++) {
      Idata[i * cols + j] /= hitdata[i * cols + j];
    }
}
__global__ void lic_kernel(int rows, int cols, Vector *vecdata, int *hitdata, int *texdata, double *Idata) {
  Point fwd[M+L-1];
  Point bwd[M+L-1];
  Point origin;
  int numvalid = 0;

  double I0;
  double I;
  int m;
  int tmpsum=0;


  for(int i = 0; i < rows; i++) {
    for(int j = 0; j < cols; j++) {
      GenStreamLine(i, j, bwd, fwd, vecdata, rows, cols, &origin);
      I = I0 = ComputeI(bwd, fwd, origin, numvalid, rows, cols, Idata, hitdata, texdata);
      tmpsum = numvalid;
      for (m=1; m < M; m++)
        ComputeIFwd(bwd, fwd, origin, I, m, tmpsum, rows, cols, Idata, hitdata, texdata);
      I = I0;
      tmpsum = numvalid;
      for (m=1; m < M; m++)
        ComputeIBwd(bwd, fwd, origin, I, -m, tmpsum, rows, cols, Idata, hitdata, texdata);
      // printf("%lf, ", Idata[i * cols + j]);
    }
  }
  return;
}



void licGPU(int rows, int cols, Vector *vecdata, int *texdata, double *IdataGPU) {
  Vector *vecdata_dev;
  int *hitdata_dev;
  int *texdata_dev;
  double *Idata_dev;


  CUDA_CHECK_RETURN(cudaMalloc((void**) &hitdata_dev, sizeof(int) * rows * cols));
  CUDA_CHECK_RETURN(cudaMalloc((void**) &vecdata_dev, sizeof(Vector) * rows * cols));
  CUDA_CHECK_RETURN(cudaMalloc((void**) &texdata_dev, sizeof(int) * rows * cols));
  CUDA_CHECK_RETURN(cudaMalloc((void**) &Idata_dev, sizeof(double) * rows * cols));


  // need to copy vecdata, texdata after readPts()
  CUDA_CHECK_RETURN(cudaMemcpy(vecdata_dev, vecdata, sizeof(Vector) * rows * cols, cudaMemcpyHostToDevice));
  CUDA_CHECK_RETURN(cudaMemcpy(texdata_dev, texdata, sizeof(int) * rows * cols, cudaMemcpyHostToDevice));
  CUDA_CHECK_RETURN(cudaMemset(hitdata_dev, 0, sizeof(int) * rows * cols));
  CUDA_CHECK_RETURN(cudaMemset(Idata_dev, 0, sizeof(double) * rows * cols));

  // set up parameters for threads structure
  dim3 dimGrid(1, 1);
  dim3 dimBlock(1, 1, 1);

  lic_kernel<<<dimGrid, dimBlock>>>(rows, cols, vecdata_dev, hitdata_dev, texdata_dev, Idata_dev);
  cudaThreadSynchronize();
  Normalize<<<dimGrid, dimBlock>>>(rows, cols, Idata_dev, hitdata_dev);
  cudaThreadSynchronize();


  CUDA_CHECK_RETURN(cudaMemcpy(IdataGPU, Idata_dev, sizeof(double) * rows * cols, cudaMemcpyDeviceToHost));
  // printf("\nFINALLLLL\n\n");
  // for(int i = 0; i < 10; i++) {
  //   for(int j = 0; j < 10; j++) {
  //     printf("%lf, ", Idata[i * cols + j]);
  //   }
  // }

  CUDA_CHECK_RETURN(cudaFree((void*) hitdata_dev));
  CUDA_CHECK_RETURN(cudaFree((void*) vecdata_dev));
  CUDA_CHECK_RETURN(cudaFree((void*) texdata_dev));
  CUDA_CHECK_RETURN(cudaFree((void*) Idata_dev));
}

