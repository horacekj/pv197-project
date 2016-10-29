#define N_ITERATIONS 32

#define BLOCK_WIDTH  32
#define BLOCK_HEIGHT 32

__global__ void mySolve(const int *matrix, float *row, float *col, const int width, const int height) {
	size_t laneID = threadIdx.x & 31;

	size_t x = blockDim.x * blockIdx.x + threadIdx.x;
	size_t y = blockDim.y * blockIdx.y;

	size_t idx = y * width + x;

	float vertical = 0;

	for(int i = 0; i < blockDim.y; ++i) {
		float v = matrix[idx];
		vertical += v;
		for(unsigned w = 16; w >= 1; w >>= 2)
			v += __shfl_down(v, w);
		if(laneID == 0)
			atomicAdd(&row[y], v);
		y++;
		idx += width;
	}

	atomicAdd(&col[x], vertical);
}

void solveGPU(const int *matrix, float *row, float *col, const int width, const int height) {
	mySolve<<<width/BLOCK_WIDTH, height/BLOCK_HEIGHT>>>(matrix, row, col, width, height);
}
