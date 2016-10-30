#define N_ITERATIONS 8

#define BLOCK_WIDTH  32
#define BLOCK_HEIGHT 32

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

__global__ void mySolve(const int *matrix, float *col, float *row, const int height, const int width) {
	size_t laneID = threadIdx.x & 31;

	size_t x = blockDim.x * blockIdx.x + threadIdx.x; // absolutni X pozice v matici
	size_t y = 32 * blockIdx.y + N_ITERATIONS * threadIdx.y; // absolutni Y pozice pocatku pocitani

	size_t idx = y * width + x;

	int vertical = 0;

	for(int i = 0; i < N_ITERATIONS; ++i) {
		int v = matrix[idx];
		vertical += v;
		for(unsigned w = 16; w >= 1; w >>= 1)
			v += __shfl_down(v, w);
		if(laneID == 0)
			atomicAdd(&row[y+i], v);
		idx += width;
	}

	atomicAdd(&col[x], vertical);
}

__global__ void bothAverage(float* first, float *second, const int firstLen, const int secondLen) {
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < firstLen)
		first[index] = first[index] / secondLen;
	if (index < secondLen)
		second[index] = second[index] / firstLen;
}

void solveGPU(const int *matrix, float *avg_stud, float *avg_que, const int students, const int questions) {
	cudaMemset(avg_que, 0, questions*sizeof(*avg_que));
	cudaMemset(avg_stud, 0, students*sizeof(*avg_stud));
	mySolve<<<dim3(questions/32, students/32), dim3(32, 32/N_ITERATIONS)>>>(matrix, avg_que, avg_stud, students, questions);
	bothAverage<<<dim3(MAX(students,questions)/32), dim3(32, 1)>>>(avg_que, avg_stud, questions, students);
}
