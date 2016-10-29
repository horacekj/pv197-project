project: framework.cu kernel.cu kernel_CPU.C
	nvcc -arch=sm_30 -o project framework.cu

test: project
	nvprof -f -o metrics.nvprof ./project

showtest:
	nvvp metrics.nvprof

clean:
	rm -r project
