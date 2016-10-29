project: framework.cu kernel.cu kernel_CPU.C
	nvcc -arch=sm_30 -o project framework.cu
clean:
	rm -r project
