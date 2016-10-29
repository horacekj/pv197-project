project: framework.cu kernel.cu kernel_CPU.C
	nvcc -o project framework.cu
clean:
	rm -r project
