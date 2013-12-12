all:	lic.cc vector.h
	nvcc -O2 -arch=sm_20 lic.cc lic.cu -o lic

clean:
	rm -rf *.o lic
