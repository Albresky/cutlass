# ARCH should be >= sm_90a. e.g., sm_90a, sm_100a
ARCH ?= sm_100a
INCLUDES = -I /root/wkspace/cutlass/include \
		   -I /root/wkspace/cutlass/tools/util/include

ifeq ("$(wildcard build)","")
$(shell mkdir build)
endif

DATETIME=$(shell date +%Y%m%d_%H%M%S)
LOGFILE=build/build_${ARCH}_$(DATETIME).log

ptx:
	nvcc $(INCLUDES) -std=c++17 -arch=$(ARCH) inline_ptx_cp_async_tensor_sm90a.cu \
	    -Xptxas -v -lineinfo -Xcompiler "-O3 -finline-functions" -ptx -o build/inline_ptx_cp_async_tensor_${ARCH}.ptx 2>&1 | tee build/build_ptx_${ARCH}_$(DATETIME).log

exe:
	nvcc $(INCLUDES) -std=c++17 -arch=$(ARCH) inline_ptx_cp_async_tensor_sm90a.cu -o build/inline_ptx_cp_async_tensor_${ARCH}.cubin \
	    -Xptxas -v -lineinfo -lcuda -Xcompiler "-O3 -finline-functions" 2>&1 | tee build/build_exe_${ARCH}_$(DATETIME).log

run:
	./build/inline_ptx_cp_async_tensor_${ARCH}.cubin 2>&1 | tee build/run_${ARCH}_$(DATETIME).log


.PHONY: ptx exe clean
clean:
	rm -rf build *.log