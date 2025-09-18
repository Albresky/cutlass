
ARCH ?= sm_100a
INCLUDES = -I /root/wkspace/cutlass/include \
		   -I /root/wkspace/cutlass/tools/util/include

# if build directory not exist, create it
ifeq ("$(wildcard build)","")
$(shell mkdir build)
endif

DATETIME=$(shell date +%Y%m%d_%H%M%S)
LOGFILE=build/build_${ARCH}_$(DATETIME).log

buildcute:
	nvcc $(INCLUDES) -std=c++17 -arch=$(ARCH) min_tma_ptx_probe_sm90a.cu \
	    -Xptxas -v -lineinfo  -ptx -o build/min_tma_ptx_probe_${ARCH}.ptx
# 	nvcc $(INCLUDES) -std=c++17 -arch=$(ARCH) min_tma_ptx_probe_sm90a.cu -o min_tma_ptx_probe_sm90a.cubin \
# 	    -Xptxas -v -lineinfo 2>&1 | tee build/build_exe_${ARCH}_$(DATETIME).log

inlineptx:
	nvcc $(INCLUDES) -std=c++17 -arch=$(ARCH) inline_ptx_cp_async_tensor_sm90a.cu \
	    -Xptxas -v -lineinfo -Xcompiler "-O3 -finline-functions" -ptx -o build/inline_ptx_cp_async_tensor_${ARCH}.ptx 2>&1 | tee build/build_ptx_${ARCH}_$(DATETIME).log

inlineexe:
	nvcc $(INCLUDES) -std=c++17 -arch=$(ARCH) inline_ptx_cp_async_tensor_sm90a.cu -o inline_ptx_cp_async_tensor_${ARCH}.cubin \
	    -Xptxas -v -lineinfo -lcuda -Xcompiler "-O3 -finline-functions" 2>&1 | tee build/build_exe_${ARCH}_$(DATETIME).log

run:
	./inline_ptx_cp_async_tensor_${ARCH}.cubin 2>&1 | tee build/run_${ARCH}_$(DATETIME).log


.PHONY: build clean
clean:
	rm -rf build min_tma_ptx_probe_${ARCH} *.log