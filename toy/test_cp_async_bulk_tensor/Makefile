
ARCH ?= sm_90a
INCLUDES = -I /root/wkspace/cutlass/include -I /root/wkspace/cutlass/tools/util/include

# if build directory not exist, create it
ifeq ("$(wildcard build)","")
$(shell mkdir build)
endif

buildcute:
	nvcc $(INCLUDES) -std=c++17 -arch=$(ARCH) min_tma_ptx_probe_sm90a.cu -o min_tma_ptx_probe_sm90a \
	    -Xptxas -v -lineinfo  -ptx -o build/min_tma_ptx_probe_sm90a.ptx
# 	nvcc $(INCLUDES) -std=c++17 -arch=$(ARCH) min_tma_ptx_probe_sm90a.cu -o min_tma_ptx_probe_sm90a \
# 	    -Xptxas -v -lineinfo 2>&1 | tee build.log

buildinline:
	nvcc $(INCLUDES) -std=c++17 -arch=$(ARCH) inline_ptx_cp_async_tensor_sm90a.cu \
	    -Xptxas -v -lineinfo -Xcompiler "-O3 -finline-functions" -ptx -o build/inline_ptx_cp_async_tensor_sm90a.ptx 2>&1 | tee build.log

.PHONY: build clean
clean:
	rm -rf build min_tma_ptx_probe_sm90a build.log