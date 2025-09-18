
ARCH ?= sm_90a
INCLUDES = -I /root/wkspace/cutlass/include \
		   -I /root/wkspace/cutlass/tools/util/include

# if build directory not exist, create it
ifeq ("$(wildcard build)","")
$(shell mkdir build)
endif

buildcute:
	nvcc $(INCLUDES) -std=c++17 -arch=$(ARCH) min_tma_ptx_probe_sm90a.cu \
	    -Xptxas -v -lineinfo  -ptx -o build/min_tma_ptx_probe_${ARCH}.ptx
# 	nvcc $(INCLUDES) -std=c++17 -arch=$(ARCH) min_tma_ptx_probe_sm90a.cu -o min_tma_ptx_probe_sm90a.cubin \
# 	    -Xptxas -v -lineinfo 2>&1 | tee build_exe.log

inlineptx:
	nvcc $(INCLUDES) -std=c++17 -arch=$(ARCH) inline_ptx_cp_async_tensor_sm90a.cu \
	    -Xptxas -v -lineinfo -Xcompiler "-O3 -finline-functions" -ptx -o build/inline_ptx_cp_async_tensor_${ARCH}.ptx 2>&1 | tee build_ptx.log

inlineexe:
	nvcc $(INCLUDES) -std=c++17 -arch=$(ARCH) inline_ptx_cp_async_tensor_sm90a.cu -o inline_ptx_cp_async_tensor_${ARCH}.cubin \
	    -Xptxas -v -lineinfo -lcuda -Xcompiler "-O3 -finline-functions" 2>&1 | tee build_exe.log


.PHONY: build clean
clean:
	rm -rf build min_tma_ptx_probe_${ARCH} build_*.log