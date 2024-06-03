#include <iostream>
#include <CL/cl.hpp>
#include <cassert>
#include <fstream>
#include <memory>
#include <chrono>

#define GPU_TO_USE "gfx1032"
#define PLATFORM_TO_USE "AMD Accelerated Parallel Processing"
#define PROGRAM_SOURCE_PATH "..//minReduction.cl"
#define N_ELEMENTS ((uint64_t)268435456)
#define SEPARATOR "--------------------------------------------\n"

template <class T>
T sumReductionCpu(T array[], uint64_t size);

float minReductionCpu(float array[], uint64_t size);

float* createdArray(uint32_t size);

int main(int arg, char* args[])
{

    float* testArray = createdArray(N_ELEMENTS);
    auto astart_time = std::chrono::steady_clock::now();
    auto aend_time = std::chrono::steady_clock::now();


    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    std::cout << "Platforms:\n";
    cl::Platform platformToUse;
    cl::Device deviceToUse;
    std::vector<cl::Device> devicesToUse;
    for(cl::Platform currentPlatform : platforms)
    {
        std::string currentPlatformName(currentPlatform.getInfo<CL_PLATFORM_NAME>());
        if(currentPlatformName == std::string(PLATFORM_TO_USE))
        {
            platformToUse = currentPlatform;
            std::cout << "-->";
        }
        std::cout << "\t" << currentPlatformName << "\n";

        std::vector<cl::Device> devices;
        currentPlatform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

        std::cout << "\tDevices:\tname\tunits\twg_s\tloc_mem\tglo_mem\t\tmhz\n";
        for(cl::Device currentDevice : devices)
        {
            std::string currentDeviceName(currentDevice.getInfo<CL_DEVICE_NAME>());
            std::cout << "\t";
            if(currentDeviceName == std::string(GPU_TO_USE))
            {
                deviceToUse = currentDevice;
                std::cout << "-->";
            }
            std::cout << "\t";
            std::cout <<    "\t" << currentDeviceName <<
                            "\t" << currentDevice.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() <<
                            "\t" << currentDevice.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() <<
                            "\t" << currentDevice.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() <<
                            "\t" << currentDevice.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() <<
                            "\t" << currentDevice.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() << "\n";
        }
    }
    platformToUse.getDevices(CL_DEVICE_TYPE_ALL, &devicesToUse);
    std::cout << SEPARATOR;
    std::cout << "CPU:\n";
    astart_time = std::chrono::steady_clock::now();
    std::cout << "sum: " << sumReductionCpu(testArray, N_ELEMENTS) << " - ";
    aend_time = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(aend_time - astart_time).count() << "ms\n";
    astart_time = std::chrono::steady_clock::now();
    std::cout << "min: " << minReductionCpu(testArray, N_ELEMENTS) << " - ";
    aend_time = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(aend_time - astart_time).count() << "ms\n";
    std::cout << SEPARATOR;
    //exit(1);
    cl::Context context(devicesToUse);

    cl::CommandQueue commandQueue(context, deviceToUse);

    std::unique_ptr<uint64_t[]> A(new uint64_t[N_ELEMENTS]);
    std::unique_ptr<uint64_t[]> B(new uint64_t[N_ELEMENTS]);
    std::unique_ptr<uint64_t[]> C(new uint64_t[N_ELEMENTS]);

    for(uint64_t i = 0; i < N_ELEMENTS; i++)
    {
        A[i] = i;
        B[i] = N_ELEMENTS - i;
    }

    cl::Buffer bufferA = cl::Buffer(context, CL_MEM_READ_ONLY, N_ELEMENTS*sizeof(uint64_t));
    cl::Buffer bufferB = cl::Buffer(context, CL_MEM_READ_ONLY, N_ELEMENTS*sizeof(uint64_t));
    cl::Buffer bufferC = cl::Buffer(context, CL_MEM_WRITE_ONLY, N_ELEMENTS*sizeof(uint64_t));

    commandQueue.enqueueWriteBuffer( bufferA, CL_FALSE, 0, N_ELEMENTS * sizeof(uint64_t), A.get() );
    commandQueue.enqueueWriteBuffer( bufferB, CL_FALSE, 0, N_ELEMENTS * sizeof(uint64_t), B.get() );

    std::ifstream sourceFile(PROGRAM_SOURCE_PATH);
    std::string sourceCode(std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));
    cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length()));
    cl::Program program = cl::Program(context, source);


    program.build(devicesToUse);
    std::cout << "build log:\n" << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devicesToUse[0]);
    std::cout << SEPARATOR;
    cl::Kernel kernel (program, "vector_sum") ;

    kernel.setArg( 0, bufferA );
    kernel.setArg( 1, bufferB );
    kernel.setArg( 2, bufferC );

    cl::NDRange global(N_ELEMENTS);
    cl::NDRange local(256);
    commandQueue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local);

    cl::finish();

    commandQueue.enqueueReadBuffer(bufferC, CL_TRUE, 0, N_ELEMENTS * sizeof(uint64_t), C.get());

    bool result = true;
    for (uint64_t i = 0; i < N_ELEMENTS; i ++)
    {
        if (C[i] != A[i] + B[i])
        {
            result = false;
            break;
        }
    }
    std::cout << result << "\n";

    delete(testArray);
    return 0;
}
template <class T>
T sumReductionCpu(T array[], uint64_t size)
{
    T sum = static_cast<T>(0);
    for(uint64_t i = 0; i < size; i++)
    {
        sum += array[i];
    }
    return sum;
}
float* createdArray(uint32_t size)
{
    float* array = (float*)malloc(size*sizeof(float));
    for(uint32_t i = 0; i < size; i++)
    {
        array[i] = 13 + i / 4;
    }
    return array;
}
float minReductionCpu(float array[], uint64_t size)
{
    float min = array[0];
    for(int i = 1; i < size; i++)
    {
        if(min > array[i])
        {
            min = array[i];
        }
    }
    return min;
}