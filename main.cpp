#include <iostream>
#include <CL/cl.hpp>
#include <cassert>
#include <fstream>
#include <memory>
#include <chrono>
#include <numeric>
#include <thread>
#include <iomanip>

#define GPU_TO_USE "gfx1032"
#define PLATFORM_TO_USE "AMD Accelerated Parallel Processing"
#define PROGRAM_SOURCE_PATH "..//sumReduction6.cl"
#define LOCAL_SIZE 128
#define WORK_GROUP_COUNT 64
#define N_ELEMENTS (LOCAL_SIZE*WORK_GROUP_COUNT*(1 << 16)) //268435456 32768
#define SEPARATOR "--------------------------------------------\n"
#define CHECK_ERROR(err) if (err != CL_SUCCESS) { std::cerr << "OpenCL error: " << err << std::endl; exit(EXIT_FAILURE); }
#define DATA_TYPE uint32_t
#define MAX_DATA_SIZE_SHIFTS 16
#define AVERAGE_OUT_OF 42

uint32_t measureSetupTime = 0;
uint32_t sumReductionCpu(std::vector<uint32_t>* array, uint64_t size);

std::vector<uint32_t>* createdArray(uint32_t size);
int SingleTest(void);

int testHost();
uint64_t test1SingleCoreCPU(uint32_t* correctResult, std::vector<uint32_t>* arr, size_t size);
void test2MultiCoreCPUPartialSum(const std::vector<uint32_t>& arr, size_t start, size_t end, uint32_t& result);
uint64_t test2MultiCoreCPU(uint32_t correctResult, std::vector<uint32_t>* arr, size_t size);
uint64_t test3Dournac(uint32_t correctResult, std::vector<uint32_t>* arr, size_t size, cl::Context context, cl::CommandQueue commandQueue, std::vector<cl::Device>& devicesToUse);
uint64_t test4Catanzaro(uint32_t correctResult, std::vector<uint32_t>* arr, size_t size, cl::Context context, cl::CommandQueue commandQueue, std::vector<cl::Device>& devicesToUse);
uint64_t test5Divergence(uint32_t correctResult, std::vector<uint32_t>* arr, size_t size, cl::Context context, cl::CommandQueue commandQueue, std::vector<cl::Device>& devicesToUse);
uint64_t test6LoopUnrolling(uint32_t correctResult, std::vector<uint32_t>* arr, size_t size, cl::Context context, cl::CommandQueue commandQueue, std::vector<cl::Device>& devicesToUse);
uint64_t test7ProducerConsumer(uint32_t correctResult, std::vector<uint32_t>* arr, size_t size, cl::Context context, cl::CommandQueue commandQueue, std::vector<cl::Device>& devicesToUse);
uint64_t test8Coalesced(uint32_t correctResult, std::vector<uint32_t>* arr, size_t size, cl::Context context, cl::CommandQueue commandQueue, std::vector<cl::Device>& devicesToUse);

int main(int arg, char* args[])
{
    testHost();
    //SingleTest();
}

int SingleTest(void)
{
    std::vector<uint32_t>* testArray = createdArray(N_ELEMENTS);
    auto astart_time = std::chrono::steady_clock::now();
    auto aend_time = std::chrono::steady_clock::now();
    cl_int err;

    uint32_t test = 0;
    std::cout << test1SingleCoreCPU(&test, testArray, N_ELEMENTS);
    std::cout << "\t\t" << test << "\n";

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

        std::cout << "\tDevices:\tname\tCUs\twg_s\tloc_mem\tglo_mem\t\tmhz\n";
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
    uint32_t cpuSum = sumReductionCpu(testArray, N_ELEMENTS);
    std::cout << "sum: " << cpuSum << " - ";
    aend_time = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(aend_time - astart_time).count() << "mus\n";
    astart_time = std::chrono::steady_clock::now();
    std::cout << SEPARATOR;

    cl::Context context(devicesToUse);

    cl::CommandQueue commandQueue(context, deviceToUse);

    //std::unique_ptr<DATA_TYPE[]> hostGlobalInput(new DATA_TYPE[N_ELEMENTS]);
    std::unique_ptr<DATA_TYPE[]> hostGlobalOutput(new DATA_TYPE[N_ELEMENTS]);

    //std::memcpy(hostGlobalInput.get(), testArray->data(), sizeof(DATA_TYPE) * N_ELEMENTS);

    std::ifstream sourceFile(PROGRAM_SOURCE_PATH);
    std::string sourceCode(std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));
    cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length()));
    cl::Program program = cl::Program(context, source);

    program.build(devicesToUse);
    std::cout << "build log:\n" << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devicesToUse[0]);
    std::cout << SEPARATOR;
    cl::Kernel kernel (program, "reduce", &err); CHECK_ERROR(err);

    size_t sizeData = N_ELEMENTS * sizeof(DATA_TYPE);
    size_t countData = N_ELEMENTS;
    cl::Buffer kernelGlobalInput = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeData, testArray->data(), &err); CHECK_ERROR(err);
    cl::Buffer kernelGlobalOutput = cl::Buffer(context, CL_MEM_KERNEL_READ_AND_WRITE, sizeData);

    astart_time = std::chrono::steady_clock::now();
    //while(countData > 1)
    for(int i = 0; i < 2; i++)
    {
        err = kernel.setArg(0, kernelGlobalInput); CHECK_ERROR(err);
        err = kernel.setArg(1, cl::Local(LOCAL_SIZE * sizeof(DATA_TYPE))); CHECK_ERROR(err);
        err = kernel.setArg(2, sizeof(cl_int), &countData); CHECK_ERROR(err);
        err = kernel.setArg(3, kernelGlobalOutput); CHECK_ERROR(err);
        err = kernel.setArg(4, cl::Local(WORK_GROUP_COUNT*LOCAL_SIZE/2 * sizeof(DATA_TYPE))); CHECK_ERROR(err);
        err = kernel.setArg(5, cl::Local(WORK_GROUP_COUNT*LOCAL_SIZE/2 * sizeof(DATA_TYPE))); CHECK_ERROR(err);

        cl::NDRange global(LOCAL_SIZE*WORK_GROUP_COUNT);
        cl::NDRange local(LOCAL_SIZE);
        err = commandQueue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local); CHECK_ERROR(err);

        countData = WORK_GROUP_COUNT;
        sizeData = WORK_GROUP_COUNT * sizeof(DATA_TYPE);

        std::vector<DATA_TYPE> testBuffer(N_ELEMENTS);
        commandQueue.enqueueReadBuffer(kernelGlobalOutput, CL_TRUE, 0, sizeof(DATA_TYPE) * N_ELEMENTS,&testBuffer.front());
        DATA_TYPE testSum = 0;
        for(int j = 0; j < WORK_GROUP_COUNT; j++)
        {
            testSum += testBuffer[j];
        }
        std::cout << "TestSum:\t" << testSum << "\t\t" << (cpuSum == testSum) << "\n";
        commandQueue.enqueueCopyBuffer(kernelGlobalOutput, kernelGlobalInput, 0, 0, sizeData);
    }
    cl::finish();
    aend_time = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(aend_time - astart_time).count() << "mus \t";

    commandQueue.enqueueReadBuffer(kernelGlobalOutput, CL_TRUE, 0, sizeof(DATA_TYPE) * N_ELEMENTS, hostGlobalOutput.get());
    std::cout << hostGlobalOutput[0] << "\t\t" << (cpuSum == hostGlobalOutput[0]) << "\n";

    delete(testArray);
    return 0;
}

int testHost()
{
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl::Platform platformToUse;
    cl::Device deviceToUse;
    std::vector<cl::Device> devicesToUse;

    //choose device
    for(cl::Platform currentPlatform : platforms)
    {
        std::string currentPlatformName(currentPlatform.getInfo<CL_PLATFORM_NAME>());
        if(currentPlatformName == std::string(PLATFORM_TO_USE))
        {
            platformToUse = currentPlatform;
        }
        std::vector<cl::Device> devices;
        currentPlatform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
        for(cl::Device currentDevice : devices)
        {
            std::string currentDeviceName(currentDevice.getInfo<CL_DEVICE_NAME>());
            if(currentDeviceName == std::string(GPU_TO_USE))
            {
                deviceToUse = currentDevice;
            }
        }
    }
    platformToUse.getDevices(CL_DEVICE_TYPE_ALL, &devicesToUse);
    cl::Context context(devicesToUse);
    cl::CommandQueue commandQueue(context, deviceToUse);
    std::ofstream* currentFile;
    std::ofstream withoutStartup("../withoutStartup.csv");
    std::ofstream withStartup("../withStartup.csv");

    std::vector<std::ofstream> deviations(8);
    std::vector<std::ofstream> deviationsStartup(6);

    deviations[0] = std::ofstream("../singleResults/singleCPU.csv");
    deviations[1] = std::ofstream("../singleResults/multiCPU.csv");

    deviations[2] = std::ofstream("../singleResults/Dournac.csv");
    deviationsStartup[0] = std::ofstream("../singleResults/DournacStartup.csv");

    deviations[3] = std::ofstream("../singleResults/Catanzaro.csv");
    deviationsStartup[1] = std::ofstream("../singleResults/CatanzaroStartup.csv");

    deviations[4] = std::ofstream("../singleResults/Divergence.csv");
    deviationsStartup[2] = std::ofstream("../singleResults/DivergenceStartup.csv");

    deviations[5] = std::ofstream("../singleResults/Loop.csv");
    deviationsStartup[3] = std::ofstream("../singleResults/LoopStartup.csv");

    deviations[6] = std::ofstream("../singleResults/ProCon.csv");
    deviationsStartup[4] = std::ofstream("../singleResults/ProConStartup.csv");

    deviations[7] = std::ofstream("../singleResults/Coalesced.csv");
    deviationsStartup[5] = std::ofstream("../singleResults/CoalescedStartup.csv");

    currentFile = &withoutStartup;

    for(int h = 0; h < 8; h++)
    {
        deviations[h] <<"Elements, Results\n";
    }
    for(int h = 0; h < 6; h++)
    {
        deviationsStartup[h] <<"Elements, Results\n";
    }
    for(int g = 0; g < 2; g++)
    {
        if(measureSetupTime)
        {
            std::cout << "--- Measurements with startup of kernel ---\n";
        }
        std::printf("%10s|%15s|%15s|%7s|%10s|%12s|%15s|%16s|%10s|\n",
                    "Elements",
                    "SingleCore CPU",
                    "MultiCore CPU",
                    "Dournac",
                    "Catanzaro",
                    "-Divergence",
                    "Loop unrolling",
                    "ProducerConsumer",
                    "Coalesced");
        (*currentFile) <<  "Elements, SingleCore CPU, MultiCore CPU, Dournac, Catanzaro, Divergence, Loop unrolling, ProducerConsumer, Coalesced\n";

        uint64_t avg = 0;
        for (int i = 0; i < MAX_DATA_SIZE_SHIFTS; i++)
        {
            size_t elementCount = LOCAL_SIZE * WORK_GROUP_COUNT * (1 << i);
            (*currentFile) << elementCount << ", ";
            if(!measureSetupTime)
            {
                for(int h = 0; h < 8; h++)
                {
                    deviations[h] << elementCount << ", ";
                }
            }
            else
            {
                for(int h = 0; h < 6; h++)
                {
                    deviationsStartup[h] << elementCount << ", ";
                }
            }
            printf("%10llu|", elementCount);
            std::vector<uint32_t> *testArray = createdArray(elementCount);
            uint32_t correctResult = 0U;

            //1
            for (int j = 0; j < AVERAGE_OUT_OF; j++)
            {
                uint64_t temp = test1SingleCoreCPU(&correctResult, testArray, elementCount);
                avg += temp;
                if(!measureSetupTime)
                {
                    deviations[0] << temp << (j == (AVERAGE_OUT_OF - 1) ? "\n" : ", ");
                }
            }
            deviations[0] << "\n";
            printf("%15llu|", avg / AVERAGE_OUT_OF);
            (*currentFile) << avg / AVERAGE_OUT_OF << ", ";
            avg = 0;

            //2
            for (int j = 0; j < AVERAGE_OUT_OF; j++)
            {
                uint64_t temp = test2MultiCoreCPU(correctResult, testArray, elementCount);
                avg += temp;
                if(!measureSetupTime)
                {
                    deviations[1] << temp << (j == (AVERAGE_OUT_OF - 1) ? "\n" : ", ");
                }
            }
            printf("%15llu|", avg / AVERAGE_OUT_OF);
            (*currentFile) << avg / AVERAGE_OUT_OF << ", ";
            avg = 0;

            //3
            for (int j = 0; j < AVERAGE_OUT_OF; j++)
            {
                uint64_t temp = test3Dournac(correctResult, testArray, elementCount, context, commandQueue, devicesToUse);
                avg += temp;
                if(!measureSetupTime)
                {
                    deviations[2] << temp << (j == (AVERAGE_OUT_OF - 1) ? "\n" : ", ");
                }
                else
                {
                    deviationsStartup[0] << temp << (j == (AVERAGE_OUT_OF - 1) ? "\n" : ", ");
                }
            }
            printf("%7llu|", avg / AVERAGE_OUT_OF);
            (*currentFile) << avg / AVERAGE_OUT_OF << ", ";
            avg = 0;

            //4
            for (int j = 0; j < AVERAGE_OUT_OF; j++)
            {
                uint64_t temp = test4Catanzaro(correctResult, testArray, elementCount, context, commandQueue, devicesToUse);
                avg += temp;
                if(!measureSetupTime)
                {
                    deviations[3] << temp << (j == (AVERAGE_OUT_OF - 1) ? "\n" : ", ");
                }
                else
                {
                    deviationsStartup[1] << temp << (j == (AVERAGE_OUT_OF - 1) ? "\n" : ", ");
                }
            }
            printf("%10llu|", avg / AVERAGE_OUT_OF);
            (*currentFile) << avg / AVERAGE_OUT_OF << ", ";
            avg = 0;

            //5
            for (int j = 0; j < AVERAGE_OUT_OF; j++)
            {
                uint64_t temp = test5Divergence(correctResult, testArray, elementCount, context, commandQueue, devicesToUse);
                avg += temp;
                if(!measureSetupTime)
                {
                    deviations[4] << temp << (j == (AVERAGE_OUT_OF - 1) ? "\n" : ", ");
                }
                else
                {
                    deviationsStartup[2] << temp << (j == (AVERAGE_OUT_OF - 1) ? "\n" : ", ");
                }
            }
            printf("%12llu|", avg / AVERAGE_OUT_OF);
            (*currentFile) << avg / AVERAGE_OUT_OF << ", ";
            avg = 0;

            //6
            for (int j = 0; j < AVERAGE_OUT_OF; j++)
            {
                uint64_t temp = test6LoopUnrolling(correctResult, testArray, elementCount, context, commandQueue, devicesToUse);
                avg += temp;
                if(!measureSetupTime)
                {
                    deviations[5] << temp << (j == (AVERAGE_OUT_OF - 1) ? "\n" : ", ");
                }
                else
                {
                    deviationsStartup[3] << temp << (j == (AVERAGE_OUT_OF - 1) ? "\n" : ", ");
                }
            }
            printf("%15llu|", avg / AVERAGE_OUT_OF);
            (*currentFile) << avg / AVERAGE_OUT_OF << ", ";
            avg = 0;

            //7
            for (int j = 0; j < AVERAGE_OUT_OF; j++)
            {
                uint64_t temp = test7ProducerConsumer(correctResult, testArray, elementCount, context, commandQueue,
                                             devicesToUse);
                avg += temp;
                if(!measureSetupTime)
                {
                    deviations[6] << temp << (j == (AVERAGE_OUT_OF - 1) ? "\n" : ", ");
                }
                else
                {
                    deviationsStartup[4] << temp << (j == (AVERAGE_OUT_OF - 1) ? "\n" : ", ");
                }
            }
            printf("%16llu|", avg / AVERAGE_OUT_OF);
            (*currentFile) << avg / AVERAGE_OUT_OF << ", ";
            avg = 0;

            //8
            for (int j = 0; j < AVERAGE_OUT_OF; j++)
            {
                uint64_t temp = test8Coalesced(correctResult, testArray, elementCount, context, commandQueue, devicesToUse);
                avg += temp;
                if(!measureSetupTime)
                {
                    deviations[7] << temp;
                }
                else
                {
                    deviationsStartup[5] << temp;
                }
            }
            printf("%10llu|", avg / AVERAGE_OUT_OF);
            (*currentFile) << avg / AVERAGE_OUT_OF;
            avg = 0;

            (*currentFile) << "\n";
            std::cout << "\n";
            delete (testArray);
        }
        (*currentFile).close();
        measureSetupTime = 1;
        currentFile = &withStartup;
    }
    for(int h = 0; h < 8; h++)
    {
        deviations[h].close();
    }
    for(int h = 0; h < 6; h++)
    {
        deviationsStartup[h].close();
    }
    return 0;
}
uint64_t test1SingleCoreCPU(uint32_t* correctResult, std::vector<uint32_t>* arr, size_t size)
{
    auto astart_time = std::chrono::steady_clock::now();
    uint32_t sum = 0;
    for(ssize_t i = 0; i < size; i++)
    {
        sum = sum + (*arr)[i];
    }
    *correctResult = sum;
    auto aend_time = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(aend_time - astart_time).count();
}
void test2MultiCoreCPUPartialSum(const std::vector<uint32_t>& arr, size_t start, size_t end, uint32_t& result) {
    result = std::accumulate(arr.begin() + start, arr.begin() + end, 0);
}
uint64_t test2MultiCoreCPU(uint32_t correctResult, std::vector<uint32_t>* arr, size_t size)
{
    auto astart_time = std::chrono::steady_clock::now();
    // Number of threads
    const size_t num_threads = 16;
    std::vector<std::thread> threads(num_threads);
    std::vector<uint32_t> results(num_threads, 0);

    // Calculate chunk size
    size_t chunk_size = size / num_threads;

    // Launch threads
    for (size_t i = 0; i < num_threads; ++i) {
        size_t start = i * chunk_size;
        size_t end = (i == num_threads - 1) ? size : start + chunk_size;
        threads[i] = std::thread(test2MultiCoreCPUPartialSum, std::ref((*arr)), start, end, std::ref(results[i]));
    }

    // Join threads
    for (size_t i = 0; i < num_threads; ++i) {
        threads[i].join();
    }

    // Calculate final sum
    uint32_t final_sum = std::accumulate(results.begin(), results.end(), 0);
    auto aend_time = std::chrono::steady_clock::now();
    if(correctResult != final_sum){exit(-69);}
    return std::chrono::duration_cast<std::chrono::microseconds>(aend_time - astart_time).count();
}
uint64_t test3Dournac(uint32_t correctResult, std::vector<uint32_t>* arr, size_t size, cl::Context context, cl::CommandQueue commandQueue, std::vector<cl::Device>& devicesToUse)
{
    auto astart_time = std::chrono::steady_clock::now();
    
    cl_int err;
    std::vector<uint32_t> hostGlobalOutput(size);
    std::ifstream sourceFile("..//sumReduction1.cl");
    std::string sourceCode(std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));
    cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length()));
    cl::Program program = cl::Program(context, source);

    program.build(devicesToUse);
    std::cout << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devicesToUse[0]);
    cl::Kernel kernel (program, "reduce", &err); CHECK_ERROR(err);

    size_t sizeData = size * sizeof(DATA_TYPE);
    size_t countData = size;
    if(measureSetupTime)
    {
        astart_time = std::chrono::steady_clock::now();
    }
    cl::Buffer kernelGlobalInput = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeData, arr->data(), &err); CHECK_ERROR(err);
    cl::Buffer kernelGlobalOutput = cl::Buffer(context, CL_MEM_KERNEL_READ_AND_WRITE, sizeData);

    if(!measureSetupTime)
    {
        astart_time = std::chrono::steady_clock::now();
    }
    while(countData > 1)
    //for(int i = 0; i < 2; i++)
    {
        err = kernel.setArg(0, kernelGlobalInput); CHECK_ERROR(err);
        err = kernel.setArg(1, cl::Local(LOCAL_SIZE * sizeof(DATA_TYPE))); CHECK_ERROR(err);
        err = kernel.setArg(2, sizeof(cl_int), &countData); CHECK_ERROR(err);
        err = kernel.setArg(3, kernelGlobalOutput); CHECK_ERROR(err);
        //err = kernel.setArg(4, cl::Local(WORK_GROUP_COUNT*LOCAL_SIZE/2 * sizeof(DATA_TYPE))); CHECK_ERROR(err);
        //err = kernel.setArg(5, cl::Local(WORK_GROUP_COUNT*LOCAL_SIZE/2 * sizeof(DATA_TYPE))); CHECK_ERROR(err);

        cl::NDRange global(countData);
        cl::NDRange local(LOCAL_SIZE > countData ? countData : LOCAL_SIZE);
        err = commandQueue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local); CHECK_ERROR(err);

        countData = countData / LOCAL_SIZE;
        sizeData = countData * sizeof(DATA_TYPE);

        commandQueue.enqueueCopyBuffer(kernelGlobalOutput, kernelGlobalInput, 0, 0, sizeData);
    }
    cl::finish();
    auto aend_time = std::chrono::steady_clock::now();


    commandQueue.enqueueReadBuffer(kernelGlobalOutput, CL_TRUE, 0, sizeof(uint32_t) * size, hostGlobalOutput.data());

    if(correctResult != hostGlobalOutput[0]){std::cout << "!" << hostGlobalOutput[0]<< "!" << correctResult << "!" << "\n";std::exit(-69);}
    return std::chrono::duration_cast<std::chrono::microseconds>(aend_time - astart_time).count();
}
uint64_t test4Catanzaro(uint32_t correctResult, std::vector<uint32_t>* arr, size_t size, cl::Context context, cl::CommandQueue commandQueue, std::vector<cl::Device>& devicesToUse)
{
    auto astart_time = std::chrono::steady_clock::now();
    
    cl_int err;
    std::vector<uint32_t> hostGlobalOutput(size);
    std::ifstream sourceFile("..//sumReduction2.cl");
    std::string sourceCode(std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));
    cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length()));
    cl::Program program = cl::Program(context, source);

    program.build(devicesToUse);
    std::cout << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devicesToUse[0]);
    cl::Kernel kernel (program, "reduce", &err); CHECK_ERROR(err);

    size_t sizeData = size * sizeof(DATA_TYPE);
    size_t countData = size;
    if(measureSetupTime)
    {
        astart_time = std::chrono::steady_clock::now();
    }
    cl::Buffer kernelGlobalInput = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeData, arr->data(), &err); CHECK_ERROR(err);
    cl::Buffer kernelGlobalOutput = cl::Buffer(context, CL_MEM_KERNEL_READ_AND_WRITE, sizeData);

    if(!measureSetupTime)
    {
        astart_time = std::chrono::steady_clock::now();
    }
    while(countData > 1)
        //for(int i = 0; i < 2; i++)
    {
        err = kernel.setArg(0, kernelGlobalInput); CHECK_ERROR(err);
        err = kernel.setArg(1, cl::Local(LOCAL_SIZE * sizeof(DATA_TYPE))); CHECK_ERROR(err);
        err = kernel.setArg(2, sizeof(cl_int), &countData); CHECK_ERROR(err);
        err = kernel.setArg(3, kernelGlobalOutput); CHECK_ERROR(err);
        //err = kernel.setArg(4, cl::Local(WORK_GROUP_COUNT*LOCAL_SIZE/2 * sizeof(DATA_TYPE))); CHECK_ERROR(err);
        //err = kernel.setArg(5, cl::Local(WORK_GROUP_COUNT*LOCAL_SIZE/2 * sizeof(DATA_TYPE))); CHECK_ERROR(err);

        cl::NDRange global(countData);
        cl::NDRange local(LOCAL_SIZE > countData ? countData : LOCAL_SIZE);
        err = commandQueue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local); CHECK_ERROR(err);

        countData = countData / LOCAL_SIZE;
        sizeData = countData * sizeof(DATA_TYPE);

        commandQueue.enqueueCopyBuffer(kernelGlobalOutput, kernelGlobalInput, 0, 0, sizeData);
    }
    cl::finish();
    auto aend_time = std::chrono::steady_clock::now();


    commandQueue.enqueueReadBuffer(kernelGlobalOutput, CL_TRUE, 0, sizeof(uint32_t) * size, hostGlobalOutput.data());

    if(correctResult != hostGlobalOutput[0]){std::cout << "!" << hostGlobalOutput[0]<< "!" << correctResult << "!" << "\n";std::exit(-69);}
    return std::chrono::duration_cast<std::chrono::microseconds>(aend_time - astart_time).count();
}
uint64_t test5Divergence(uint32_t correctResult, std::vector<uint32_t>* arr, size_t size, cl::Context context, cl::CommandQueue commandQueue, std::vector<cl::Device>& devicesToUse)
{
    auto astart_time = std::chrono::steady_clock::now();
    
    cl_int err;
    std::vector<uint32_t> hostGlobalOutput(size);
    std::ifstream sourceFile("..//sumReduction3.cl");
    std::string sourceCode(std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));
    cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length()));
    cl::Program program = cl::Program(context, source);

    program.build(devicesToUse);
    std::cout << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devicesToUse[0]);
    cl::Kernel kernel (program, "reduce", &err); CHECK_ERROR(err);

    size_t sizeData = size * sizeof(DATA_TYPE);
    size_t countData = size;
    if(measureSetupTime)
    {
        astart_time = std::chrono::steady_clock::now();
    }
    cl::Buffer kernelGlobalInput = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeData, arr->data(), &err); CHECK_ERROR(err);
    cl::Buffer kernelGlobalOutput = cl::Buffer(context, CL_MEM_KERNEL_READ_AND_WRITE, sizeData);

    if(!measureSetupTime)
    {
        astart_time = std::chrono::steady_clock::now();
    }
    while(countData > 1)
        //for(int i = 0; i < 2; i++)
    {
        err = kernel.setArg(0, kernelGlobalInput); CHECK_ERROR(err);
        err = kernel.setArg(1, cl::Local(LOCAL_SIZE * sizeof(DATA_TYPE))); CHECK_ERROR(err);
        err = kernel.setArg(2, sizeof(cl_int), &countData); CHECK_ERROR(err);
        err = kernel.setArg(3, kernelGlobalOutput); CHECK_ERROR(err);
        //err = kernel.setArg(4, cl::Local(WORK_GROUP_COUNT*LOCAL_SIZE/2 * sizeof(DATA_TYPE))); CHECK_ERROR(err);
        //err = kernel.setArg(5, cl::Local(WORK_GROUP_COUNT*LOCAL_SIZE/2 * sizeof(DATA_TYPE))); CHECK_ERROR(err);

        cl::NDRange global(countData);
        cl::NDRange local(LOCAL_SIZE > countData ? countData : LOCAL_SIZE);
        err = commandQueue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local); CHECK_ERROR(err);

        countData = countData / LOCAL_SIZE;
        sizeData = countData * sizeof(DATA_TYPE);

        commandQueue.enqueueCopyBuffer(kernelGlobalOutput, kernelGlobalInput, 0, 0, sizeData);
    }
    cl::finish();
    auto aend_time = std::chrono::steady_clock::now();


    commandQueue.enqueueReadBuffer(kernelGlobalOutput, CL_TRUE, 0, sizeof(uint32_t) * size, hostGlobalOutput.data());

    if(correctResult != hostGlobalOutput[0]){std::cout << "!" << hostGlobalOutput[0]<< "!" << correctResult << "!" << "\n";std::exit(-69);}
    return std::chrono::duration_cast<std::chrono::microseconds>(aend_time - astart_time).count();
}
uint64_t test6LoopUnrolling(uint32_t correctResult, std::vector<uint32_t>* arr, size_t size, cl::Context context, cl::CommandQueue commandQueue, std::vector<cl::Device>& devicesToUse)
{
    auto astart_time = std::chrono::steady_clock::now();
    
    cl_int err;
    std::vector<uint32_t> hostGlobalOutput(size);
    std::ifstream sourceFile("..//sumReduction4.cl");
    std::string sourceCode(std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));
    cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length()));
    cl::Program program = cl::Program(context, source);

    program.build(devicesToUse);
    std::cout << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devicesToUse[0]);
    cl::Kernel kernel (program, "reduce", &err); CHECK_ERROR(err);

    size_t sizeData = size * sizeof(DATA_TYPE);
    size_t countData = size;
    if(measureSetupTime)
    {
        astart_time = std::chrono::steady_clock::now();
    }
    cl::Buffer kernelGlobalInput = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeData, arr->data(), &err); CHECK_ERROR(err);
    cl::Buffer kernelGlobalOutput = cl::Buffer(context, CL_MEM_KERNEL_READ_AND_WRITE, sizeData);

    if(!measureSetupTime)
    {
        astart_time = std::chrono::steady_clock::now();
    }
    //while(countData > 1)
    for(int i = 0; i < 2; i++)
    {
        err = kernel.setArg(0, kernelGlobalInput); CHECK_ERROR(err);
        err = kernel.setArg(1, cl::Local(LOCAL_SIZE * sizeof(DATA_TYPE))); CHECK_ERROR(err);
        err = kernel.setArg(2, sizeof(cl_int), &countData); CHECK_ERROR(err);
        err = kernel.setArg(3, kernelGlobalOutput); CHECK_ERROR(err);
        //err = kernel.setArg(4, cl::Local(WORK_GROUP_COUNT*LOCAL_SIZE/2 * sizeof(DATA_TYPE))); CHECK_ERROR(err);
        //err = kernel.setArg(5, cl::Local(WORK_GROUP_COUNT*LOCAL_SIZE/2 * sizeof(DATA_TYPE))); CHECK_ERROR(err);

        cl::NDRange global(LOCAL_SIZE*WORK_GROUP_COUNT);
        cl::NDRange local(LOCAL_SIZE);
        err = commandQueue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local); CHECK_ERROR(err);

        countData = WORK_GROUP_COUNT;
        sizeData = WORK_GROUP_COUNT * sizeof(DATA_TYPE);

        commandQueue.enqueueCopyBuffer(kernelGlobalOutput, kernelGlobalInput, 0, 0, sizeData);
    }
    cl::finish();
    auto aend_time = std::chrono::steady_clock::now();


    commandQueue.enqueueReadBuffer(kernelGlobalOutput, CL_TRUE, 0, sizeof(uint32_t) * size, hostGlobalOutput.data());

    if(correctResult != hostGlobalOutput[0]){std::cout << "!" << hostGlobalOutput[0]<< "!" << correctResult << "!" << "\n";std::exit(-69);}
    return std::chrono::duration_cast<std::chrono::microseconds>(aend_time - astart_time).count();
}
uint64_t test7ProducerConsumer(uint32_t correctResult, std::vector<uint32_t>* arr, size_t size, cl::Context context, cl::CommandQueue commandQueue, std::vector<cl::Device>& devicesToUse)
{
    auto astart_time = std::chrono::steady_clock::now();
    
    cl_int err;
    std::vector<uint32_t> hostGlobalOutput(size);
    std::ifstream sourceFile("..//sumReduction5.cl");
    std::string sourceCode(std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));
    cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length()));
    cl::Program program = cl::Program(context, source);

    program.build(devicesToUse);
    std::cout << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devicesToUse[0]);
    cl::Kernel kernel (program, "reduce", &err); CHECK_ERROR(err);

    size_t sizeData = size * sizeof(DATA_TYPE);
    size_t countData = size;
    if(measureSetupTime)
    {
        astart_time = std::chrono::steady_clock::now();
    }
    cl::Buffer kernelGlobalInput = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeData, arr->data(), &err); CHECK_ERROR(err);
    cl::Buffer kernelGlobalOutput = cl::Buffer(context, CL_MEM_KERNEL_READ_AND_WRITE, sizeData);

    if(!measureSetupTime)
    {
        astart_time = std::chrono::steady_clock::now();
    }
    //while(countData > 1)
    for(int i = 0; i < 2; i++)
    {
        err = kernel.setArg(0, kernelGlobalInput); CHECK_ERROR(err);
        err = kernel.setArg(1, cl::Local(LOCAL_SIZE * sizeof(DATA_TYPE))); CHECK_ERROR(err);
        err = kernel.setArg(2, sizeof(cl_int), &countData); CHECK_ERROR(err);
        err = kernel.setArg(3, kernelGlobalOutput); CHECK_ERROR(err);
        err = kernel.setArg(4, cl::Local(8*LOCAL_SIZE/2 * sizeof(DATA_TYPE))); CHECK_ERROR(err);
        err = kernel.setArg(5, cl::Local(8*LOCAL_SIZE/2 * sizeof(DATA_TYPE))); CHECK_ERROR(err);

        cl::NDRange global(LOCAL_SIZE*WORK_GROUP_COUNT);
        cl::NDRange local(LOCAL_SIZE);
        err = commandQueue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local); CHECK_ERROR(err);

        countData = WORK_GROUP_COUNT;
        sizeData = WORK_GROUP_COUNT * sizeof(DATA_TYPE);

        commandQueue.enqueueCopyBuffer(kernelGlobalOutput, kernelGlobalInput, 0, 0, sizeData);
    }
    cl::finish();
    auto aend_time = std::chrono::steady_clock::now();


    commandQueue.enqueueReadBuffer(kernelGlobalOutput, CL_TRUE, 0, sizeof(uint32_t) * size, hostGlobalOutput.data());

    if(correctResult != hostGlobalOutput[0]){std::cout << "!" << hostGlobalOutput[0]<< "!" << correctResult << "!" << "\n";std::exit(-69);}
    return std::chrono::duration_cast<std::chrono::microseconds>(aend_time - astart_time).count();
}
uint64_t test8Coalesced(uint32_t correctResult, std::vector<uint32_t>* arr, size_t size, cl::Context context, cl::CommandQueue commandQueue, std::vector<cl::Device>& devicesToUse)
{
    auto astart_time = std::chrono::steady_clock::now();
    
    cl_int err;
    std::vector<uint32_t> hostGlobalOutput(size);
    std::ifstream sourceFile("..//sumReduction6.cl");
    std::string sourceCode(std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));
    cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length()));
    cl::Program program = cl::Program(context, source);

    program.build(devicesToUse);
    std::cout << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devicesToUse[0]);
    cl::Kernel kernel (program, "reduce", &err); CHECK_ERROR(err);

    size_t sizeData = size * sizeof(DATA_TYPE);
    size_t countData = size;
    if(measureSetupTime)
    {
        astart_time = std::chrono::steady_clock::now();
    }
    cl::Buffer kernelGlobalInput = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeData, arr->data(), &err); CHECK_ERROR(err);
    cl::Buffer kernelGlobalOutput = cl::Buffer(context, CL_MEM_KERNEL_READ_AND_WRITE, sizeData);

    if(!measureSetupTime)
    {
        astart_time = std::chrono::steady_clock::now();
    }
    //while(countData > 1)
    for(int i = 0; i < 2; i++)
    {
        err = kernel.setArg(0, kernelGlobalInput); CHECK_ERROR(err);
        err = kernel.setArg(1, cl::Local(LOCAL_SIZE * sizeof(DATA_TYPE))); CHECK_ERROR(err);
        err = kernel.setArg(2, sizeof(cl_int), &countData); CHECK_ERROR(err);
        err = kernel.setArg(3, kernelGlobalOutput); CHECK_ERROR(err);
        err = kernel.setArg(4, cl::Local(8*LOCAL_SIZE/2 * sizeof(DATA_TYPE))); CHECK_ERROR(err);
        err = kernel.setArg(5, cl::Local(8*LOCAL_SIZE/2 * sizeof(DATA_TYPE))); CHECK_ERROR(err);

        cl::NDRange global(LOCAL_SIZE*WORK_GROUP_COUNT);
        cl::NDRange local(LOCAL_SIZE);
        err = commandQueue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local); CHECK_ERROR(err);

        countData = WORK_GROUP_COUNT;
        sizeData = WORK_GROUP_COUNT * sizeof(DATA_TYPE);

        commandQueue.enqueueCopyBuffer(kernelGlobalOutput, kernelGlobalInput, 0, 0, sizeData);
    }
    cl::finish();
    auto aend_time = std::chrono::steady_clock::now();


    commandQueue.enqueueReadBuffer(kernelGlobalOutput, CL_TRUE, 0, sizeof(uint32_t) * size, hostGlobalOutput.data());

    if(correctResult != hostGlobalOutput[0]){std::cout << "!" << hostGlobalOutput[0]<< "!" << correctResult << "!" << "\n";std::exit(-69);}
    return std::chrono::duration_cast<std::chrono::microseconds>(aend_time - astart_time).count();
}





uint32_t sumReductionCpu(std::vector<uint32_t>* array, uint64_t size)
{
    uint32_t sum = 0U;
    for(uint64_t i = 0; i < size; i++)
    {
        sum += (*array)[i];
    }
    return sum;
}
std::vector<uint32_t>* createdArray(uint32_t size)
{
    auto array = new std::vector<uint32_t>(size);
    std::srand(time(nullptr));
    for(uint32_t i = 0; i < size; i++)
    {
        (*array)[i] = static_cast<uint32_t>(N_ELEMENTS  + 419 - i + (static_cast<DATA_TYPE>(rand()) / RAND_MAX) * 133769);

        //(*array)[i] = N_ELEMENTS - i + 13;
        //(*array)[i] = N_ELEMENTS - i;
        //*array)[i] = 1;
    }
    return array;
}