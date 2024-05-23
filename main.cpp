#include <iostream>
#include <CL/cl.hpp>
#include <cassert>
#include <fstream>

#define GPU_TO_USE "gfx1032"
#define PLATFORM_TO_USE "AMD Accelerated Parallel Processing"
#define PROGRAM_SOURCE_PATH "../sum.cl"
#define SEPARATOR "--------------------------------------------\n"

int main(int arg, char* args[])
{
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    std::cout << "Platforms:\n";
    cl::Device deviceToUse;
    for(cl::Platform currentPlatform : platforms)
    {
        std::string currentPlatformName(currentPlatform.getInfo<CL_PLATFORM_NAME>());
        if(currentPlatformName == std::string(PLATFORM_TO_USE))
        {
            std::cout << "-->";
        }
        std::cout << "\t" << currentPlatformName << "\n";

        std::vector<cl::Device> devices;
        currentPlatform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

        std::cout << "\tDevices:\tname\tunits\tmhz\n";
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
                            "\t" << currentDevice.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() << "\n";
        }
    }
    std::cout << SEPARATOR;

    return 0;
}
