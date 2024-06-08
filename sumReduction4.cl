#define UNROLLING_FACTOR 8
__kernel void reduce(global uint* input,
                     local uint* localSum,
                     const int length,
                     global uint* result)
{
    int global_index = get_global_id(0);
    int global_size = get_global_size(0);
    int local_index = get_local_id(0);
    int group_size = get_local_size(0);

    uint accumulator = 0U;
    // Loop sequentially over chunks of input vector
    for (uint pos = global_index * UNROLLING_FACTOR; pos < length; pos += global_size * UNROLLING_FACTOR)
    {
        accumulator +=
            ((pos + 0<length) * (input[pos + 0])
            +(pos + 1<length) * (input[pos + 1])
            +(pos + 2<length) * (input[pos + 2])
            +(pos + 3<length) * (input[pos + 3])
            +(pos + 4<length) * (input[pos + 4])
            +(pos + 5<length) * (input[pos + 5])
            +(pos + 6<length) * (input[pos + 6])
            +(pos + 7<length) * (input[pos + 7]));
    }
    //Perform parallel reduction
    localSum[local_index] = accumulator;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int offset = group_size/2; offset > 0; offset = offset/2)
    {
        localSum[local_index] += (local_index < offset) * localSum[local_index + offset];
    }
    if (local_index == 0)
    {
        result [get_group_id(0)] = localSum [0];
    }
}