__kernel void reduce(   global uint* input,
                        local uint* localSum,
                        const int length,
                        global uint* result)
{
    int local_index = get_local_id(0);
    localSum[local_index] = input[get_global_id(0)];

    int group_size = get_local_size(0);
    for (int offset = group_size/2; offset > 0; offset = offset/2)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (local_index < offset)
        {
            localSum[local_index] += localSum[local_index + offset];
        }
    }
    if (local_index == 0)
    {
        result [get_group_id(0)] = localSum [0];
    }
}