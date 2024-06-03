__kernel void reduce(global float* buffer,
                     local float* scratch,
                     const int length,
                     global float* result)
{
    int global_index = get_global_id(0);
    float a = INFINITY;
    //Loop sequentially over chunks of input vector
    while (global_index < length)
    {
        float element = buffer [global_index];
        a = (a < element) ? a : element;
    }
    int local_index = get_local_id(0);
    scratch[local_index] = a;
    barrier(CLK_LOCAL_MEM_FENCE);
    //Perform parallel reduction
    int iGLS = get_local_size(0);
    for (int offset = iGLS/2; offset > 0; offset = offset/2)
    {
        if (local_index < offset)
        {
            float other = scratch[local_index + offset];
            float mine = scratch [local_index];
            scratch [local_index] = (mine < other) ? mine : other;
        }
        barrier (CLK_LOCAL_MEM_FENCE) ;
    }
    if (local_index == 0)
    {
        result [get_group_id(0)] = scratch [0];
    }
}