#define UNROLLING_FACTOR 8

void load(local uint* buffer_load, uint buffer_offset, const int length, global uint* input, uint pos);
void compose(local uint* buffer_compose, uint buffer_offset, uint* accumulator);

__kernel void reduce(global uint* input,
                     local uint* localSum,
                     const int length,
                     global uint* result,
                     local uint* buffer_load,
                     local uint* buffer_compose)
{
    int global_index = get_global_id(0);
    int global_size = get_global_size(0);
    int local_index = get_local_id(0);
    int group_size = get_local_size(0);
    int group_index = get_group_id(0);
    int is_producer = (local_index % 2 == 0);

    uint buffer_offset   =is_producer * local_index * UNROLLING_FACTOR / 2
                        +(!is_producer) * (local_index - 1) * UNROLLING_FACTOR / 2;
    local uint* swap;
    uint accumulator = 0U;

    // Loop sequentially over chunks of input vector

    uint pos = global_index * UNROLLING_FACTOR / 2;
    if(is_producer)
    {
        load(buffer_load, buffer_offset, length, input, pos);
        pos += global_size * UNROLLING_FACTOR / 2;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    swap = buffer_load; buffer_load = buffer_compose; buffer_compose = swap;

    while (pos < length)
    {
        if(is_producer)
        {
            load(buffer_load, buffer_offset, length, input, pos);
        }

        if(!is_producer)
        {
            compose(buffer_compose, buffer_offset, &accumulator);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        swap = buffer_load; buffer_load = buffer_compose; buffer_compose = swap;

        pos += global_size * UNROLLING_FACTOR / 2;
    }
    //Perform parallel reduction

    localSum[((local_index - 1)/2) * (!is_producer) + is_producer * group_size*get_num_groups(0)] = accumulator;
    for (int offset = group_size/4; offset > 0; offset = offset/2)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        localSum[local_index] += (local_index < offset) * localSum[local_index + offset];
    }
    if (local_index == 0)
    {
        result [group_index] = localSum [0];
    }
}
void load(local uint* buffer_load, uint buffer_offset, const int length, global uint* input, uint pos)
{
    buffer_load[buffer_offset + 0] = (pos + 0<length) * (input[pos + 0]);
    buffer_load[buffer_offset + 1] = (pos + 1<length) * (input[pos + 1]);
    buffer_load[buffer_offset + 2] = (pos + 2<length) * (input[pos + 2]);
    buffer_load[buffer_offset + 3] = (pos + 3<length) * (input[pos + 3]);
    buffer_load[buffer_offset + 4] = (pos + 4<length) * (input[pos + 4]);
    buffer_load[buffer_offset + 5] = (pos + 5<length) * (input[pos + 5]);
    buffer_load[buffer_offset + 6] = (pos + 6<length) * (input[pos + 6]);
    buffer_load[buffer_offset + 7] = (pos + 7<length) * (input[pos + 7]);
}
void compose(local uint* buffer_compose, uint buffer_offset, uint* accumulator)
{
    *accumulator += buffer_compose[buffer_offset + 0];
    *accumulator += buffer_compose[buffer_offset + 1];
    *accumulator += buffer_compose[buffer_offset + 2];
    *accumulator += buffer_compose[buffer_offset + 3];
    *accumulator += buffer_compose[buffer_offset + 4];
    *accumulator += buffer_compose[buffer_offset + 5];
    *accumulator += buffer_compose[buffer_offset + 6];
    *accumulator += buffer_compose[buffer_offset + 7];
}