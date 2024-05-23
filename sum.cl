__kernel void vector_sum(__constant float* a, __constant float* b, __global float* c)
{
    int i = get_global_id(0);
    c[i] = a[i] + b[i];
}