__kernel void vector_sum(__global long long int* a, __global long long int* b, __global long long int* c)
{
    int i = get_global_id(0);
    c[i] = a[i] + b[i];
}