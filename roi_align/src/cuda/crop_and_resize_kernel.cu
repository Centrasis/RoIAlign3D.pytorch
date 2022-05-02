#include <math.h>
#include <stdio.h>
#include "crop_and_resize_kernel.h"

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
     i += blockDim.x * gridDim.x)


__global__
void CropAndResizeKernel(
    const int nthreads, const float *image_ptr, const float *boxes_ptr,
    const int *box_ind_ptr, int num_boxes, int batch, int image_height,
    int image_width, int crop_height, int crop_width, int depth,
    float extrapolation_value, float *crops_ptr)
{
    CUDA_1D_KERNEL_LOOP(out_idx, nthreads)
    {
        // NHWC: out_idx = d + depth * (w + crop_width * (h + crop_height * b))
        // NCHW: out_idx = w + crop_width * (h + crop_height * (d + depth * b))
        int idx = out_idx;
        const int x = idx % crop_width;
        idx /= crop_width;
        const int y = idx % crop_height;
        idx /= crop_height;
        const int d = idx % depth;
        const int b = idx / depth;
        
        const float y1 = boxes_ptr[b * 6];
        const float x1 = boxes_ptr[b * 6 + 1];
        const float y2 = boxes_ptr[b * 6 + 2];
        const float x2 = boxes_ptr[b * 6 + 3];

        const int b_in = box_ind_ptr[b];
        if (b_in < 0 || b_in >= batch)
        {
            continue;
        }

        const float height_scale =
            (crop_height > 1) ? (y2 - y1) * (image_height - 1) / (crop_height - 1)
                                : 0;
        const float width_scale =
            (crop_width > 1) ? (x2 - x1) * (image_width - 1) / (crop_width - 1) : 0;

        const float in_y = (crop_height > 1)
                                ? y1 * (image_height - 1) + y * height_scale
                                : 0.5 * (y1 + y2) * (image_height - 1);
        if (in_y < 0 || in_y > image_height - 1)
        {
            crops_ptr[out_idx] = extrapolation_value;
            continue;
        }

        const float in_x = (crop_width > 1)
                                ? x1 * (image_width - 1) + x * width_scale
                                : 0.5 * (x1 + x2) * (image_width - 1);
        if (in_x < 0 || in_x > image_width - 1)
        {
            crops_ptr[out_idx] = extrapolation_value;
            continue;
        }

        const int top_y_index = floorf(in_y);
        const int bottom_y_index = ceilf(in_y);
        const float y_lerp = in_y - top_y_index;

        const int left_x_index = floorf(in_x);
        const int right_x_index = ceilf(in_x);
        const float x_lerp = in_x - left_x_index;

        const float *pimage = image_ptr + (b_in * depth + d) * image_height * image_width;
        const float top_left = pimage[top_y_index * image_width + left_x_index];
        const float top_right = pimage[top_y_index * image_width + right_x_index];
        const float bottom_left = pimage[bottom_y_index * image_width + left_x_index];
        const float bottom_right = pimage[bottom_y_index * image_width + right_x_index];

        const float top = top_left + (top_right - top_left) * x_lerp;
        const float bottom = bottom_left + (bottom_right - bottom_left) * x_lerp;
        crops_ptr[out_idx] = top + (bottom - top) * y_lerp;
    }
}


__global__
void CropAndResizeKernel3d(
    const int nthreads, const float *image_ptr, const float *boxes_ptr,
    const int *box_ind_ptr, int num_boxes, int batch, int image_height,
    int image_width, int image_depth, int crop_height, int crop_width, int crop_depth, int depth,
    float extrapolation_value, float *crops_ptr)
{
    CUDA_1D_KERNEL_LOOP(out_idx, nthreads)
    {
        // NHWC: out_idx = d + depth * (w + crop_width * (h + crop_height * b))
        // NCHW: out_idx = w + crop_width * (h + crop_height * (d + depth * b))
        int idx = out_idx;
        const int x = idx % crop_width;
        idx /= crop_width;
        const int y = idx % crop_height;
        idx /= crop_height;
        const int z = idx % crop_depth;
        idx /= crop_depth;
        const int d = idx % depth;
        const int b = idx / depth;
        
        const float z1 = boxes_ptr[b * 6];
        const float y1 = boxes_ptr[b * 6 + 1];
        const float x1 = boxes_ptr[b * 6 + 2];
        const float z2 = boxes_ptr[b * 6 + 3];
        const float y2 = boxes_ptr[b * 6 + 4];
        const float x2 = boxes_ptr[b * 6 + 5];

        const int b_in = box_ind_ptr[b];
        if (b_in < 0 || b_in >= batch)
        {
            continue;
        }

        const float height_scale =
            (crop_height > 1) ? (y2 - y1) * (image_height - 1) / (crop_height - 1)
                                : 0;
        const float width_scale =
            (crop_width > 1) ? (x2 - x1) * (image_width - 1) / (crop_width - 1) : 0;

        const float depth_scale =
            (crop_depth > 1) ? (z2 - z1) * (image_depth - 1) / (crop_depth - 1) : 0;

        const float in_y = (crop_height > 1)
                                ? y1 * (image_height - 1) + y * height_scale
                                : 0.5 * (y1 + y2) * (image_height - 1);
        if (in_y < 0 || in_y > image_height - 1)
        {
            crops_ptr[out_idx] = extrapolation_value;
            continue;
        }

        const float in_x = (crop_width > 1)
                                ? x1 * (image_width - 1) + x * width_scale
                                : 0.5 * (x1 + x2) * (image_width - 1);
        if (in_x < 0 || in_x > image_width - 1)
        {
            crops_ptr[out_idx] = extrapolation_value;
            continue;
        }

        const float in_z = (crop_depth > 1)
                                ? z1 * (image_depth - 1) + z * depth_scale
                                : 0.5 * (z1 + z2) * (image_depth - 1);
        if (in_z < 0 || in_z > image_depth - 1)
        {
            crops_ptr[out_idx] = extrapolation_value;
            continue;
        }

        const int top_y_index = floorf(in_y);
        const int bottom_y_index = ceilf(in_y);
        const float y_lerp = in_y - top_y_index;

        const int left_x_index = floorf(in_x);
        const int right_x_index = ceilf(in_x);
        const float x_lerp = in_x - left_x_index;

        const int back_z_index = floorf(in_z);
        const int front_z_index = ceilf(in_z);
        const float z_lerp = in_z - back_z_index;

        const float *pimage = image_ptr + (b_in * depth + d) * image_height * image_width * image_depth;
        const float top_left_back = pimage[back_z_index * image_height + top_y_index * image_width + left_x_index];
        const float top_left_front = pimage[front_z_index * image_height + top_y_index * image_width + left_x_index];
        const float top_right_back = pimage[back_z_index * image_height + top_y_index * image_width + right_x_index];
        const float top_right_front = pimage[front_z_index * image_height + top_y_index * image_width + right_x_index];
        const float bottom_left_back = pimage[back_z_index * image_height + bottom_y_index * image_width + left_x_index];
        const float bottom_left_front = pimage[front_z_index * image_height + bottom_y_index * image_width + left_x_index];
        const float bottom_right_back = pimage[back_z_index * image_height + bottom_y_index * image_width + right_x_index];
        const float bottom_right_front = pimage[front_z_index * image_height + bottom_y_index * image_width + right_x_index];

        const float top_front = top_left_front + (top_right_front - top_left_front) * x_lerp;
        const float top_back = top_left_back + (top_right_back - top_left_back) * x_lerp;
        const float bottom_front = bottom_left_front + (bottom_right_front - bottom_left_front) * x_lerp;
        const float bottom_back = bottom_left_back + (bottom_right_back - bottom_left_back) * x_lerp;
        const float front = top_front + (bottom_front - top_front) * y_lerp;
        const float back = top_back + (bottom_back - top_back) * y_lerp;
        crops_ptr[out_idx] = front + (back - front) * z_lerp;
    }
}

__global__
void CropAndResizeBackpropImageKernel(
    const int nthreads, const float *grads_ptr, const float *boxes_ptr,
    const int *box_ind_ptr, int num_boxes, int batch, int image_height,
    int image_width, int crop_height, int crop_width, int depth,
    float *grads_image_ptr)
{
    CUDA_1D_KERNEL_LOOP(out_idx, nthreads)
    {
        // NHWC: out_idx = d + depth * (w + crop_width * (h + crop_height * b))
        // NCHW: out_idx = w + crop_width * (h + crop_height * (d + depth * b))
        int idx = out_idx;
        const int x = idx % crop_width;
        idx /= crop_width;
        const int y = idx % crop_height;
        idx /= crop_height;
        const int d = idx % depth;
        const int b = idx / depth;

        const float y1 = boxes_ptr[b * 4];
        const float x1 = boxes_ptr[b * 4 + 1];
        const float y2 = boxes_ptr[b * 4 + 2];
        const float x2 = boxes_ptr[b * 4 + 3];

        const int b_in = box_ind_ptr[b];
        if (b_in < 0 || b_in >= batch)
        {
            continue;
        }

        const float height_scale =
            (crop_height > 1) ? (y2 - y1) * (image_height - 1) / (crop_height - 1)
                                : 0;
        const float width_scale =
            (crop_width > 1) ? (x2 - x1) * (image_width - 1) / (crop_width - 1) : 0;

        const float in_y = (crop_height > 1)
                                ? y1 * (image_height - 1) + y * height_scale
                                : 0.5 * (y1 + y2) * (image_height - 1);
        if (in_y < 0 || in_y > image_height - 1)
        {
            continue;
        }

        const float in_x = (crop_width > 1)
                                ? x1 * (image_width - 1) + x * width_scale
                                : 0.5 * (x1 + x2) * (image_width - 1);
        if (in_x < 0 || in_x > image_width - 1)
        {
            continue;
        }

        const int top_y_index = floorf(in_y);
        const int bottom_y_index = ceilf(in_y);
        const float y_lerp = in_y - top_y_index;

        const int left_x_index = floorf(in_x);
        const int right_x_index = ceilf(in_x);
        const float x_lerp = in_x - left_x_index;

        float *pimage = grads_image_ptr + (b_in * depth + d) * image_height * image_width;
        const float dtop = (1 - y_lerp) * grads_ptr[out_idx];
        atomicAdd(
            pimage + top_y_index * image_width + left_x_index, 
            (1 - x_lerp) * dtop
        );
        atomicAdd(
            pimage + top_y_index * image_width + right_x_index, 
            x_lerp * dtop
        );

        const float dbottom = y_lerp * grads_ptr[out_idx];
        atomicAdd(
            pimage + bottom_y_index * image_width + left_x_index, 
            (1 - x_lerp) * dbottom
        );
        atomicAdd(
            pimage + bottom_y_index * image_width + right_x_index, 
            x_lerp * dbottom
        );
    }
}


__global__
void CropAndResizeBackpropImageKernel3d(
    const int nthreads, const float *grads_ptr, const float *boxes_ptr,
    const int *box_ind_ptr, int num_boxes, int batch, int image_height,
    int image_width, int image_depth, int crop_height, int crop_width, int crop_depth, int depth,
    float *grads_image_ptr)
{
    CUDA_1D_KERNEL_LOOP(out_idx, nthreads)
    {
        // NHWC: out_idx = d + depth * (w + crop_width * (h + crop_height * b))
        // NCHW: out_idx = w + crop_width * (h + crop_height * (d + depth * b))
        int idx = out_idx;
        const int x = idx % crop_width;
        idx /= crop_width;
        const int y = idx % crop_height;
        idx /= crop_height;
        const int z = idx % crop_depth;
        idx /= crop_depth;
        const int d = idx % depth;
        const int b = idx / depth;

        const float z1 = boxes_ptr[b * 6];
        const float y1 = boxes_ptr[b * 6 + 1];
        const float x1 = boxes_ptr[b * 6 + 2];
        const float z2 = boxes_ptr[b * 6 + 3];
        const float y2 = boxes_ptr[b * 6 + 4];
        const float x2 = boxes_ptr[b * 6 + 5];

        const int b_in = box_ind_ptr[b];
        if (b_in < 0 || b_in >= batch)
        {
            continue;
        }

        const float height_scale =
            (crop_height > 1) ? (y2 - y1) * (image_height - 1) / (crop_height - 1)
                                : 0;
        const float width_scale =
            (crop_width > 1) ? (x2 - x1) * (image_width - 1) / (crop_width - 1) : 0;
        
        const float depth_scale =
            (crop_depth > 1) ? (z2 - z1) * (image_depth - 1) / (crop_depth - 1) : 0;

        const float in_y = (crop_height > 1)
                                ? y1 * (image_height - 1) + y * height_scale
                                : 0.5 * (y1 + y2) * (image_height - 1);
        if (in_y < 0 || in_y > image_height - 1)
        {
            continue;
        }

        const float in_x = (crop_width > 1)
                                ? x1 * (image_width - 1) + x * width_scale
                                : 0.5 * (x1 + x2) * (image_width - 1);
        if (in_x < 0 || in_x > image_width - 1)
        {
            continue;
        }

        const float in_z = (crop_depth > 1)
                                ? z1 * (image_depth - 1) + x * depth_scale
                                : 0.5 * (z1 + z2) * (image_depth - 1);
        if (in_z < 0 || in_z > image_depth - 1)
        {
            continue;
        }

        const int top_y_index = floorf(in_y);
        const int bottom_y_index = ceilf(in_y);
        const float y_lerp = in_y - top_y_index;

        const int left_x_index = floorf(in_x);
        const int right_x_index = ceilf(in_x);
        const float x_lerp = in_x - left_x_index;

        const int back_z_index = floorf(in_z);
        const int front_z_index = ceilf(in_z);
        const float z_lerp = in_z - back_z_index;

        float *pimage = grads_image_ptr + (b_in * depth + d) * image_height * image_width * image_depth;
        const float dtop = (1 - y_lerp) * grads_ptr[out_idx];
        atomicAdd(
            pimage + front_z_index * image_height + top_y_index * image_width + left_x_index, 
            (1 - x_lerp) * dtop * (1 - z_lerp)
        );
        atomicAdd(
            pimage + front_z_index * image_height + top_y_index * image_width + right_x_index, 
            x_lerp * dtop * (1 - z_lerp)
        );

        // for back too
        atomicAdd(
            pimage + back_z_index * image_height + top_y_index * image_width + left_x_index, 
            (1 - x_lerp) * dtop * z_lerp
        );
        atomicAdd(
            pimage + back_z_index * image_height + top_y_index * image_width + right_x_index, 
            x_lerp * dtop * z_lerp
        );

        const float dbottom = y_lerp * grads_ptr[out_idx];
        atomicAdd(
            pimage + front_z_index * image_height + bottom_y_index * image_width + left_x_index, 
            (1 - x_lerp) * dbottom * (1 - z_lerp)
        );
        atomicAdd(
            pimage + front_z_index * image_height + bottom_y_index * image_width + right_x_index, 
            x_lerp * dbottom * (1 - z_lerp)
        );

        //for back too
        atomicAdd(
            pimage + back_z_index * image_height + bottom_y_index * image_width + left_x_index, 
            (1 - x_lerp) * dbottom * z_lerp
        );
        atomicAdd(
            pimage + back_z_index * image_height + bottom_y_index * image_width + right_x_index, 
            x_lerp * dbottom * z_lerp
        );
    }
}


void CropAndResizeLaucher(
    const float *image_ptr, const float *boxes_ptr,
    const int *box_ind_ptr, int num_boxes, int batch, int image_height,
    int image_width, int crop_height, int crop_width, int depth,
    float extrapolation_value, float *crops_ptr, cudaStream_t stream)
{   
    const int total_count = num_boxes * crop_height * crop_width * depth;
    const int thread_per_block = 1024;
    const int block_count = (total_count + thread_per_block - 1) / thread_per_block;
    cudaError_t err;

    if (total_count > 0)
    {
        CropAndResizeKernel<<<block_count, thread_per_block, 0, stream>>>(
            total_count, image_ptr, boxes_ptr,
            box_ind_ptr, num_boxes, batch, image_height, image_width,
            crop_height, crop_width, depth, extrapolation_value, crops_ptr);

        err = cudaGetLastError();
        if (cudaSuccess != err)
        {
            fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
            exit(-1);
        }
    }
}


void CropAndResizeBackpropImageLaucher(
    const float *grads_ptr, const float *boxes_ptr,
    const int *box_ind_ptr, int num_boxes, int batch, int image_height,
    int image_width, int crop_height, int crop_width, int depth,
    float *grads_image_ptr, cudaStream_t stream)
{   
    const int total_count = num_boxes * crop_height * crop_width * depth;
    const int thread_per_block = 1024;
    const int block_count = (total_count + thread_per_block - 1) / thread_per_block;
    cudaError_t err;

    if (total_count > 0)
    {
        CropAndResizeBackpropImageKernel<<<block_count, thread_per_block, 0, stream>>>(
            total_count, grads_ptr, boxes_ptr,
            box_ind_ptr, num_boxes, batch, image_height, image_width,
            crop_height, crop_width, depth, grads_image_ptr);

        err = cudaGetLastError();
        if (cudaSuccess != err)
        {
            fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
            exit(-1);
        }
    }
}


void CropAndResizeLaucher3d(
    const float *image_ptr, const float *boxes_ptr,
    const int *box_ind_ptr, int num_boxes, int batch, int image_height,
    int image_width, int image_depth, int crop_height, int crop_width, int crop_depth, int depth,
    float extrapolation_value, float *crops_ptr, cudaStream_t stream)
{   
    const int total_count = num_boxes * crop_height * crop_width * crop_depth * depth;
    const int thread_per_block = 1024;
    const int block_count = (total_count + thread_per_block - 1) / thread_per_block;
    cudaError_t err;

    if (total_count > 0)
    {
        CropAndResizeKernel3d<<<block_count, thread_per_block, 0, stream>>>(
            total_count, image_ptr, boxes_ptr,
            box_ind_ptr, num_boxes, batch, image_height, image_width, image_depth,
            crop_height, crop_width, crop_depth, depth, extrapolation_value, crops_ptr);

        err = cudaGetLastError();
        if (cudaSuccess != err)
        {
            fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
            exit(-1);
        }
    }
}

void CropAndResizeBackpropImageLaucher3d(
    const float *grads_ptr, const float *boxes_ptr,
    const int *box_ind_ptr, int num_boxes, int batch, int image_height,
    int image_width, int image_depth, int crop_height, int crop_width, int crop_depth, int depth,
    float *grads_image_ptr, cudaStream_t stream)
{   
    const int total_count = num_boxes * crop_height * crop_width * crop_depth * depth;
    const int thread_per_block = 1024;
    const int block_count = (total_count + thread_per_block - 1) / thread_per_block;
    cudaError_t err;

    if (total_count > 0)
    {
        CropAndResizeBackpropImageKernel3d<<<block_count, thread_per_block, 0, stream>>>(
            total_count, grads_ptr, boxes_ptr,
            box_ind_ptr, num_boxes, batch, image_height, image_width, image_depth,
            crop_height, crop_width, crop_depth, depth, grads_image_ptr);

        err = cudaGetLastError();
        if (cudaSuccess != err)
        {
            fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
            exit(-1);
        }
    }
}