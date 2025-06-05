import torch
import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(
    x_ptr,  # pointer to first input vector
    y_ptr,  # pointer to second input vector
    output_ptr,  # pointer to output vector
    n_elements,  # size of the vectors
    BLOCK_SIZE: tl.constexpr,  # number of elements each program should process
):
    # program ID - identifies which block we're in
    pid = tl.program_id(axis=0)
    
    # thread ID within the block - identifies which thread we are
    # This is implicitly handled by Triton, but conceptually each thread
    # in the block processes one element
    thread_id = tl.arange(0, BLOCK_SIZE)
    
    # compute the base index for this block
    block_start = pid * BLOCK_SIZE
    
    # compute the actual indices for this block's threads
    # Each thread in the block processes one element
    offsets = block_start + thread_id
    
    # create a mask to handle the case where the block size is not a multiple of n_elements
    mask = offsets < n_elements
    
    # load the input vectors
    # Each thread loads one element from each input vector
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # compute the sum
    # Each thread computes one addition
    output = x + y
    
    # store the result
    # Each thread stores one result
    tl.store(output_ptr + offsets, output, mask=mask)

def main():
    # set the size of the vectors
    n_elements = 1024
    
    # create input vectors
    x = torch.randn(n_elements, device='cuda')
    y = torch.randn(n_elements, device='cuda')
    
    # create output vector
    output = torch.empty_like(x)
    
    # set the block size - this determines how many threads are in each block
    BLOCK_SIZE = 128
    
    # compute the grid size - this determines how many blocks we need
    # For 1024 elements and block size 128, we need 8 blocks
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # launch the kernel
    # This will create 8 blocks, each with 128 threads
    # Total threads = 8 blocks * 128 threads = 1024 threads
    vector_add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE)
    
    # verify the result
    expected = x + y
    print("Max difference:", torch.max(torch.abs(output - expected)).item())
    print("All close:", torch.allclose(output, expected))

if __name__ == "__main__":
    main()
