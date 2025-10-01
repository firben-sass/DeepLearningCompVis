import torch 
from torchvision.io import read_image

TEMP_IMAGE_PATH = 'family.png'

def read_image_to_tensor(PATH: str)->torch.tensor:
    image_tensor = read_image(PATH)
    return image_tensor

def move_to_gpu(tensor: torch.tensor)->torch.tensor:
    try:
        if torch.cuda.is_available():
            gpu_tensor = tensor.to('cuda')
            print("Tensor moved to GPU succesfully")
            return gpu_tensor
        else:
            print("GPU not available")
    except e : 
        print(f"Error moving tensor to GPU {e}")

def save_tensor_to_disk(tensor:torch.tensor, PATH:str):
    try:
        torch.save(tensor, PATH)
        print(f'Tensor saved to: {PATH}')
    except e: 
        print(f'Error saving tensor to disk: {e}')
def main():
    print("Hello to simple script!")
    # Read an image and transfor to tensor 
    cpu_tensor = read_image_to_tensor(TEMP_IMAGE_PATH)
    
    print(f'Tensor shape: {cpu_tensor.shape}')
    print(f'Data type: {cpu_tensor.dtype}')    
    print(f'Tensor in {cpu_tensor.device}')

    # Move tensor to GPU
    gpu_tensor = move_to_gpu(cpu_tensor)
    print(f'Tensor in {gpu_tensor.device}')

    # Save tensor to disk
    save_tensor_to_disk(gpu_tensor, 'gpu_tensor.pt')

if __name__ =="__main__":
    main()