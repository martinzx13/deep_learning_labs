import torch
import torch.nn as nn 
import os 

def check_system():
    command = os.system("lspci | grep -i vga")
    print(torch.version.cuda)
    return;

def main():
    check_system()

if __name__ == "__main__":
    main()
