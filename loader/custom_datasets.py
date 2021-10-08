from PIL import Image
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder


class Custom_CIFAR10(CIFAR10):
    #------------------------
    #Custom CIFAR-10 dataset which returns returns 1 images, 1 target, image index
    #------------------------
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
            

        if self.target_transform is not None:
            target = self.target_transform(target)

        
        return img, target, index


class Custom_CIFAR100(CIFAR100):
    #------------------------
    #Custom CIFAR-100 dataset which returns returns 1 images, 1 target, image index
    #------------------------
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index
    

class Custom_ImageFolder(ImageFolder):
    #------------------------
    #Custom ImageFolder dataset which returns 1 images, 1 target, image index
    #------------------------
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, index
    