from discriminator import Discriminator
import torchvision.datasets as dset
from torchvision import transforms
import torch.utils.data


if __name__ == "__main__":
    saved_state = torch.load("C:\\Users\\ankit\\Workspaces\\CS7150\\FinalProject\\models\\trained_model_Mon_05_45.pth")
    dis = Discriminator(ngpu=1, num_channels=3, num_features=64)
    dis.load_state_dict(saved_state['discriminator'])

    dis.eval()

    dataset = dset.ImageFolder(root="C:\\Users\\ankit\\Workspaces\\CS7150\\data\\imagenet",
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    ]))
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    images = next(iter(dataloader))
    out = dis(images[0])

    print()


