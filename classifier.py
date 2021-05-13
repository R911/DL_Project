import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
from discriminator import Discriminator
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils
import sklearn.svm as svm
import pickle
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import datetime


class SVM_Classifier:
    def __init__(self, batch_size, image_size=64):
        self.image_size = image_size
        self.device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        self.save_filename = f'model_{datetime.datetime.now().strftime("%a_%H_%M")}.sav'

        transform = transforms.Compose(
            [
                # transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                     download=True, transform=transform)
        self.trainloader = data.DataLoader(self.trainset, batch_size=batch_size,
                                           shuffle=True, num_workers=2)

        self.testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                    download=True, transform=transform)
        self.testloader = data.DataLoader(self.testset, batch_size=batch_size,
                                          shuffle=False, num_workers=2)

        saved_state = torch.load(
            "C:\\Users\\ankit\\Workspaces\\CS7150\\FinalProject\\models\\imagenet\\trained_model_Tue_17_06.pth")
        self.discriminator = Discriminator(ngpu=1, num_channels=3, num_features=64, data_generation_mode=1,
                                           input_size=image_size)
        self.discriminator.load_state_dict(saved_state['discriminator'])
        self.discriminator.eval()  # change the mode of the network.

    def plot_training_data(self):
        # Plot some training images
        real_batch = next(iter(self.trainloader))
        real_batch = real_batch[0][0:8]
        plt.figure(figsize=(8, 8))
        plt.axis("off")
        plt.title("Training Images")
        plt.imshow(
            np.transpose(vutils.make_grid(real_batch[0].to(self.device)[:64], padding=2, normalize=True).cpu(),
                         (1, 2, 0)))
        plt.show()

    def train(self):
        train_data, train_labels = next(iter(self.trainloader))
        modified_train_data = self.discriminator(train_data)
        l2_svm = svm.LinearSVC(verbose=2, max_iter=2000)

        modified_train_data_ndarray = modified_train_data.detach().numpy()
        train_labels_ndarray = train_labels.detach().numpy()
        self.l2_svm = l2_svm.fit(modified_train_data_ndarray, train_labels_ndarray)

        # save model
        with open(self.save_filename, 'wb') as file:
            pickle.dump(self.l2_svm, file)

    def train_test_SGD_Classifier(self):
        est = make_pipeline(StandardScaler(), SGDClassifier(max_iter=200))
        training_data = self.discriminator(next(iter(self.trainloader))[0])
        training_data = training_data.detach().numpy()
        est.steps[0][1].fit(training_data)

        self.est = est

        for i, data in enumerate(self.trainloader):
            train_data, train_labels = data
            modified_train_data = self.discriminator(train_data)

            modified_train_data_ndarray = modified_train_data.detach().numpy()
            train_labels_ndarray = train_labels.detach().numpy()
            modified_train_data_ndarray = est.steps[0][1].transform(modified_train_data_ndarray)

            est.steps[1][1].partial_fit(modified_train_data_ndarray, train_labels_ndarray,
                                        classes=np.unique(train_labels_ndarray))
            print(f'Batch: {i}')

        with open(self.save_filename, 'wb') as file:
            pickle.dump(est.steps[1][1], file)

    def test(self):
        l2_svm = self.est.steps[1][1]
        accuracy = []

        for i, data in enumerate(self.testloader):
            test_data, test_labels = data
            modified_test_data = self.discriminator(test_data)

            modified_test_data_ndarray = modified_test_data.detach().numpy()
            test_labels_ndarray = test_labels.detach().numpy()
            modified_test_data_ndarray = self.est.steps[0][1].transform(modified_test_data_ndarray)

            predictions = l2_svm.predict(modified_test_data_ndarray)

            accuracy.append(metrics.accuracy_score(test_labels_ndarray, predictions))

        print(f'Accuracy: {np.mean(accuracy)}')


if __name__ == "__main__":
    classifier = SVM_Classifier(batch_size=1024, image_size=32)
    # classifier.plot_training_data()
    # classifier.train()
    classifier.train_test_SGD_Classifier()
    classifier.test()
