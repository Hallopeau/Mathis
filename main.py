import torch
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.datasets import VisionDataset
from PIL import Image
import imageio.v2 as imageio
import numpy as np
import os
import random
import h5py
import timm
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import h5py
import pickle
from UCroma import PretrainedCROMA
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import jaccard_score, classification_report, confusion_matrix
import statistics

class Loader:
    def __init__(self, root_dir, num_folds=5):
        """
        Initializes the Loader class.

        Parameters:
        root_dir (str): Directory with all the images, organized by class.
        num_folds (int): Number of folds to split the data into.
        """
        self.root_dir = root_dir
        self.num_folds = num_folds
        self.loaders = []

    def load_data(self, sample_size=224, batch_size=32, data_seed=None, split_seed=None):
        """
        Loads the data, applies transformations, and splits it into folds.

        Parameters:
        sample_size (int): Size to resize the images to (sample_size x sample_size).
        batch_size (int): Number of images per batch.
        data_seed (int, optional): Seed for random sampling of images.
        split_seed (int, optional): Seed for random splitting into folds.
        """
        class CustomDataset(VisionDataset):
            def __init__(self, root, transform=None, target_transform=None):
                """
                Custom dataset for loading images.

                Parameters:
                root (str): Root directory of the dataset.
                transform (callable, optional): Optional transform to be applied on a sample.
                target_transform (callable, optional): Optional transform to be applied on a target.
                """
                super(CustomDataset, self).__init__(root, transform=transform, target_transform=target_transform)
                self.root = root
                self.transform = transform
                self.target_transform = target_transform
                self.classes = sorted(os.listdir(root))
                
                min_num_images = float('inf')
                
                for class_name in self.classes:
                    class_path = os.path.join(root, class_name)
                    if os.path.isdir(class_path):
                        num_images = len(os.listdir(class_path))
                        min_num_images = min(min_num_images, num_images)

                self.samples = []
                        
                for class_name in self.classes:
                    class_path = os.path.join(root, class_name)
                    if os.path.isdir(class_path):
                        images = [img_name for img_name in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, img_name))]
                        if data_seed is not None:
                            random.seed(data_seed)
                        selected_images = random.sample(images, min_num_images)
                        for img_name in selected_images:
                            img_path = os.path.join(class_path, img_name)
                            self.samples.append((img_path, self.classes.index(class_name)))

            def __len__(self):
                """
                Returns the total number of samples.
                """
                return len(self.samples)

            def __getitem__(self, idx):
                """
                Fetches the sample and target at the given index.

                Parameters:
                idx (int): Index of the sample to fetch.

                Returns:
                tuple: (image, target, image_path)
                """
                img_path, target = self.samples[idx]
                img = imageio.imread(img_path)[:, :, :3]
                img = (img * 255).astype(np.uint8)
                img = Image.fromarray(img)
                if self.transform:
                    img = self.transform(img)
                if self.target_transform:
                    target = self.target_transform(target)
                return img, target, img_path

        transform = transforms.Compose([
            transforms.Resize((sample_size, sample_size)),
            transforms.ToTensor()
        ])

        dataset = CustomDataset(root=self.root_dir, transform=transform)
        self.dataset = dataset

        fold_sizes = [len(dataset) // self.num_folds] * self.num_folds
        remainder = len(dataset) % self.num_folds
        for i in range(remainder):
            fold_sizes[i] += 1

        if split_seed is not None:
            torch.manual_seed(split_seed)
        folds = random_split(dataset, fold_sizes)

        loaders = []
        for f in folds: 
            loaders.append(DataLoader(f, batch_size=batch_size, shuffle=True))
        self.loaders = loaders

class CRLoader:
    """
    A class to load optical and SAR image data, apply transformations, and create data loaders.

    Attributes:
        opt_root_dir: Root directory containing optical images.
        sar_root_dir: Root directory containing SAR images.
        num_folds: Number of folds for cross-validation.
        loaders: List of DataLoader objects for each fold.
        sar_loaders: List of DataLoader objects for SAR images (not used in this code).
    """

    def __init__(self, opt_root_dir, sar_root_dir, num_folds=5):
        """
        Initializes the Loader with the given parameters.

        Args:
            opt_root_dir: Root directory containing optical images.
            sar_root_dir: Root directory containing SAR images.
            num_folds: Number of folds for cross-validation.
        """
        self.opt_root_dir = opt_root_dir
        self.sar_root_dir = sar_root_dir
        self.num_folds = num_folds
        self.loaders = []
        self.sar_loaders = []

    def load_data(self, sample_size=224, batch_size=32, data_seed=None, split_seed=None):
        """
        Loads data, applies transformations, and creates data loaders.

        Args:
            sample_size: Size to which each image will be resized.
            batch_size: Number of samples per batch.
            data_seed: Seed for random sampling of images.
            split_seed: Seed for splitting the dataset into folds.

        Returns:
            List of DataLoader objects for each fold.
        """
        class CustomDataset(VisionDataset):
            """
            Custom dataset class for loading optical and SAR images.
            """

            def __init__(self, opt_root, sar_root, transform=None, target_transform=None):
                """
                Initializes the CustomDataset with the given parameters.

                Args:
                    opt_root: Root directory containing optical images.
                    sar_root: Root directory containing SAR images.
                    transform: Transformations to be applied to the images.
                    target_transform: Transformations to be applied to the targets.
                """
                super(CustomDataset, self).__init__(root=opt_root, transform=transform, target_transform=target_transform)
                self.opt_root = opt_root
                self.sar_root = sar_root
                self.transform = transform
                self.target_transform = target_transform

                self.opt_classes = sorted(os.listdir(opt_root))
                self.sar_classes = sorted(os.listdir(sar_root))

                assert self.opt_classes == self.sar_classes, "Optical and SAR classes do not match."
                self.classes = self.opt_classes
                
                min_num_images = float('inf')
                for class_name in self.classes:
                    class_path = os.path.join(opt_root, class_name)
                    if os.path.isdir(class_path):
                        num_images = len(os.listdir(class_path))
                        min_num_images = min(min_num_images, num_images)

                self.samples = []
                for class_name in self.opt_classes:
                    opt_class_path = os.path.join(opt_root, class_name)
                    sar_class_path = os.path.join(sar_root, class_name)

                    if os.path.isdir(opt_class_path) and os.path.isdir(sar_class_path):
                        opt_images = [img_name for img_name in os.listdir(opt_class_path) if os.path.isfile(os.path.join(opt_class_path, img_name))]
                        sar_images = [img_name for img_name in os.listdir(sar_class_path) if os.path.isfile(os.path.join(sar_class_path, img_name))]
                        
                        if data_seed is not None:
                            random.seed(data_seed)
                        selected_images = random.sample(opt_images, min_num_images)
                        for img_name in selected_images:
                            opt_img_path = os.path.join(opt_class_path, img_name)
                            sar_img_path = os.path.join(sar_class_path, img_name)
                            self.samples.append((opt_img_path, sar_img_path, self.opt_classes.index(class_name)))

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                opt_img_path, sar_img_path, target = self.samples[idx]

                opt_img = imageio.imread(opt_img_path)
                sar_img = imageio.imread(sar_img_path)

                opt_img = np.transpose(opt_img, (2, 0, 1)).astype(np.float32)
                sar_img = np.transpose(sar_img, (2, 0, 1)).astype(np.float32)

                opt_img = torch.tensor(opt_img)
                sar_img = torch.tensor(sar_img)

                if self.transform:
                    opt_img = self.transform(opt_img)
                    sar_img = self.transform(sar_img)

                if self.target_transform:
                    target = self.target_transform(target)

                return opt_img, sar_img, target, opt_img_path, sar_img_path

        transform = transforms.Resize((sample_size, sample_size), antialias=True)
        combined_dataset = CustomDataset(opt_root=self.opt_root_dir, sar_root=self.sar_root_dir, transform=transform)
        self.combined_dataset = combined_dataset

        fold_sizes = [len(combined_dataset) // self.num_folds] * self.num_folds
        remainder = len(combined_dataset) % self.num_folds
        for i in range(remainder):
            fold_sizes[i] += 1

        if split_seed is not None:
            torch.manual_seed(split_seed)
        folds = random_split(combined_dataset, fold_sizes)

        loaders = []
        for fold in folds:
            loaders.append(DataLoader(fold, batch_size=batch_size, shuffle=True))
        self.loaders = loaders

class FeatureExtractor:
    def __init__(self, model_name):
        """
        Initializes the FeatureExtractor class.

        Parameters:
        model_name (str): Name of the pretrained model from the timm library.
        """
        self.model_name = model_name
        if model_name not in timm.list_models(pretrained=True):
            print("Error, '%s' is not a valid model name for timm library. For a list of available pretrained models, "
                  "run: \n'timm.list_models(pretrained=True)'" % model_name)
            return
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.model.to(self.device)
        self.model.eval()

        data_cfg = timm.data.resolve_data_config(self.model.pretrained_cfg)
        self.model_transform = timm.data.create_transform(**data_cfg)

        self.file_name = None

    def get_features(self, dataloader, name=None):
        """
        Extracts features from the data using the pretrained model.

        Parameters:
        dataloader (DataLoader): DataLoader containing the data to extract features from.
        name (str, optional): Name to use for saving the features file. If None, features are not saved to a file.

        Returns:
        tuple: (features, labels) where features and labels are numpy arrays.
        """
        if name is not None:
            self.file_name = "features_%s_%s.h5" % (name, self.model_name)
            if os.path.exists(self.file_name):
                features, labels = self._load_features(self.file_name)
            else:
                features, labels = self._extract_features(dataloader)
                self._save_features(features, labels, self.file_name)
        else:
            features, labels = self._extract_features(dataloader)

        return features, labels

    def _extract_features(self, dataloader):
        """
        Extracts features from the data using the pretrained model.

        Parameters:
        dataloader (DataLoader): DataLoader containing the data to extract features from.

        Returns:
        tuple: (features, labels) where features and labels are tensors.
        """
        features, labels = [], []
        with torch.no_grad():
            for inputs, labs, _ in tqdm(dataloader, desc="Extracting Features"):
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                features.append(outputs.cpu())
                labels.append(labs)
        features = torch.cat(features).numpy()
        labels = torch.cat(labels).numpy()

        return features, labels

    @staticmethod
    def _save_features(features, labels, file_name):
        """
        Saves extracted features to an HDF5 file.

        Parameters:
        features (numpy.ndarray): Array of extracted features.
        labels (numpy.ndarray): Array of labels corresponding to the features.
        file_name (str): Name of the file to save the features and labels.
        """
        with h5py.File(file_name, 'w') as hf:
            hf.create_dataset('features', data=features)
            hf.create_dataset('labels', data=labels)

    @staticmethod
    def _load_features(file_name):
        """
        Loads features and labels from an HDF5 file.

        Parameters:
        file_name (str): Name of the file to load the features and labels from.

        Returns:
        tuple: (features, labels) where features and labels are numpy arrays.
        """
        with h5py.File(file_name, 'r') as hf:
            features = np.array(hf.get("features"))
            labels = np.array(hf.get("labels"))

        return features, labels

class FExtractor:
    """
    A class to extract features from a dataset using the pretrained CROMA model.

    Attributes:
        dataloader: DataLoader for the dataset.
        use_8_bit: Flag to determine if 8-bit normalization should be used.
        device: Device on which to run the model (CPU or GPU).
        FE: Pretrained CROMA feature extractor model.
    """

    def __init__(self, dataloader, use_8_bit=True):
        """
        Initializes the FExtractor with the given parameters.

        Args:
            dataloader: DataLoader for the dataset.
            use_8_bit: Flag to determine if 8-bit normalization should be used. Defaults to True.
        """
        self.dataloader = dataloader
        self.use_8_bit = use_8_bit
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.FE = PretrainedCROMA(pretrained_path='CR.pt', size='base', modality='both', image_resolution=120)
        self.FE.to(self.device)
        self.FE.eval()

    def normalize(self, x):
        """
        Normalizes the input images.

        Args:
            x: Input tensor of images.

        Returns:
            Normalized tensor of images.
        """
        x = x.float()
        imgs = []
        for channel in range(x.shape[1]):
            min_value = x[:, channel, :, :].mean() - 2 * x[:, channel, :, :].std()
            max_value = x[:, channel, :, :].mean() + 2 * x[:, channel, :, :].std()
            if self.use_8_bit:
                img = (x[:, channel, :, :] - min_value) / (max_value - min_value) * 255.0
                img = torch.clip(img, 0, 255).unsqueeze(dim=1).to(torch.uint8)
                imgs.append(img)
            else:
                img = (x[:, channel, :, :] - min_value) / (max_value - min_value)
                img = torch.clip(img, 0, 1).unsqueeze(dim=1)
                imgs.append(img)
        return torch.cat(imgs, dim=1)

    def extract_features(self, save_name=None):
        """
        Extracts features from the dataset and optionally saves them to a file.

        Args:
            save_name: Optional; Name of the file to save the features and labels.

        Returns:
            Tuple containing features and labels.
        """
        features_batches = []
        labels_batches = []

        with torch.no_grad():
            for optical_images, sar_images, labels, _, _ in tqdm(self.dataloader, desc="Extracting Features"):
                optical_images = optical_images.to(self.device)
                optical_images = self.normalize(optical_images)
                sar_images = sar_images.to(self.device)
                sar_images = self.normalize(sar_images)
                if self.use_8_bit:
                    optical_images = optical_images.float() / 255
                    sar_images = sar_images.float() / 255
                outputs = self.FE(SAR_images=sar_images, optical_images=optical_images)['joint_GAP']
                features_batches.append(outputs.cpu())
                labels_batches.append(labels)

        features = torch.cat(features_batches).numpy()
        labels = torch.cat(labels_batches).numpy()

        if save_name:
            with h5py.File(f'{save_name}.h5', 'w') as hf:
                hf.create_dataset('features', data=features)
                hf.create_dataset('labels', data=labels)

            with open(f'{save_name}.pkl', 'wb') as f:
                pickle.dump([features, labels], f)

        return features, labels

class FExtractorB:
    def __init__(self, dataloader, use_8_bit=True):
        self.dataloader = dataloader
        self.use_8_bit = use_8_bit
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.FE = PretrainedCROMA(pretrained_path='CR.pt', size='base', modality='both', image_resolution=120)
        self.FE.to(self.device)
        self.FE.eval()

    def normalize(self, x):
        x = x.float()
        imgs = []
        for channel in range(x.shape[1]):
            min_value = x[:, channel, :, :].mean() - 2 * x[:, channel, :, :].std()
            max_value = x[:, channel, :, :].mean() + 2 * x[:, channel, :, :].std()
            if self.use_8_bit:
                img = (x[:, channel, :, :] - min_value) / (max_value - min_value) * 255.0
                img = torch.clip(img, 0, 255).unsqueeze(dim=1).to(torch.uint8)
                imgs.append(img)
            else:
                img = (x[:, channel, :, :] - min_value) / (max_value - min_value)
                img = torch.clip(img, 0, 1).unsqueeze(dim=1)
                imgs.append(img)
        return torch.cat(imgs, dim=1)

    def extract_features(self, save_name=None):
        features_batches = []
        labels_batches = []
        id_batches = []
        with torch.no_grad():
            for optical_images, sar_images, labels, optical_img_paths, _ in tqdm(self.dataloader, desc="Extracting Features"):
                optical_images = optical_images.to(self.device)
                optical_images = self.normalize(optical_images)
                sar_images = sar_images.to(self.device)
                sar_images = self.normalize(sar_images)
                if self.use_8_bit:
                    optical_images = optical_images.float() / 255
                    sar_images = sar_images.float() / 255
                outputs = self.FE(SAR_images=sar_images, optical_images=optical_images)['joint_GAP']
                features_batches.append(outputs.cpu())
                labels_batches.append(labels)
                ids_i = torch.tensor([int((path.split(".")[0]).split("/")[-1]) for path in optical_img_paths])
                id_batches.append(ids_i)

        features = torch.cat(features_batches).numpy()
        labels = torch.cat(labels_batches).numpy()
        ids = torch.cat(id_batches).numpy()

        if save_name:
            hf = h5py.File(f'{save_name}.h5', 'w')
            hf.create_dataset('features', data=features)
            hf.create_dataset('labels', data=labels)
            hf.create_dataset('ids', data=ids)
            hf.close()

            with open(f'{save_name}.pkl', 'wb') as f:
                pickle.dump([features, labels, ids], f)

        return features, labels, ids

class Trainer:
    def __init__(self, model, train_loader, criterion=nn.CrossEntropyLoss(), optimizer=None, lr=0.001, output_file_name=None):
        """
        Initializes the Trainer class.

        Parameters:
        model (nn.Module): The neural network model to train.
        train_loader (DataLoader): DataLoader for the training dataset.
        criterion (nn.Module): Loss function. Default is CrossEntropyLoss.
        optimizer (torch.optim.Optimizer, optional): Optimizer for training. Default is Adam.
        lr (float): Learning rate for the optimizer. Default is 0.001.
        output_file_name (str, optional): Filename to save the model after training.
        """
        self.model = model
        self.train_loader = train_loader
        self.criterion = criterion
        self.optimizer = optimizer or Adam(model.parameters(), lr=lr)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_file_name = output_file_name

    def train(self, num_epochs=30):
        """
        Trains the model for a specified number of epochs.

        Parameters:
        num_epochs (int): Number of epochs to train the model. Default is 30.
        """
        self.model.to(self.device)
        self.model.train()  # Set the model to training mode
        
        for epoch in tqdm(range(num_epochs), desc="Training", unit=" epochs"):
            total_loss = 0.0
            for images, labels in self.train_loader:
                self.optimizer.zero_grad()  # Reset previously stored gradients
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)  # Make predictions on the batch
                loss = self.criterion(outputs, labels)  # Compute the loss
                loss.backward()  # Compute gradients of the loss
                self.optimizer.step()  # Update the model weights
                total_loss += loss.item() * images.size(0)

        if self.output_file_name is not None:
            torch.save(self.model, f"{self.output_file_name}_{num_epochs}.pth")

class TrainerB:
    def __init__(self, model, train_loader, criterion=nn.CrossEntropyLoss(), optimizer=None, lr=0.001, output_file_name=None):
        self.model = model
        self.train_loader = train_loader
        self.criterion = criterion
        self.optimizer = optimizer or Adam(model.parameters(), lr=lr)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_file_name = output_file_name

    def train(self, num_epochs=30):
        self.model.to(self.device)
        self.model.train() 
        for epoch in tqdm(range(num_epochs),   desc="Training", unit=" epochs"):
            total_loss = 0.0
            for images, labels, _ in self.train_loader:
                self.optimizer.zero_grad()  
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images) 
                loss = self.criterion(outputs, labels) 
                loss.backward()  
                self.optimizer.step() 
                total_loss += loss.item() * images.size(0)
        if self.output_file_name != None:
            torch.save(self.model, f"{self.output_file_name}_{num_epochs}.pth")

class Tester:
    def __init__(self, model, test_loader, dataset):
        """
        Initializes the Tester class.

        Parameters:
        model (nn.Module): The trained neural network model to evaluate.
        test_loader (DataLoader): DataLoader for the test dataset.
        dataset (Dataset): The dataset containing class information.
        """
        self.model = model
        self.test_loader = test_loader
        self.dataset = dataset
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def evaluate(self):
        """
        Evaluates the model on the test data and prints evaluation metrics.
        """
        self.model.eval()  # Set the model to evaluation mode
        
        df = pd.DataFrame(columns=['target', 'output'])

        with torch.no_grad():
            for j, (images, labels) in enumerate(self.test_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)

                for i in range(outputs.shape[0]): 
                    label_i = labels[i]
                    output_i = outputs[i]

                    id = int(j * 10000 + i)
                    
                    _, predicted = torch.max(output_i.data, 0)

                    if label_i == self.dataset.classes.index("favelas"):
                        df.loc[id, "target"] = 1
                    if label_i == self.dataset.classes.index("residential"):
                        df.loc[id, "target"] = 0
                    
                    if predicted == self.dataset.classes.index("favelas"):
                        df.loc[id, "output"] = 1
                    if predicted == self.dataset.classes.index("residential"):
                        df.loc[id, "output"] = 0
    
        target_array = df['target'].to_numpy().astype(np.int64)
        output_array = df['output'].to_numpy().astype(np.int64)

        IoU = jaccard_score(target_array, output_array)
        cm = confusion_matrix(target_array, output_array)
        df_cm = pd.DataFrame(cm, index=['Actual Class 0', 'Actual Class 1'], columns=['Predicted Class 0', 'Predicted Class 1'])
        cr = classification_report(target_array, output_array)

        self.report = [cm, cr, IoU]

        print(f"\nJaccard index: {IoU * 100: 0.1f}%\n")
        print(f"\n{df_cm}\n")
        print(f"\n{cr}\n")

class TesterB:
    def __init__(self, model, test_loader, dataset):
        self.model = model
        self.test_loader = test_loader
        self.dataset = dataset
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def evaluate(self, global_list):

        self.model.eval()
        
        df = pd.DataFrame(columns=['target', 'output'])

        with torch.no_grad():
            for images, labels, ids in self.test_loader:
                
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)

                ids = ids.tolist()

                for i in range(outputs.shape[0]): 

                    label_i = labels[i]
                    output_i = outputs[i]

                    id_i = str(ids[i])
                    
                    _, predicted = torch.max(output_i.data, 0)

                    if label_i == self.dataset.classes.index("favelas"):
                        df.loc[id_i, "target"] = 1
                    if label_i == self.dataset.classes.index("residential"):
                        df.loc[id_i, "target"] = 0
                    
                    if predicted == self.dataset.classes.index("favelas"):
                        df.loc[id_i, "output"] = 1
                        global_list.loc[id_i, "List"].append(1)
                    if predicted == self.dataset.classes.index("residential"):
                        df.loc[id_i, "output"] = 0
                        global_list.loc[id_i, "List"].append(0)
    
        target_array = df['target'].to_numpy().astype(np.int64)
        output_array = df['output'].to_numpy().astype(np.int64)

        IoU = jaccard_score(target_array, output_array)
        cm = confusion_matrix(target_array, output_array)
        df_cm = pd.DataFrame(cm, index=['Actual Class 0', 'Actual Class 1'], columns=['Predicted Class 0', 'Predicted Class 1'])
        cr = classification_report(target_array, output_array)

        self.report = [cm, cr, IoU]

        print(f"\nJaccard index: {IoU*100: 0.1f}%\n")
        print(f"\n{df_cm}\n")
        print(f"\n{cr}\n")

class RGenerator:
    """
    A class to generate a report based on evaluation metrics.

    Attributes:
        metrics: List of evaluation metrics.
    """
    def __init__(self, metrics):
        """
        Initializes the RGenerator with the given metrics.

        Args:
            metrics: List of evaluation metrics.
        """
        self.metrics = metrics

    def report(self):
        """
        Generates and prints a report based on the evaluation metrics.

        Returns:
            A list containing mean and standard deviation of precision, recall, F1-score, and IoU.
        """
        precision = []
        recall = []
        f1_score = []
        iou = []
        cm = []

        for metric in self.metrics:
            cr = metric[1]
            cr = cr.encode().decode('unicode_escape')
            lines = cr.strip().split('\n')
            class_1_line = lines[3]
            class_1_values = class_1_line.split()[1:4]

            precision.append(float(class_1_values[0]))
            recall.append(float(class_1_values[1]))
            f1_score.append(float(class_1_values[2]))

            iou.append(metric[2])
            cm.append(metric[0])

        report = [
            [statistics.mean(precision), statistics.pstdev(precision)],
            [statistics.mean(recall), statistics.pstdev(recall)],
            [statistics.mean(f1_score), statistics.pstdev(f1_score)],
            [statistics.mean(iou), statistics.pstdev(iou)]
        ]

        print(f"Precision (mean, stdev): {report[0][0]*100:.0f}%, {report[0][1]*100:.0f}%")
        print(f"Recall (mean, stdev): {report[1][0]*100:.0f}%, {report[1][1]*100:.0f}%")
        print(f"F1-score (mean, stdev): {report[2][0]*100:.0f}%, {report[2][1]*100:.0f}%")
        print(f"IoU (mean, stdev): {report[3][0]*100:.0f}%, {report[3][1]*100:.0f}%")

        mean_cm = np.mean(np.array(cm), axis=0)
        print(f"\n{np.round(mean_cm, decimals=0)}")
        std_cm = np.std(np.array(cm), axis=0)
        print(f"\n{np.round(std_cm, decimals=0)}\n")

        return report
        