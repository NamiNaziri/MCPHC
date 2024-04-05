import os
import sys
sys.path.append(os.path.abspath('./src'))

from argparse import ArgumentParser
import ast
import inspect
import numpy as np
import torch.distributions as td
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pickle
from vispy import scene

from isaacgymenvs.utils.visual_data import VisualDataContainer
import debugpy


# # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
# debugpy.listen(5678)
# print("Waiting for debugger attach")
# debugpy.wait_for_client()
# print('break on this line')

class Sourceable:

    def get_src(self, depth):
        my_src = f"global {self.__class__.__name__}\n" + inspect.getsource(self.__class__)
        srcs = []
        if depth == 0:
            srcs.append("global Sourceable\n" + inspect.getsource(Sourceable))
        for child_module in self._modules:
            child_module_inst = getattr(self, child_module)
            if isinstance(child_module_inst, Sourceable):
                srcs.append(child_module_inst.get_src(depth + 1))
        srcs.append(my_src)
        return "\n".join(srcs)

class MLPAE(nn.Module, Sourceable):

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance.args = args
        instance.kwargs = kwargs
        return instance

    def __init__(self, nail_size: int, hidden_size: int, hammer_size: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(nail_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, hammer_size),
            nn.Tanh()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hammer_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, nail_size)
        )

    def forward(self, x: torch.Tensor, train_yes: bool) -> (torch.Tensor, torch.Tensor):
        """
        Returns two tensors: encoded and reconstructed
        """
        # x is shape (B, nail_size)
        z0 = self.encoder(x)
        z1 = self.decoder(z0)
        return z0, z1




# Function to extract import statements from a given AST node
def extract_imports(node):
    imports = []
    for item in node.body:
        if isinstance(item, ast.Import):
            for alias in item.names:
                imports.append(f"import {alias.name}")
        elif isinstance(item, ast.ImportFrom):
            module_name = item.module
            for alias in item.names:
                imports.append(f"from {module_name} import {alias.name}")
    return imports


# Function to get import statements as a formatted string from a given file
def get_imports_as_string(file_path):
    with open(file_path, 'r') as file:
        source_code = file.read()

    tree = ast.parse(source_code)
    imports = extract_imports(tree)

    imports_string = "\n".join(imports)
    return imports_string


def find_parent_dir(path, dir_name):
    while path != os.path.dirname(path):
        if os.path.basename(path) == dir_name:
            return path
        path = os.path.dirname(path)
    return None


proj_dir = find_parent_dir(os.path.abspath(__file__), "handy")




class RunningMeanStd(nn.Module, Sourceable):
    def __init__(self, epsilon: float = 1e-4, shape=(), *args, **kwargs):
        """
        Calulates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

        :param epsilon: helps with arithmetic issues
        :param shape: the shape of the data stream's output
        """
        super().__init__(*args, **kwargs)
        self.mean = nn.Parameter(torch.zeros(shape, dtype=torch.float), requires_grad=False)
        self.var = nn.Parameter(torch.ones(shape, dtype=torch.float), requires_grad=False)
        self.count = epsilon
        self.epsilon = nn.Parameter(torch.tensor(epsilon), requires_grad=False)

    def copy(self) -> "RunningMeanStd":
        """
        :return: Return a copy of the current object.
        """
        new_object = RunningMeanStd(shape=self.mean.shape)
        new_object.mean = self.mean * 1
        new_object.var = self.var * 1
        new_object.count = float(self.count)
        return new_object

    def combine(self, other: "RunningMeanStd") -> None:
        """
        Combine stats from another ``RunningMeanStd`` object.

        :param other: The other object to combine with.
        """
        self.update_from_moments(other.mean, other.var, other.count)

    def update(self, arr: torch.Tensor) -> None:
        batch_mean = torch.mean(arr, dim=0)
        batch_var = torch.var(arr, dim=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean: torch.Tensor, batch_var: torch.Tensor, batch_count: float) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + torch.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean.data = new_mean
        self.var.data = new_var
        self.count = new_count

    def normalize(self, arr: torch.Tensor) -> torch.Tensor:
        return torch.clip((arr - self.mean) / torch.sqrt(self.var + self.epsilon), -1000, 1000)


class MLPSkipper(nn.Module, Sourceable):

    def __init__(self, nail_size: int, hidden_size: int, hammer_size: int):
        super().__init__()
        self.lns = nn.ModuleList([
            nn.LayerNorm(nail_size),
            nn.LayerNorm(hidden_size),
            nn.LayerNorm(hidden_size),
        ])
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(nail_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
            ),
            nn.Sequential(
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
            ),
            nn.Sequential(
                nn.ReLU(),
                nn.Linear(hidden_size, hammer_size),
            )
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lns[0].forward(x)
        x = self.layers[0].forward(x)

        x = self.lns[1].forward(x)
        y = self.layers[1].forward(x)
        x = x + y

        x = self.lns[2].forward(x)
        x = self.layers[2].forward(x)
        return x


class MLPSkipperAE(nn.Module, Sourceable):

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance.args = args
        instance.kwargs = kwargs
        return instance

    def __init__(self, nail_size: int, hidden_size: int, hammer_size: int):
        super().__init__()
        self.rms = RunningMeanStd(shape=(nail_size,))
        self.encoder = nn.Sequential(MLPSkipper(nail_size, hidden_size, hammer_size), nn.Tanh())
        self.decoder = MLPSkipper(hammer_size, hidden_size, nail_size)

    def forward(self, x: torch.Tensor, train_yes: bool) -> (torch.Tensor, torch.Tensor):
        """
        Returns two tensors: encoded and reconstructed
        """
        with torch.no_grad():
            if train_yes:
                self.rms.update(x)
            x = self.rms.normalize(x * 1)
        # x is shape (B, nail_size)
        z0 = self.encoder(x)
        z1 = self.decoder(z0)
        return z0, z1

#Source: https://medium.com/@sofeikov/implementing-variational-autoencoders-from-scratch-533782d8eb95
class VAE(nn.Module, Sourceable):
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance.args = args
        instance.kwargs = kwargs
        return instance

    def __init__(self, nail_size: int, hidden_size: int, hammer_size: int):
        super().__init__()
        self.rms = RunningMeanStd(shape=(nail_size,))
        
        self.encoder = nn.Sequential(
            nn.Linear(nail_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size, track_running_stats=False),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size, track_running_stats=False),
            nn.Linear(hidden_size, hammer_size),
            nn.ReLU(),
            nn.BatchNorm1d(hammer_size),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hammer_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, nail_size)
        )
        '''
        self.encoder = nn.Sequential(MLPSkipper(nail_size, hidden_size, hammer_size))
        self.decoder = MLPSkipper(hammer_size, hidden_size, nail_size)
        
        '''

        self.test = nn.Sequential(
            nn.Linear(nail_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size, track_running_stats=False))
        self.hammer_size = hammer_size
        # Add mu and log_var layers for reparameterization
        self.mu = nn.Linear(hammer_size,hammer_size)
        self.log_var = nn.Linear(hammer_size,hammer_size)

    def reparameterize(self, mu, log_var):
        # Compute the standard deviation from the log variance
        std = torch.exp(0.5 * log_var)
        # Generate random noise using the same shape as std
        eps = torch.randn_like(std)
        # Return the reparameterized sample
        return mu + eps * std
    
    def reparameterize_zeroNoise(self, mu, log_var):
        # Compute the standard deviation from the log variance
        std = torch.exp(0.5 * log_var)
        # Generate random noise using the same shape as std
        eps = torch.zeros_like(std)
        # Return the reparameterized sample
        return mu + eps * std

    def forward(self, x: torch.Tensor, train_yes: bool) -> (torch.Tensor, torch.Tensor):
        with torch.no_grad():
            if train_yes:
                self.rms.update(x)
            x = self.rms.normalize(x * 1)
        # Pass the input through the encoder
        test = self.test(x)
        encoded = self.encoder(x)
        # Compute the mean and log variance vectors
        mu = self.mu(encoded)
        log_var = self.log_var(encoded)
        # Reparameterize the latent variable
        z = self.reparameterize(mu, log_var)
        # Pass the latent variable through the decoder
        decoded = self.decoder(z)
        # Return the encoded output, decoded output, mean, and log variance
        return encoded, decoded, mu, log_var, test

    def sample(self):
        with torch.no_grad():
            # Generate random noise
            z = torch.randn(1, self.hammer_size)
            # Pass the noise through the decoder to generate samples
            samples = self.decoder(z)
        # Return the generated samples
        return samples


class ThroughDataset(Dataset):
    """
    Sacrifice some readability to make life easier.
    Whatever input array/argument tensor provided will be the output for dataset.
    """

    def __init__(self, *args):
        self.args = args
        for a1, a2 in zip(self.args, self.args[1:]):
            assert a1.shape[0] == a2.shape[0]

    def __getitem__(self, index):
        indexed = tuple(torch.as_tensor(a[index]) for a in self.args)
        return indexed

    def __len__(self):
        return self.args[0].shape[0]


def main():
    # # Example usage:
    # file_path = __file__
    # imports_string = get_imports_as_string(file_path)
    # print(imports_string)

    # # Treat this training as OTF augmentation since we have lots of poses
    # motion_lib_args = torch.load("motion_lib_args.pkl")
    # motion_lib: MotionLibSMPL = torch.load("their_motion_lib.pkl")
    # n_loads_per_epoch = 1024
    # n_samples_per_load = 128
    # # gender_betas = motion_lib_args["gender_betas"]
    # # limb_weights = motion_lib_args["limb_weights"]
    # # skeleton_trees = motion_lib_args["skeleton_trees"]
    # gender_betas = motion_lib_args["gender_betas"].repeat_interleave(n_loads_per_epoch, 0)
    # limb_weights = motion_lib_args["limb_weights"].repeat_interleave(n_loads_per_epoch, 0)
    # skeleton_trees = motion_lib_args["skeleton_trees"] * n_loads_per_epoch
    # motion_lib_kwargs = motion_lib.kwargs
    # motion_lib_kwargs["motion_file"] = "/home/nhgk/scratch/workspace/PerpetualHumanoidControl/data/amass/pkls/amass_copycat_take5_train.pkl"
    # motion_lib: MotionLibSMPL = motion_lib.__class__(**motion_lib_kwargs)
    #

    if args.debug1:
        debugpy.listen(5678)
        print("Waiting for debugger attach")
        debugpy.wait_for_client()

    canvas = scene.SceneCanvas(size=(800, 600), show=False)
    view = canvas.central_widget.add_view()
    view.camera = scene.TurntableCamera(
        up="z", fov=1.0, elevation=15.0, distance=300, azimuth=15
    )
    view.camera.rect = 0, 0, 1, 1
    _ = scene.visuals.XYZAxis(parent=view.scene)
    vd = VisualDataContainer(view)
    img = canvas.render()






    model: MLPSkipperAE or None = None
    optimizer = None

    writer = SummaryWriter(f"logdir/ae/{args.kld_weight}")

    n_epochs = 150


    current_path = os.getcwd()
    print("Current Path:", current_path)

    
    #pkl_file_path = f"{current_path}/data/amass/torchready_1.pkl"
    #xs = torch.load(pkl_file_path)["rg_pos"]
    
    #NOTE: Use this if you want to use dance only anim set.
    
    pkl_file_path = f"{current_path}/data/amass/torchready_all.pkl"
    with open(pkl_file_path, 'rb') as file:
        # Load the data from the pkl file
        unpickler = pickle.Unpickler(file)
        data = unpickler.load()
    #xs1 = pickle.load(f"{current_path}/data/amass/torchready.pkl")
    with open (pkl_file_path, 'rb') as pickle_file:
        content = pickle.load(pickle_file)
    #xs = {key: torch.tensor(value) for key, value in data.items()}
    
    xs = data #xs["rg_pos"]
    print(xs[:, [0], :2].shape)
    a
    xs[:, :, :2] -= xs[:, [0], :2]
    
    n_train = int(xs.shape[0] * 0.90)
    n_valid = xs.shape[0] - n_train
    train_idxs = np.arange(n_train)
    valid_idxs = np.setdiff1d(np.arange(xs.shape[0]), train_idxs)
    xs_train = xs[train_idxs] * 1.0
    print(valid_idxs.shape)
    xs_valid = xs[valid_idxs[:int(valid_idxs.shape[0]/4)]] * 1.0

    #print(torch.max(xs_train, dim = -1))
    xs_train = xs_train.reshape(xs_train.shape[0], -1)
    xs_valid = xs_valid.reshape(xs_valid.shape[0], -1)

    model = VAE(xs_valid.shape[-1], 4096, 10)
    model = model.cuda()
    model.rms = model.rms.cuda()
    model.get_src(0)
    optimizer = Adam(model.parameters(), 3e-4)
    model.eval()
    with torch.no_grad():
        encoded, decoded, mu, log_var, test = model.forward(xs_valid, False)
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        MSE = torch.nn.functional.mse_loss(xs_valid, decoded) 
        loss = MSE + KLD   
        writer.add_scalar("valid/MSEloss", MSE.item(), 0)
        writer.add_scalar("valid/KLDloss", KLD.item(), 0)  
        for j in range(10):
            writer.add_scalar(f"valid/mean{j}", torch.mean(encoded[:,j]), 0)
            writer.add_scalar(f"valid/std{j}", torch.std(encoded[:,j]), 0)


    for ep in tqdm(range(n_epochs)):
        if model is None:
            
            #model = MLPSkipperAE(xs_train.shape[-1], 1024, 10)
            model = VAE(xs_train.shape[-1], 4096, 10)
            model = model.cuda()
            model.rms = model.rms.cuda()
            model.get_src(0)
            optimizer = Adam(model.parameters(), 3e-4)
            

        dataset = ThroughDataset(xs_train.cpu().detach().numpy())
        dataloader = DataLoader(dataset, batch_size=4096, shuffle=False)
        pbar = tqdm(total=len(dataset))

        for i, (x,) in enumerate(dataloader):
            x = x.pin_memory().to("cuda", non_blocking=True, dtype=torch.float)

            encoded, decoded, mu, log_var = model(x, True)



             # Compute the loss and perform backpropagation
            KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

            loss = torch.nn.functional.mse_loss(x, decoded) #+ args.kld_weight * KLD 
            #loss += 1e-2 * torch.mean(encoded ** 2)
            #loss_function(decoded, x, mu, log_var, xs_train.shape[-1])#
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.update(x.shape[0])
            pbar.set_postfix({"loss": loss.item(), "ep": ep})
            

        writer.add_scalar("train/loss", loss.item(), ep)
        pbar.close()
        #print(model.get_src(0))
        d = {
            "imports": get_imports_as_string(__file__),
            "model_src": model.get_src(0),
            "model_cls_name": model.__class__.__name__,
            "model_args": model.args,
            "model_kwargs": model.kwargs,
            "model_state_dict": model.state_dict(),
        }
  
        torch.save(d, f"good/vae{args.kld_weight}.pkl")
        
        with torch.no_grad():
            '''
            encoded, decoded, mu, log_var = model.forward(xs_train, False)
            xs_train_np = xs_train.detach().cpu().numpy()
            decoded_train_np = decoded.detach().cpu().numpy()

            for j in range(5):

                vd.body_markers[0].set_data(pos=xs_train_np[j].reshape(-1,3), size=0.1, edge_width=0, edge_color=(0, 0, 0, 0), face_color=(1, 0, 0, 1))
                vd.body_markers[1].set_data(pos=decoded_train_np[j].reshape(-1,3), size=0.1, edge_width=0, edge_color=(0, 0, 0, 0), face_color=(0, 0, 1, 1))
                canvas.update()
                img = canvas.render()

                writer.add_image(f"valid/trainImage_{j}", img,global_step=ep, dataformats='HWC')

            
            encoded = model.encoder(model.rms.normalize(xs_valid * 1))
            decoded = model.decoder(encoded)
            xs_valid_np_full = xs_valid.detach().cpu().numpy()
            decoded_valie_np_full = decoded.detach().cpu().numpy()

            for j in range(5):

                vd.body_markers[0].set_data(pos=xs_valid_np_full[j].reshape(-1,3), size=0.1, edge_width=0, edge_color=(0, 0, 0, 0), face_color=(1, 0, 0, 1))
                vd.body_markers[1].set_data(pos=decoded_valie_np_full[j].reshape(-1,3), size=0.1, edge_width=0, edge_color=(0, 0, 0, 0), face_color=(0, 0, 1, 1))
                canvas.update()
                img = canvas.render()

                writer.add_image(f"valid/AEImage_{j}", img,global_step=ep, dataformats='HWC')

            
            encoded2 = model.encoder(model.rms.normalize(xs_valid * 1))
            mu2 = model.mu(encoded2)
            log_var2 = model.log_var(encoded)
            z2 = model.reparameterize_zeroNoise(mu2,log_var2)
            decoded2 = model.decoder(z2)
            xs_valid_np_full2 = xs_valid.detach().cpu().numpy()
            decoded_valie_np_full2 = decoded2.detach().cpu().numpy()

            for j in range(5):

                vd.body_markers[0].set_data(pos=xs_valid_np_full2[j].reshape(-1,3), size=0.1, edge_width=0, edge_color=(0, 0, 0, 0), face_color=(1, 0, 0, 1))
                vd.body_markers[1].set_data(pos=decoded_valie_np_full2[j].reshape(-1,3), size=0.1, edge_width=0, edge_color=(0, 0, 0, 0), face_color=(0, 0, 1, 1))
                canvas.update()
                img = canvas.render()

                writer.add_image(f"valid/AEImage_zeroNoise_{j}", img,global_step=ep, dataformats='HWC')
            '''
            encoded, decoded, mu, log_var = model.forward(xs_valid, False)
            KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            MSE = torch.nn.functional.mse_loss(xs_valid, decoded) 
            loss = MSE + KLD
            xs_valid_np = xs_valid.detach().cpu().numpy()
            decoded_np = decoded.detach().cpu().numpy()

            digit_size = 128
            n = 5
            norm = td.Normal(0, 1)
            grid_y = norm.icdf(torch.linspace(0.05, 0.95, n))
            
            '''for i in range(10):
                big_img = []
                for j, xi in enumerate(grid_y):
                    z = torch.zeros(1,10).to("cuda", non_blocking=True, dtype=torch.float)
                    z[0,i] = xi
                    x_decoded = model.decoder(z)
                    x_decoded = x_decoded.detach().cpu().numpy()
                    vd.body_markers[0].set_data(pos=x_decoded.reshape(-1,3), size=0.1, edge_width=0, edge_color=(0, 0, 0, 0), face_color=(1, 0, 0, 1))
                    vd.body_markers[1].set_data(pos=x_decoded.reshape(-1,3), size=0.1, edge_width=0, edge_color=(0, 0, 0, 0), face_color=(0, 0, 1, 1))
                    canvas.update()
                    img = canvas.render()
                    big_img.append(img)
                big_img = np.concatenate(big_img,axis=1)
                writer.add_image(f"sample/image_{i}", big_img,global_step=ep, dataformats='HWC')
            '''
            for j in range(5):

                vd.body_markers[0].set_data(pos=xs_valid_np[j].reshape(-1,3), size=0.1, edge_width=0, edge_color=(0, 0, 0, 0), face_color=(1, 0, 0, 1))
                vd.body_markers[1].set_data(pos=decoded_np[j].reshape(-1,3), size=0.1, edge_width=0, edge_color=(0, 0, 0, 0), face_color=(0, 0, 1, 1))
                canvas.update()
                img = canvas.render()

                writer.add_image(f"valid/image_{j}", img,global_step=ep, dataformats='HWC')
            #loss += 1e-2 * torch.mean(encoded ** 2)
            writer.add_scalar("valid/MSEloss", MSE.item(), ep)
            writer.add_scalar("valid/KLDloss", KLD.item(), ep)

            for j in range(10):
                writer.add_scalar(f"valid/mean{j}", torch.mean(encoded[:,j]), ep)
                writer.add_scalar(f"valid/std{j}", torch.std(encoded[:,j]), ep)
            


    writer.flush()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--kld_weight", type=float, default=0)
    parser.add_argument("--experiment", type=str, default="")
    parser.add_argument("--debug1", type=bool)
    args = parser.parse_args()

    main()
