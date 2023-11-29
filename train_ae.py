import ast
import inspect
import os

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


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
    model: MLPSkipperAE or None = None
    optimizer = None

    writer = SummaryWriter("logdir/ae")

    n_epochs = 100
    xs = torch.load(f"{proj_dir}/../datasets/AMASS/torchready.pkl")["rg_pos"]
    xs[:, :, :2] -= xs[:, [0], :2]
    xs = xs.reshape(xs.shape[0], -1)
    for ep in tqdm(range(n_epochs)):
        if model is None:
            model = MLPSkipperAE(xs.shape[-1], 2048, 10)
            model = model.cuda()
            model.rms = model.rms.cuda()
            model.get_src(0)
            optimizer = Adam(model.parameters(), 3e-4)

        dataset = ThroughDataset(xs.cpu().detach().numpy())
        dataloader = DataLoader(dataset, batch_size=4096, shuffle=False)
        pbar = tqdm(total=len(dataset))
        for i, (x,) in enumerate(dataloader):
            x = x.pin_memory().to("cuda", non_blocking=True, dtype=torch.float)
            z0, z1 = model.forward(x, True)
            loss = torch.nn.functional.mse_loss(x, z1)
            loss += 1e-2 * torch.mean(z0 ** 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.update(x.shape[0])
            pbar.set_postfix({"loss": loss.item()})

        writer.add_scalar("train/loss", loss.item(), ep)
        pbar.close()

        d = {
            "imports": get_imports_as_string(__file__),
            "model_src": model.get_src(0),
            "model_cls_name": model.__class__.__name__,
            "model_args": model.args,
            "model_kwargs": model.kwargs,
            "model_state_dict": model.state_dict(),
        }
        torch.save(d, "ae.pkl")


if __name__ == "__main__":
    # import pydevd_pycharm
    #
    # pydevd_pycharm.settrace('localhost', port=12345, stdoutToServer=True, stderrToServer=True, suspend=False)
    main()
