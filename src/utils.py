import os
import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Dropout, BatchNorm1d, Sequential as Seq
from torch_geometric.utils import get_laplacian
from torch_geometric.datasets import (
    CitationFull,
    Amazon,
    WebKB,
    Planetoid,
    Coauthor,
    WikipediaNetwork,
    Airports,
    PolBlogs,
)
from torch_geometric.nn import (
    GCNConv,
    GATConv,
    GINConv,
    TransformerConv,
    SGConv,
    FAConv,
    SAGEConv,
    ChebConv,
    GraphConv,
    GatedGraphConv,
    GCN2Conv,
    GATv2Conv,
    APPNP,
    GPSConv,
)

gnn_models = {
    "GCN_loop": GCNConv,
    "GCN": GCNConv,
    "GCN2": GCN2Conv,
    "GAT": GATConv,
    "GATv2": GATv2Conv,
    "GIN": GINConv,
    "Transformer": TransformerConv,
    "SGC": SGConv,
    "FA": FAConv,
    "SAGE": SAGEConv,
    "Cheb": ChebConv,
    "Graph": GraphConv,
    "GatedGraph": GatedGraphConv,
    "APPNP": APPNP,
    "GPS": GPSConv,
}

class MLP(torch.nn.Module):
    def __init__(self, args, in_channels, layer_widths, num_classes):
        super(MLP, self).__init__()
        self.args = args
        self.layer_widths = layer_widths
        self.num_layers = len(layer_widths)
        self.dropout = args.dropout
        self.use_batchnorm = args.batchnorm

        self.layers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList() if args.batchnorm else None

        self.layers.append(Linear(in_channels, layer_widths[0]))
        if self.use_batchnorm:
            self.batch_norms.append(BatchNorm1d(layer_widths[0]))
        for i in range(self.num_layers - 1):
            self.layers.append(Linear(layer_widths[i], layer_widths[i + 1]))
            if self.use_batchnorm:
                self.batch_norms.append(BatchNorm1d(layer_widths[i + 1]))

        self.classifier_head = Linear(layer_widths[-1], num_classes)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.use_batchnorm:
                x = self.batch_norms[i](x)
            x = F.relu(x)
            x = Dropout(self.dropout)(x)
        x = torch.mean(x, dim=0)
        x = self.classifier_head(x)

        return x


class GNNModel(torch.nn.Module):
    def __init__(
        self,
        args,
        gnn,
        in_channels,
    ):
        super(GNNModel, self).__init__()
        self.gnn = gnn
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList() if args.batchnorm else None
        self.hidden_channels = args.h_dim
        self.num_layers = args.layers
        self.dropout = args.dropout
        self.use_batchnorm = args.batchnorm
        self.out_channels = args.num_node_classes

        if gnn in ("SGC", "GCN"):
            self.convs.append(
                gnn_models[gnn](in_channels, self.hidden_channels, add_self_loops=False)
            )
        elif gnn == "GPS":
            self.projector = Linear(in_channels, self.hidden_channels)
            self.convs.append(
                GPSConv(
                    channels=self.hidden_channels,
                    conv=GCNConv(self.hidden_channels, self.hidden_channels),
                    norm=None,
                    heads=2,
                )
            )
        elif gnn == "GCN2":
            self.projector = Linear(in_channels, self.hidden_channels)
            self.convs.append(GCN2Conv(self.hidden_channels, alpha=0.2))
        elif gnn == "GIN":
            nn_start = Seq(
                Linear(in_channels, self.hidden_channels),
                ReLU(),
                Linear(self.hidden_channels, self.hidden_channels),
            )
            self.convs.append(GINConv(nn_start))
        elif gnn == "APPNP":
            self.projector = Linear(in_channels, self.hidden_channels)
            self.convs.append(APPNP(K=self.num_layers, alpha=0.1))
            self.out_projector = Linear(self.hidden_channels, self.out_channels)
        elif gnn == "FA":
            conv_layer = Seq(
                FAConv(in_channels), Linear(in_channels, self.hidden_channels)
            )
        elif gnn == "Cheb":
            self.convs.append(ChebConv(in_channels, self.hidden_channels, 3))
        elif gnn == "GatedGraph":
            self.projector = Linear(in_channels, self.hidden_channels)
            conv_layer = GatedGraphConv(out_channels=self.hidden_channels, num_layers=5)
            self.convs.append(conv_layer)
        else:
            self.convs.append(gnn_models[gnn](in_channels, self.hidden_channels))
        if self.use_batchnorm:
            self.batch_norms.append(BatchNorm1d(self.hidden_channels))
        for _ in range(self.num_layers - 2):
            if gnn in ("SGC", "GCN"):
                self.convs.append(
                    gnn_models[gnn](
                        self.hidden_channels, self.hidden_channels, add_self_loops=False
                    )
                )
            elif gnn == "GCN2":
                self.convs.append(GCN2Conv(self.hidden_channels, alpha=0.2))
            elif gnn == "GPS":
                self.convs.append(
                    GPSConv(
                        channels=self.hidden_channels,
                        conv=GCNConv(self.hidden_channels, self.hidden_channels),
                        norm=None,
                        heads=2,
                    )
                )
            elif gnn == "GIN":
                nn_mid = Seq(
                    Linear(self.hidden_channels, 32),
                    ReLU(),
                    Linear(32, self.hidden_channels),
                )
                self.convs.append(GINConv(nn_mid))
            elif gnn == "FA":
                conv_layer = Seq(
                    FAConv(self.hidden_channels),
                    Linear(self.hidden_channels, self.hidden_channels),
                )
            elif gnn == "Cheb":
                self.convs.append(
                    ChebConv(self.hidden_channels, self.hidden_channels, 2)
                )
            elif gnn == "GatedGraph":
                conv_layer = GatedGraphConv(self.hidden_channels, 5)
                self.convs.append(conv_layer)
            else:
                self.convs.append(
                    gnn_models[gnn](self.hidden_channels, self.hidden_channels)
                )
            if self.use_batchnorm:
                self.batch_norms.append(BatchNorm1d(self.hidden_channels))
        if gnn == "SGC" or gnn == "GCN":
            self.convs.append(
                gnn_models[gnn](
                    self.hidden_channels, self.out_channels, add_self_loops=False
                )
            )
        elif gnn == "GPS":
            self.convs.append(
                GPSConv(
                    channels=self.hidden_channels,
                    conv=GCNConv(self.hidden_channels, self.hidden_channels),
                    norm=None,
                    heads=2,
                )
            )
            self.out_projection = Linear(self.hidden_channels, self.out_channels)
        elif gnn == "GIN":
            nn_mid = Seq(
                Linear(self.hidden_channels, self.hidden_channels),
                ReLU(),
                Linear(self.hidden_channels, self.hidden_channels),
            )
            self.convs.append(GINConv(nn_mid))
            self.out_projection = Linear(self.hidden_channels, self.out_channels)
        elif gnn == "FA":
            conv_layer = Seq(
                FAConv(in_channels), Linear(in_channels, self.out_channels)
            )
            self.convs.append(conv_layer)
        elif gnn == "GCN2":
            self.convs.append(GCN2Conv(self.hidden_channels, alpha=0.2))
            self.out_projection = Linear(self.hidden_channels, self.out_channels)
        elif gnn == "Cheb":
            self.convs.append(ChebConv(self.hidden_channels, self.out_channels, 2))
        elif gnn == "GatedGraph":
            conv_layer = GatedGraphConv(self.hidden_channels, 5)
            self.convs.append(conv_layer)
            self.out_projector = Linear(self.hidden_channels, self.out_channels)
        else:
            self.convs.append(gnn_models[gnn](self.hidden_channels, self.out_channels))
        self.dropout = Dropout(self.dropout)
        self.X = None


    def forward(self, x, edge_index):
        x = x.float()
        if self.gnn == "FA":
            x_0 = x.clone().detach()
            for i, conv in enumerate(self.convs[:-1]):
                x = conv(x, x_0, edge_index)
                if self.batch_norms is not None:
                    x = self.batch_norms[i](x)
                x = F.relu(x)
                x = self.dropout(x)
            x = self.convs[-1][0](x, x_0, edge_index)
            x = self.convs[-1][1](x)
        elif self.gnn == "GPS":
            x = self.projector(x)
            for i, conv in enumerate(self.convs):
                x = conv(x, edge_index)
                x = self.dropout(x)
            x = self.out_projection(x)
        elif self.gnn == "GatedGraph":
            x = self.projector(x)
            for i, conv in enumerate(self.convs[:-1]):
                x = conv(x, edge_index)
                x = self.dropout(x)
            x = self.out_projector(x)
        elif self.gnn == "GCN2":
            initial_x = self.projector(x.clone().detach())
            x = initial_x
            for i, conv in enumerate(self.convs[:-1]):
                x = conv(x, x_0=initial_x, edge_index=edge_index)
                if self.batch_norms is not None:
                    x = self.batch_norms[i](x)
                x = F.relu(x)
                x = self.dropout(x)
            x = self.convs[-1](x, x_0=initial_x, edge_index=edge_index)
            x = self.out_projection(x)
        elif self.gnn == "APPNP":
            x = self.projector(x)
            x = self.convs[0](x, edge_index)
            x = self.out_projector(x)
        else:
            for i, conv in enumerate(self.convs[:-1]):
                x = conv(x, edge_index)
                if self.batch_norms is not None:
                    x = self.batch_norms[i](x)
                x = F.relu(x)
                x = self.dropout(x)
            x = self.convs[-1](x, edge_index)

        if self.gnn == "GIN":
            x = self.out_projection(x)

        return x


class EigenStorage:
    def __init__(self, args, path, data) -> None:
        if not os.path.exists(path):
            os.makedirs(path)

        self.path = path
        self.data = data

    def save(self, eigenvalues, eigenvectors):
        torch.save(eigenvalues, self.path + "eigenvalues.pt")
        torch.save(eigenvectors, self.path + "eigenvectors.pt")

    def load(self):
        eigenvalues = torch.load(self.path + "eigenvalues.pt")
        eigenvectors = torch.load(self.path + "eigenvectors.pt")
        if eigenvectors.shape[0] != self.data.num_nodes:
            eigenvalues, eigenvectors = self.decompose()
            self.save(eigenvalues, eigenvectors)

        return eigenvalues, eigenvectors

    def get(self, force_decompose=False):
        if force_decompose:
            eigenvalues, eigenvectors = self.decompose()
            self.save(eigenvalues, eigenvectors)
            return eigenvalues, eigenvectors

        try:
            eigenvalues, eigenvectors = self.load()
        except FileNotFoundError:
            eigenvalues, eigenvectors = self.decompose()
            self.save(eigenvalues, eigenvectors)

        return eigenvalues, eigenvectors

    def decompose(self):
        laplacian_matrix = get_laplacian(
            self.data.edge_index, num_nodes=self.data.num_nodes, normalization="sym"
        )
        laplacian_matrix = torch.sparse_coo_tensor(
            laplacian_matrix[0], laplacian_matrix[1]
        ).to_dense()
        print("\tstart decompose adj...")
        eigenvalues, eigenvectors = torch.linalg.eig(laplacian_matrix)
        print("\tadj decomposition finished")
        eigenvalues, eigenvectors = torch.real(eigenvalues), torch.real(eigenvectors)
        return eigenvalues, eigenvectors


def make_splits(dataset, train_val_test_ratios=(0.8, 0.1, 0.1)):
    train_ratio, val_ratio, test_ratio = train_val_test_ratios

    num_nodes = dataset.data.x.shape[0]
    nodes = torch.arange(num_nodes)
    perm = torch.randperm(num_nodes)
    nodes = nodes[perm]

    train_mask = nodes < int(train_ratio * num_nodes)
    val_mask = (nodes >= int(train_ratio * num_nodes)) & (
        nodes < int((train_ratio + val_ratio) * num_nodes)
    )
    test_mask = nodes >= int((train_ratio + val_ratio) * num_nodes)

    dataset.train_mask = train_mask
    dataset.val_mask = val_mask
    dataset.test_mask = test_mask


def dataset_picker(dataset_name: str):
    data_dir = "../../data"

    if dataset_name in ["Physics", "CS"]:
        dataset = Coauthor(root=data_dir, name=dataset_name)

    elif dataset_name in ("chameleon", "squirrel", "crocodile"):
        dataset = WikipediaNetwork(root=data_dir, name=dataset_name)

    # Planetoid datasets
    elif dataset_name in ["Cora", "Citeseer", "PubMed"]:
        dataset = Planetoid(root=data_dir, name=dataset_name)

    # CitationFull datasets
    elif dataset_name in [
        "Cora_Full",
        "Cora-ML",
        "CiteSeer-Full",
        "PubMed-Full",
        "DBLP",
    ]:
        if dataset_name == "Cora_Full":
            citation_dataset = "Cora"
        elif dataset_name == "DBLP":
            citation_dataset = "DBLP"

        if dataset_name in ["Cora-ML", "CiteSeer-Full", "PubMed-Full"]:
            citation_dataset = dataset_name.split("-")[0]

        dataset = CitationFull(root=data_dir, name=citation_dataset)

    # Amazon datasets
    elif dataset_name in ["Computers", "Photo", "Products"]:
        dataset = Amazon(root=data_dir, name=dataset_name)

    # WebKB datasets
    elif dataset_name in ["Cornell", "Texas", "Wisconsin"]:
        dataset = WebKB(root=data_dir, name=dataset_name)

    elif dataset_name.startswith("Airports"):
        country = dataset_name.split("_")[1]
        dataset = Airports(root=data_dir, name=country)
    elif dataset_name == "PolBlogs":
        dataset = PolBlogs(root=data_dir)
    else:
        raise ValueError(f"Dataset {dataset_name} not found")

    if not hasattr(
        dataset, "x"
    ):
        dataset.data.x = torch.eye(dataset.num_nodes)

    heterophilic_datasets = [
        "chameleon",
        "squirrel",
        "crocodile",
        "Cornell",
        "Texas",
        "Wisconsin",
    ]

    if not hasattr(dataset, "train_mask") or dataset_name in heterophilic_datasets:
        make_splits(dataset, (0.8, 0.1, 0.1))

    if dataset.data.y.dim() != 1:
        dataset.data.y = dataset.data.y.argmax(dim=-1)

    return dataset, dataset.train_mask, dataset.val_mask, dataset.test_mask

def get_laplacian_spectrum(edge_index: torch.Tensor, edge_weight: torch.Tensor):
    laplacian = get_laplacian(edge_index, edge_weight, normalization="sym")

    laplacian_matrix = torch.sparse_coo_tensor(laplacian[0], laplacian[1]).to_dense()
    eigenvalues, eigenvectors = torch.linalg.eig(laplacian_matrix)
    eigenvalues, eigenvectors = (
        torch.real(eigenvalues),
        torch.real(eigenvectors),
    )

    eigenvalues, idx = eigenvalues.sort()
    eigenvectors = eigenvectors[:, idx]

    return eigenvalues, eigenvectors
