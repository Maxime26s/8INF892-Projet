import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


class GAT(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels=32,
        num_layers=2,
        dropout=0.2,
        heads=1,
        concat=False,
    ):
        super(GAT, self).__init__()

        self.dropout = dropout

        if num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        )
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(
                    hidden_channels * heads,
                    hidden_channels,
                    heads=heads,
                    dropout=dropout,
                )
            )
        self.convs.append(
            GATConv(
                hidden_channels * heads,
                out_channels,
                heads=1,
                concat=concat,
                dropout=dropout,
            )
        )

    def forward(self, x, edge_index, batch):
        x = x.to(torch.float)

        for conv in self.convs[:-1]:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, edge_index)
            x = F.elu(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index)

        x = global_mean_pool(x, batch)

        return x
