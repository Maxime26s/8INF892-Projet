from torch_geometric.datasets import QM9

dataset = QM9(root="/tmp/QM9")

num_molecules = len(dataset)
print(f"Number of molecules: {num_molecules}")

num_atoms = [data.num_nodes for data in dataset]
avg_num_atoms = sum(num_atoms) / len(num_atoms)
print(f"Average number of atoms per molecule: {avg_num_atoms:.2f}")

# Inspect one molecule
sample_data = dataset[0]
print(f"Sample molecule features: {sample_data}")

# Assuming the first feature represents atom types
atom_types = [data.x[:, 0].tolist() for data in dataset]
flat_atom_types = [atom for sublist in atom_types for atom in sublist]
unique_atom_types = set(flat_atom_types)

print(f"Unique atom types in the dataset: {unique_atom_types}")

import matplotlib.pyplot as plt

# Choose one property to analyze, for example the first property
property_index = 0
properties = [data.y[:, property_index].item() for data in dataset]

plt.hist(properties, bins=100)
plt.xlabel("Property Value")
plt.ylabel("Frequency")
plt.title("Distribution of a Molecular Property")
plt.show()

import numpy as np

degrees = []
for data in dataset:
    d = np.bincount(data.edge_index[0].numpy(), minlength=data.num_nodes)
    degrees.extend(d)

plt.hist(degrees, bins=np.arange(min(degrees), max(degrees) + 1) - 0.5, rwidth=0.8)
plt.xlabel("Degree")
plt.ylabel("Frequency")
plt.title("Distribution of Node Degrees")
plt.show()
