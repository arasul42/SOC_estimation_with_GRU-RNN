import scipy.io as sio
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import DataLoader, TensorDataset

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Path to trainset folder
trainrootdir = "trainset"
file_list = [f for f in os.listdir(trainrootdir) if f.endswith(".mat")]

Finaltrain_X, Finaltrain_y = [], []

# Load all .mat files
for j, file in enumerate(file_list):
    path = os.path.join(trainrootdir, file)
    
    # Load MATLAB file
    try:
        load_data = sio.loadmat(path, struct_as_record=False, squeeze_me=True)
    except Exception as e:
        print(f"Error loading {file}: {e}")
        continue

    # Identify the key containing the data
    data_keys = [key for key in load_data.keys() if not key.startswith("__")]
    if len(data_keys) == 0:
        print(f"Skipping {file}, no valid data found.")
        continue

    key_name = data_keys[0]  # Use the first valid key
    features = load_data[key_name]
    
    if not isinstance(features, np.ndarray):
        print(f"Skipping {file}, unexpected data format.")
        continue

    # Convert to Pandas DataFrame
    dfdata = pd.DataFrame(features)
    values1 = dfdata.values.astype("float32")

    # Extract input (X) and output (y) data
    trainshape = (int(values1.shape[0] / 1000) - 1) * 1000
    train_X, train_y = values1[:trainshape, :-1], values1[:trainshape, -1]

    # Reshape for PyTorch: [batch, timesteps, features]
    train_X = train_X.reshape(-1, 1000, train_X.shape[1])

    # **FIX SIZE MISMATCH HERE**: Take every 1000th label to match `train_X`
    train_y = train_y[::1000].reshape(-1, 1)

    # Store in final dataset
    if len(Finaltrain_X) == 0:
        Finaltrain_X, Finaltrain_y = train_X, train_y
    else:
        Finaltrain_X = np.concatenate((Finaltrain_X, train_X), axis=0)
        Finaltrain_y = np.concatenate((Finaltrain_y, train_y), axis=0)

    print(f"Loaded {file} - Shape: {train_X.shape}, {train_y.shape}")

# Convert to PyTorch tensors
Finaltrain_X = torch.tensor(Finaltrain_X, dtype=torch.float32).to(device)
Finaltrain_y = torch.tensor(Finaltrain_y, dtype=torch.float32).to(device)

print(f"Final dataset shape: {Finaltrain_X.shape}, {Finaltrain_y.shape}")

# Create dataset
dataset = TensorDataset(Finaltrain_X, Finaltrain_y)
train_loader = DataLoader(dataset, batch_size=72, shuffle=True)


# Training loop
print("Starting training...")
start_time = time.time()

for epoch in range(opt_epoch):
    model.train()
    epoch_loss = 0

    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = opt_loss(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}/{opt_epoch}, Loss: {epoch_loss / len(train_loader):.6f}")

end_time = time.time()
print(f"Training completed in {end_time - start_time:.2f} seconds.")

# Save model
savename = f"{opt_timestep}Tstep_{opt_sampling}Sampl{opt_hiddennode}Hnode_{opt_epoch}Epoch_{opt_batchsize}Bsize_mae"
torch.save(model.state_dict(), f"trainmodel/{savename}.pth")

# Save loss history
sio.savemat(f"trainmodel/{savename}_traintime_{end_time - start_time:.2f}.mat",
            {'loss': epoch_loss / len(train_loader)})
