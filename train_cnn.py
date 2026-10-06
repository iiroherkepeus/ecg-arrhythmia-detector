import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# --- 1. Model Architecture ---

class ECG_CNN1D(nn.Module):
    def __init__(self):
        super(ECG_CNN1D, self).__init__()
        
        # Expected input shape: [batch_size, channels=1, length=200]
        
        # Block 1
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2) # Length becomes 100
        
        # Block 2
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2) # Length becomes 50
        
        # Block 3
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2) # Length becomes 25
        
        self.flatten = nn.Flatten()
        
        # Fully connected layers
        # 64 channels * 25 spatial length = 1600
        self.fc1 = nn.Linear(64 * 25, 128)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(128, 1) # Single output logit for binary classification

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.flatten(x)
        x = self.relu4(self.fc1(x))
        x = self.fc2(x) # No sigmoid here because we use BCEWithLogitsLoss
        return x


# --- 2. Main Execution ---

if __name__ == "__main__":
    # Setup device for GPU acceleration if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print("Loading data")
    try:
        X_raw = np.load('X_data.npy')
        y_raw = np.load('y_data.npy')
    except FileNotFoundError:
        print("Error: Could not find 'X_data.npy' or 'y_data.npy'.")
        exit(1)

    # Preprocess labels: 'N' -> 0, 'V' -> 1
    y_binary = np.where(y_raw == 'V', 1, 0)
    
    # Reshape X for 1D CNN: from [samples, length] to [samples, channels, length]
    X = np.expand_dims(X_raw, axis=1)

    # Data split (80/20)
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
    )

    # Convert Numpy arrays to PyTorch Tensors
    # Labels need to be shape [batch, 1] to match the output of the model
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    # Create Datasets and DataLoaders
    batch_size = 32
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = ECG_CNN1D().to(device)

    # Calculate class weight for handling the imbalance (N is much more frequent than V)
    # pos_weight = negative_samples / positive_samples
    num_pos = np.sum(y_train == 1)
    num_neg = np.sum(y_train == 0)
    pos_weight_val = num_neg / num_pos
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32).to(device)
    
    print(f"Calculated pos_weight for loss function: {pos_weight_val:.2f}")

    # Loss and Optimizer
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training Loop
    epochs = 10
    print("\nStarting training...")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass & Optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")

    # Evaluation
    print("\nEvaluating model on test set...")
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            
            # Get logits
            logits = model(inputs)
            
            # The model outputs logits, so we use > 0 to predict class 1
            # (which is mathematically equivalent to sigmoid(logits) > 0.5)
            preds = (logits > 0.0).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Print Metrics
    print("-" * 40)
    print("Classification Report:")
    print("-" * 40)
    print(classification_report(all_labels, all_preds, target_names=['Normal (0)', 'PVC (1)']))
    
    print("-" * 40)
    print("Confusion Matrix:")
    print("-" * 40)
    print(confusion_matrix(all_labels, all_preds))

    # Save Model
    save_path = "cnn_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\nModel state dictionary saved successfully as '{save_path}'")