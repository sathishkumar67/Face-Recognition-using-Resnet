from __future__ import annotations
import time
import torch
from tqdm import tqdm
import numpy as np


def train_model(model, device, train_loader, val_loader, criterion, optimizer, 
                epochs=50, patience=5, checkpoint_path='best_model.pth'):
    """
    Professional training loop with key features:
    - Validation with early stopping
    - Mixed precision training
    - Gradient clipping
    - Learning rate scheduling
    - Best checkpoint saving
    - Rich progress reporting
    - GPU memory optimization
    """
    # Initialize training state
    best_val_loss = np.inf
    epochs_no_improve = 0
    scaler = torch.amp.GradScaler(device=device)  # For mixed precision
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
    history = {'train_loss': [], 'val_loss': [], 'lr': []}

    # Early stopping loop
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        train_loss = 0.0

        # Training phase with mixed precision
        with tqdm(train_loader, unit="batch", desc=f"Train Epoch {epoch+1}") as pbar:
            for batch in pbar:
                anchor, positive, negative = (t.to(device, non_blocking=True) 
                                            for t in batch.values())

                # Mixed precision forward
                with torch.autocast(device_type=device, dtype=torch.float16):
                    anchor_emb = model(anchor)
                    positive_emb = model(positive)
                    negative_emb = model(negative)
                    loss = criterion(anchor_emb, positive_emb, negative_emb)

                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                
                # Optimizer step
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()

                # Update metrics
                batch_loss = loss.detach().item()
                train_loss += batch_loss * anchor.size(0)
                pbar.set_postfix(loss=batch_loss, lr=optimizer.param_groups[0]['lr'])

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.inference_mode():
            for batch in val_loader:
                anchor, positive, negative = (t.to(device, non_blocking=True) 
                                            for t in batch.values())
                # Forward pass
                anchor_emb = model(anchor)
                positive_emb = model(positive)
                negative_emb = model(negative)
                loss = criterion(anchor_emb, positive_emb, negative_emb)
                
                # Update metrics
                val_loss += loss.item() * anchor.size(0)

        # Calculate epoch metrics
        train_loss = train_loss / len(train_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)
        lr = optimizer.param_groups[0]['lr']
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(lr)

        # Early stopping check
        if val_loss < best_val_loss - 1e-4:  # Minimum delta threshold
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Save best checkpoint
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                "Train Loss": train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
        else:
            epochs_no_improve += 1

        # Epoch summary
        epoch_time = time.time() - start_time
        print(f"\nEpoch {epoch+1:03d} Summary:")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Learning Rate: {lr:.2e} | Time: {epoch_time:.1f}s")
        print(f"Best Val Loss: {best_val_loss:.4f} | Patience Left: {patience-epochs_no_improve}")
        
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs!")
            break

    return history, model
