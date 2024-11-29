import torch 
from tqdm import tqdm

def training_loop(n_epochs, optimizer, model, loss_fn, train_loader, val_loader, device, resize_transform=None, early_stopping_patience=10):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf') 
    patience_counter = 0 

    scaler = torch.amp.GradScaler("cuda")

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0

        for img, mask in tqdm(train_loader, desc="Training", leave=True, mininterval=2.0):
            if resize_transform:
                img = resize_transform(img)
                mask = resize_transform(mask)

            img, mask = img.to(device), mask.to(device)

            optimizer.zero_grad()  

            with torch.cuda.amp.autocast():
                y_pred = model(img)
                loss = loss_fn(y_pred, mask)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0

        with torch.no_grad():
            for idx, (img, mask) in enumerate(tqdm(val_loader, desc="Validation", leave=True, mininterval=2.0)):
                if resize_transform:
                    img = resize_transform(img)
                    mask = resize_transform(mask)

                img, mask = img.to(device), mask.to(device)

                y_pred = model(img)
                loss = loss_fn(y_pred, mask)
                val_loss += loss.item()

            val_loss /= len(val_loader)
            val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }
            torch.save(checkpoint, "best_model_checkpoint.pth")
            print(f"New best model saved at epoch {epoch+1} with val_loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement in val_loss for {patience_counter} epoch(s).")

        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch+1}.")
            break

        torch.cuda.empty_cache()  
        gc.collect()  

        print("-" * 30)
        print(f"Epoch: {epoch + 1}, Train Loss: {train_loss:.4f}")
        print(f"Epoch: {epoch + 1}, Val Loss: {val_loss:.4f}")
        print("-" * 30)

    return train_losses, val_losses