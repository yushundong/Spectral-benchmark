import os
import torch

from utils import GNNModel, dataset_picker
from config import config


def train(args, epochs, data, device, num_times=3):
    best_val_acc = 0
    best_val_model = None
    best_val_pred = None

    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)

    runs = []
    test_accs = []

    model = GNNModel(args, args.gnn, data.x.shape[1]).to(device)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.7, patience=10, min_lr=1e-5
    )

    criterion = torch.nn.CrossEntropyLoss()
    for run in range(num_times):
        print(f"Training run {run}...", flush=True)
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            loss = criterion(
                model(data.x, data.edge_index)[data.train_mask],
                data.y[data.train_mask],
            )

            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                val_acc, val_loss, val_pred = evaluate(model, data, criterion)
                print(
                    f"Epoch {epoch}, Loss: {loss.item()}, Val Acc: {val_acc}, Val Loss: {val_loss}",
                    flush=True,
                )
                if val_acc > best_val_acc and epoch > 0:
                    best_val_acc = val_acc
                    best_val_model = model.state_dict()
                    best_val_pred = val_pred

            scheduler.step(loss)

        # test with the best model
        model.load_state_dict(best_val_model)
        test_acc, test_loss, test_pred = evaluate(model, data, criterion, mode="test")
        test_accs.append(test_acc)

        print(f"Best val accuracy: {best_val_acc}", flush=True)
        print(f"Test accuracy: {test_acc}, Test loss: {test_loss}", flush=True)

        run = {
            "test_acc": test_acc,
            "test_loss": test_loss,
            "test_pred": test_pred,
            "best_val_acc": best_val_acc,
            "best_val_pred": best_val_pred,
            "model_state_dict": model.state_dict(),
        }

        runs.append(run)

    fname = f"{args.gnn}_{args.dataset}_{args.h_dim}_{args.dropout}_{args.layers}.pt"
    torch.save(runs, f"{args.results_dir}/{fname}.pt")
    print(f"Run saved to {args.results_dir}/{fname}.pt", flush=True)

    test_accs = torch.tensor(test_accs)
    print(
        f"Average test accuracy: {test_accs.mean()}, Variance: {test_accs.var()}",
        flush=True,
    )


def evaluate(model, data, criterion, mode="val"):
    model.eval()
    mask = data.val_mask if mode == "val" else data.test_mask

    with torch.no_grad():
        pred = model(data.x, data.edge_index)
        loss = criterion(pred[mask], data.y[mask])
        acc = (
            pred[mask].argmax(dim=1) == data.y[mask]
        ).sum().item() / mask.sum().item()

    return acc, loss, pred


def main():
    args = config()
    device = torch.device(f"cuda:{args.device}")

    dataset, train_mask, val_mask, test_mask = dataset_picker(args.dataset)
    args.num_node_classes = dataset.num_classes

    dataset = dataset[0]
    dataset.train_mask = train_mask
    dataset.val_mask = val_mask
    dataset.test_mask = test_mask

    dataset = dataset.to(device)

    print(f"Training {args.gnn} on dataset {args.dataset}", flush=True)
    train(
        args,
        args.epochs,
        dataset,
        device,
        num_times=args.num_exp_times,
    )


if __name__ == "__main__":
    main()
