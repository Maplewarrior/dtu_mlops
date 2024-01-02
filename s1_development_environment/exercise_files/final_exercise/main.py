import click
import torch
import os
from model import MyAwesomeModel
from data import mnist

def compute_accuracy(y_pred, y_true):
    cnt = 0
    for i in range(y_pred.size(0)):
        if torch.argmax(y_pred[i]).item() == y_true[i].item():
            cnt += 1
    return cnt

@click.group()
def cli():
    """Command line interface."""
    pass


@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
def train(lr):
    """Train a model on MNIST."""
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    accuracies = []
    mean_losses = []
    N_epochs = 15
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    model = MyAwesomeModel().to(device)
    model.train()
    train_set, _ = mnist()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print(f'Beginning model training on device: {device}')
    
    for epoch in range(N_epochs):
        print(f'\n\nEpoch: {epoch}')
        acc = 0
        losses = []
        for i, (images, labels) in enumerate(train_set): # loop over datset
            images, labels = images.to(device), labels.to(device)
            out = model(images)
            predictions = out['logits']
            loss = criterion(predictions, labels)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc += compute_accuracy(out['probabilities'], labels)
            if max(1, i) % 300 == 0:
                print(f'Iter: {i}/{len(train_set)}')
                print(f'Current acc: {acc/((i+1)*images.size(0))}')
                print(f'Current loss: {torch.mean(torch.tensor(losses))}')

        acc = acc / len(train_set.dataset)
        accuracies.append(acc)
        mean_losses.append(torch.mean(torch.tensor(losses)))
    
    torch.save(model.state_dict(), 'model_weights.pth')
    print(f"Saved model weights to path:\n{os.path.join(os.getcwd(), 'model_weights.pth')}")
    return accuracies, losses
        

@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint='model_weights.pth'):
    """Evaluate a trained model."""
    print("Evaluating like my life dependends on it")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    losses = []
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    sd = torch.load(model_checkpoint, map_location=device)
    model = MyAwesomeModel().to(device)
    model.load_state_dict(sd)
    model.eval()
    _, test_set = mnist()
    criterion = torch.nn.CrossEntropyLoss()
    acc = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_set): # loop over datset
            images, labels = images.to(device), labels.to(device)
            out = model(images)
            predictions = out['logits']
            loss = criterion(predictions, labels)
            losses.append(loss.item())
            acc += compute_accuracy(out['probabilities'], labels)
            if max(1, i) % 300 == 0:
                print(f'Iter: {i}/{len(test_set)}')
                print(f'Current acc: {acc/((i+1)*images.size(0))}')
                print(f'Current loss: {torch.mean(torch.tensor(losses))}')

        acc = acc / len(test_set.dataset)
        mean_loss = torch.mean(torch.tensor(losses))
    print(f'Final test loss: {mean_loss}')
    print(f'Final test accuracy: {acc}')
    return acc, mean_loss


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
