import argparse
import matplotlib.pyplot as plt

total_iterations = 0
total_tests = 0
total_train_logs = 0

train_logs = []
losses = []

tests = []
val_losses = []
cers = []
wers = []

with open("results/training_results.txt", "r") as f:
    for i, line in enumerate(f.readlines()):
        if "Train Epoch:" in line:
            train_logs.append(total_train_logs)

            loss = float(line.split("Loss:")[-1][:-1])
            losses.append(loss)
            
            total_train_logs += 1
        
        if "Test set:" in line:
            tests.append(total_tests)
            
            val_loss = float(line.split("Average loss: ")[1].split(',')[0])
            val_losses.append(val_loss)

            cer = float(line.split("CER: ")[1].split(' ')[0])
            cers.append(cer)

            wer = float(line.split("WER: ")[1].split('\n')[0])
            wers.append(wer)

            total_tests += 1
    
def plot_train_losses():
    plt.figure(figsize=(10,6)) 

    plt.plot(train_logs, losses)
    plt.xlabel('Iterations (x1000)')
    plt.ylabel('CTC loss value')
    plt.title('Training loss')

    plt.show()

def plot_val_losses():
    plt.figure(figsize=(8,6)) 

    plt.plot(tests, val_losses, color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Average CTC loss value')
    plt.title('Average validation loss')

    plt.show()

def plot_wers():
    plt.figure(figsize=(10,6)) 

    plt.plot(tests, wers, color='darkviolet')
    plt.xlabel('Epochs')
    plt.ylabel('Average WER')
    plt.title('Average validation WER')

    plt.show()

def plot_cers():
    plt.figure(figsize=(8,6)) 

    plt.plot(tests, cers, color='hotpink')
    plt.xlabel('Epochs')
    plt.ylabel('Average CER')
    plt.title('Average validation CER')

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-t', default=False, required=False, action='store_true',
                        help='show training losses plot')
    parser.add_argument('-v', default=False, required=False, action='store_true',
                        help='show validation losses plot')
    parser.add_argument('-w', default=False, required=False, action='store_true',
                        help='show wers plot')
    parser.add_argument('-c', default=False, required=False, action='store_true',
                        help='show cers plot')

    args = parser.parse_args()

    if args.t:
        plot_train_losses()
    elif args.v:
        plot_val_losses()
    elif args.w:
        plot_wers()
    elif args.c:
        plot_cers()
