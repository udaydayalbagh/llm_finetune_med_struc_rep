import numpy as np
import matplotlib.pyplot as plt
import json

def plot_rewards(data: list, plot_path: str, window_size=10):
    # Compute moving average and moving standard deviation manually
    moving_mean = [np.mean(data[max(0, i-window_size+1):i+1]) for i in range(len(data))]
    moving_std = [np.std(data[max(0, i-window_size+1):i+1]) for i in range(len(data))]

    # Convert to numpy arrays for easier plotting
    moving_mean = np.array(moving_mean)
    moving_std = np.array(moving_std)

    # Plot data
    plt.figure(figsize=(10, 5))
    plt.plot(data, label="Original Data", alpha=0.5)
    plt.plot(moving_mean, label="Moving Mean", color='blue', linewidth=2)

    # Fill the area between (mean ± std deviation)
    plt.fill_between(range(len(data)),
                    moving_mean - moving_std,
                    moving_mean + moving_std,
                    color='blue', alpha=0.2, label="±1 Standard Deviation")

    # Labels and legend
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.title("Moving Mean with Standard Deviation (Without Pandas)")
    plt.legend()
    plt.savefig(plot_path)
    # plt.show()
    
with open('logs/DeepSeek-R1-Distill-Llama-8B-GRPO-Finetuned-Training-Logs.json') as f:
    d = json.load(f)
    plot_data = []
    for data in d:
        try:
            plot_data.append(data["reward"])
        except:
            pass
    plot_path = "logs/plot.png"
    plot_rewards(plot_data, plot_path)

