import numpy as np
import matplotlib.pyplot as plt
import json

def plot_rewards(data: list, plot_path: str, window_size=20):
    # Compute moving average and moving standard deviation manually
    moving_mean = [np.mean(data[max(0, i-window_size+1):i+1]) for i in range(len(data))]
    moving_std = [np.std(data[max(0, i-window_size+1):i+1]) for i in range(len(data))]

    # Convert to numpy arrays for easier plotting
    moving_mean = np.array(moving_mean)
    moving_std = np.array(moving_std)

    # Plot data
    plt.figure(figsize=(20, 10))
    plt.plot(data, label="Loss", alpha=0.5)
    plt.plot(moving_mean, label="Loss Mean", color='blue', linewidth=2)

    # Fill the area between (mean ± std deviation)
    plt.fill_between(range(len(data)),
                    moving_mean - moving_std,
                    moving_mean + moving_std,
                    color='blue', alpha=0.2, label="Loss SD")

    # Labels and legend
    plt.rcParams.update({'font.size': 20})
    plt.xlabel("Training Steps", fontsize=28)
    plt.ylabel("Loss", fontsize=28)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title("SFT Training Loss (DeepSeek-R1 8B)", fontsize=32)
    plt.legend()
    plt.savefig(plot_path)
    # plt.show()

def plot_two_rewards(data1, data2, plot_path, window_size=20):
    # Compute moving average and moving standard deviation manually
    moving_mean_1 = [np.mean(data1[max(0, i-window_size+1):i+1]) for i in range(len(data1))]
    moving_std_1 = [np.std(data1[max(0, i-window_size+1):i+1]) for i in range(len(data1))]
    moving_mean_2 = [np.mean(data2[max(0, i-window_size+1):i+1]) for i in range(len(data2))]
    moving_std_2 = [np.std(data2[max(0, i-window_size+1):i+1]) for i in range(len(data2))]

    # Convert to numpy arrays for easier plotting
    moving_mean_1 = np.array(moving_mean_1)
    moving_std_1 = np.array(moving_std_1)
    moving_mean_2 = np.array(moving_mean_2)
    moving_std_2 = np.array(moving_std_2)

    # Plot data
    plt.figure(figsize=(20, 10))
    # plt.plot(data, label="Original Data", alpha=0.5)
    plt.plot(moving_mean_1, label="Reward Mean (GRPO)", color='blue', linewidth=2)

    # Fill the area between (mean ± std deviation)
    plt.fill_between(range(len(data1)),
                    moving_mean_1 - moving_std_1,
                    moving_mean_1 + moving_std_1,
                    color='blue', alpha=0.2, label="Reward SD (GRPO)")

    plt.plot(moving_mean_2, label="Reward Mean (SFT + GRPO)", color='green', linewidth=2)
    # Fill the area between (mean ± std deviation)
    plt.fill_between(range(len(data1)),
                    moving_mean_2 - moving_std_2,
                    moving_mean_2 + moving_std_2,
                    color='green', alpha=0.2, label="Reward SD (SFT + GRPO)")

    # Labels and legend
    plt.rcParams.update({'font.size': 20})
    plt.xlabel("Training Steps", fontsize=28)
    plt.ylabel("Reward", fontsize=28)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title("Rewards During Fine-tuning (DeepSeek-R1 8B)", fontsize=32)
    plt.legend()
    plt.savefig(plot_path)
    # plt.show()


if __name__ == "__main__":
    # with open('logs/DeepSeek-R1-Distill-Llama-8B-SFT-Training-Logs.json') as f:
    #     d = json.load(f)
    #     plot_data = []
    #     for data in d:
    #         try:
    #             plot_data.append(data["loss"])
    #         except:
    #             pass
    #     plot_path = "logs/DeepSeek-R1-Distill-Llama-8B-SFT-Training_Loss.png"
    #     plot_rewards(plot_data, plot_path,  window_size=20)

    with open('logs/DeepSeek-R1-Distill-Llama-8B-GRPO-Finetuned-Training-Logs.json') as f:
        with open('logs/DeepSeek-R1-Distill-Llama-8B-SFT-GRPO-Finetuned-2-Training-Logs.json') as f2:
            d = json.load(f)
            d2 = json.load(f2)
            plot_data_1 = []
            plot_data_2 = []
            i = 0
            for i in range(500):
                try:
                    plot_data_1.append(d[i]["reward"])
                    plot_data_2.append(d2[i]["reward"])
                    i += 1
                    if i == 1000:
                        break
                except:
                    pass
            plot_path = "logs/comparison_plot.png"
            plot_two_rewards(plot_data_1, plot_data_2, plot_path)

