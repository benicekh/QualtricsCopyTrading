import os
import re
import ast
import matplotlib.pyplot as plt

# Select first valid folder path
folder_paths = [
    r"C:\Users\U338144\surfdrive - Hiepler, B. (Benjamin)@surfdrive.surf.nl\Projects\Copy Trading\Trading game",
    r"D:\Surfdrive\Projects\Copy Trading\Trading game"
]
folder_path = next((path for path in folder_paths if os.path.exists(path)), None)
if folder_path is None:
    raise FileNotFoundError("None of the specified folder paths exist.")

# Input and output setup
input_path = folder_path
output_path = os.path.join(folder_path, "CRRAPaths", "Plots")
os.makedirs(output_path, exist_ok=True)

# --- Functions ---

def calculate_increase_probs(P_list, ch=0.15, omega=1, gamma=1, q=0.15):
    def p_hat(p, z, omega, ch):
        num = (0.5 + ch)**z * (0.5 - ch)**(1 - z) * p
        denom = num + (0.5 - ch)**z * (0.5 + ch)**(1 - z) * (1 - p)
        return num / denom

    def p_update(p, z, omega, gamma, ch):
        prob = p_hat(p, z, omega, ch)
        change = q * gamma
        return (1 - change) * prob + change * (1 - prob)

    increase_probs = [0.5]
    p = 0.5
    for i in range(1, len(P_list)):
        z = 1 if P_list[i] > P_list[i - 1] else 0
        p = p_update(p, z, omega, gamma, ch)
        increase_probs.append(round(p, 4))
    return increase_probs

def price_up(p, ch=0.15):
    up = (8 * p * ch + 2 - 4 * ch) / 4
    return round(up * 100, 2)

# --- Main Loop ---

crra_path = os.path.join(folder_path, "CRRAPaths")
for filename in os.listdir(crra_path):
    if filename.endswith(".txt"):
        with open(os.path.join(crra_path, filename), 'r') as file:
            content = file.read()

        matches = re.findall(r'pricePathsCRRA\.(\w+)\s*=\s*(\[[^\]]+\])', content)
        price_data = {name: ast.literal_eval(prices) for name, prices in matches}

        for name, prices in price_data.items():
            increase_probs = calculate_increase_probs(prices)
            price_up_vals = [price_up(p) for p in increase_probs]
            x = list(range(len(prices)))

            fig, ax1 = plt.subplots(figsize=(10, 5))

            ax1.plot(x, prices, label='Price Path', color='tab:blue')
            ax1.set_xlabel("Round")
            ax1.set_ylabel("Price", color='tab:blue')
            ax1.tick_params(axis='y', labelcolor='tab:blue')

            for i, pu in enumerate(price_up_vals):
                if 48.0 <= pu <= 58.0:
                    ax1.axvspan(i - 0.5, i + 0.5, color='gray', alpha=0.2)

            ax2 = ax1.twinx()
            ax2.plot(x, price_up_vals, label='Price Up %', color='tab:orange', linestyle='--')
            ax2.set_ylabel("Price Up (%)", color='tab:orange')
            ax2.tick_params(axis='y', labelcolor='tab:orange')

            plt.title(f"{name} from {filename}")
            fig.tight_layout()

            # Save figure
            safe_name = f"{os.path.splitext(filename)[0]}_{name}.png"
            save_path = os.path.join(output_path, safe_name)
            plt.savefig(save_path)
            plt.close(fig)