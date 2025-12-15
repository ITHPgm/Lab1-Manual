import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os

# --- 1. Setup and Initialization ---
# Set up for better visual style
sns.set(style="whitegrid")

# Define the 42 KDD Cup 99 column names
KDD_COLUMNS = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes",
    "land","wrong_fragment","urgent","hot","num_failed_logins","logged_in",
    "num_compromised","root_shell","su_attempted","num_root","num_file_creations",
    "num_shells","num_access_files","num_outbound_cmds","is_host_login",
    "is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
    "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate",
    "srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
    "dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","attack_type"
]

# Define categorical features for synthetic data generation
PROTOCOLS = ["tcp", "udp", "icmp"]
SERVICES = ["http", "ftp", "smtp", "domain_u"]
FLAGS = ["SF", "REJ", "S0"]
ATTACKS = ["normal.", "smurf.", "neptune.", "back.", "teardrop."]
NUM_ROWS = 5000
FILE_NAME = "kddcup99.csv"


# --- 2. Synthetic Data Generation and CSV Creation ---
def generate_synthetic_data(num_rows):
    """Generates synthetic KDD Cup 99-like data."""
    print(f"Generating {num_rows} rows of synthetic data...")
    data = []
    
    # Generate rows with EXACTLY 42 values
    for i in range(num_rows):
        # A simple mechanism to ensure some data is 'normal.'
        is_attack = random.random() > 0.3  # About 70% chance of being an attack
        
        row = [
            random.randint(0, 1000),                 # duration
            random.choice(PROTOCOLS),                # protocol_type
            random.choice(SERVICES),                 # service
            random.choice(FLAGS),                    # flag
            random.randint(0, 50000),                # src_bytes
            random.randint(0, 50000),                # dst_bytes
            0,                                       # land
            0,                                       # wrong_fragment
            0,                                       # urgent
            random.randint(0, 10),                   # hot
            random.randint(0, 5),                    # num_failed_logins
            random.randint(0, 1),                    # logged_in
            random.randint(0, 10),                   # num_compromised
            random.randint(0, 1),                    # root_shell
            random.randint(0, 1),                    # su_attempted
            random.randint(0, 10),                   # num_root
            random.randint(0, 5),                    # num_file_creations
            random.randint(0, 2),                    # num_shells
            random.randint(0, 2),                    # num_access_files
            0,                                       # num_outbound_cmds
            0,                                       # is_host_login
            random.randint(0, 1),                    # is_guest_login
            random.randint(1, 100),                  # count
            random.randint(1, 100),                  # srv_count
            random.random(),                         # serror_rate
            random.random(),                         # srv_serror_rate
            random.random(),                         # rerror_rate
            random.random(),                         # srv_rerror_rate
            random.random(),                         # same_srv_rate
            random.random(),                         # diff_srv_rate
            random.random(),                         # srv_diff_host_rate
            random.randint(1, 255),                  # dst_host_count
            random.randint(1, 255),                  # dst_host_srv_count
            random.random(),                         # dst_host_same_srv_rate
            random.random(),                         # dst_host_diff_srv_rate
            random.random(),                         # dst_host_same_src_port_rate
            random.random(),                         # dst_host_srv_diff_host_rate
            random.random(),                         # dst_host_serror_rate
            random.random(),                         # dst_host_srv_serror_rate
            random.random(),                         # dst_host_rerror_rate
            random.random(),                         # dst_host_srv_rerror_rate
            random.choice(ATTACKS[1:]) if is_attack else ATTACKS[0] # attack_type
        ]
        data.append(row)
        
    # Create DataFrame safely
    df = pd.DataFrame(data, columns=KDD_COLUMNS)
    
    # Save CSV (NO header like real KDD Cup 99)
    df.to_csv(FILE_NAME, index=False, header=False)
    print(f"Data saved to {FILE_NAME}. Shape: {df.shape}")
    return df

# --- 3. Data Loading and Preparation ---
def load_and_prepare_data():
    """Loads the CSV and prepares the DataFrame for analysis."""
    # Load dataset with no header
    df = pd.read_csv(FILE_NAME, header=None)
    
    # Assign column names
    df.columns = KDD_COLUMNS
    
    print("\nInitial Data Inspection (Head):")
    print(df.head())
    print(f"\nDataset size: {df.shape}")
    
    # Create the 'attack_category' column based on the original logic
    df['attack_category'] = df['attack_type'].apply(
        lambda x: 'Normal' if x == 'normal.' else 'Attack'
    )
    
    print("\nAttack Category Distribution:")
    print(df['attack_category'].value_counts())
    
    return df

# --- 4. Visualization Functions ---
def plot_top_attacks(df):
    """Plots the frequency of the top 10 attack types."""
    # Filter for attack types and count the top 10
    top_attacks = df[df['attack_category'] == 'Attack']['attack_type'].value_counts().head(10)
    
    if top_attacks.empty:
        print("\nNo attacks found to plot top 10.")
        return

    plt.figure(figsize=(10, 5))
    top_attacks.plot(kind='bar', color='darkred')
    plt.title("Top 10 Attack Types (Synthetic Data)", fontsize=14)
    plt.xlabel("Attack Type", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def plot_attack_trend(df):
    """Plots the attack trend over the length of the dataset."""
    
    # Add a 'time' column representing the index of the record
    df['time'] = range(len(df))
    
    # Group the attack data into intervals (using 1000 rows as an interval)
    # Note: The original document used 'df['time'] // 10000', which groups into 10k intervals.
    # Since our synthetic data is only 5000 rows, we use // 1000 for a visible trend.
    interval_size = 1000
    attack_trend = df[df['attack_category'] == 'Attack'].groupby(df['time'] // interval_size).size()
    
    if attack_trend.empty:
        print("\nNo attack trend to plot.")
        return

    plt.figure(figsize=(10, 5))
    attack_trend.plot(kind='line', marker='o', color='forestgreen')
    plt.title("Attack Trend Over Time (Intervals of 1000 records)", fontsize=14)
    plt.xlabel(f"Time Interval (Block of {interval_size} Records)", fontsize=12)
    plt.ylabel("Number of Attacks", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


# --- 5. Main Execution Block ---
if __name__ == "__main__":
    try:
        # Generate and save data
        generate_synthetic_data(NUM_ROWS)
        
        # Load, inspect, and prepare data
        df_processed = load_and_prepare_data()
        
        # Run visualizations
        plot_top_attacks(df_processed)
        plot_attack_trend(df_processed)
        
        print("\nAnalysis complete. Two plots have been generated.")
        
    except Exception as e:
        print(f"\nAn error occurred during execution: {e}")
        print("Please ensure you have pandas, matplotlib, and seaborn installed (e.g., pip install pandas matplotlib seaborn).")
    
    finally:
        # Clean up the generated CSV file
        if os.path.exists(FILE_NAME):
            # The file is needed for the script to re-run, so we won't delete it automatically.
            # If cleanup is desired: os.remove(FILE_NAME)
            pass
