## Price-Optimization-using-Reinforcement-learning

A reinforcement learning framework for dynamic product pricing using Deep Q-Networks (DQN) and Q-Learning.
It learns optimal price adjustments for multiple retail categories — Books, Clothing, Electronics, Groceries, Health & Beauty, and Home & Kitchen to maximize profit while staying competitive.

⚙️ Built with TensorFlow / Keras, NumPy, Pandas, and Matplotlib.

**✨ Features**

📊 Multi-category dynamic pricing simulation

🧠 Implements both DQN and Tabular Q-Learning

🔁 Experience Replay + Epsilon-Greedy exploration

🪄 Reward function penalizes uncompetitive or unchanged prices

📈 Plots training vs validation loss, epoch rewards, and cumulative rewards

💾 Runs independently on synthetic datasets generated from CSVs


✅ **Requirements**

Tested on Python 3.10 + TensorFlow 2.x

pandas
numpy
matplotlib
scikit-learn
tensorflow


**Install dependencies:**

pip install -r requirements.txt

⚙️ Setup & Run
**1️⃣ Clone and open the folder**
git clone https://github.com/<yourusername>/dynamic-pricing-rl.git
cd dynamic-pricing-rl

**2️⃣ (Optional) Create and activate a virtual environment**

python -m venv .venv
.venv\Scripts\activate        # Windows

or
source .venv/bin/activate     # macOS/Linux

**3️⃣ Install dependencies**

pip install -r requirements.txt

**4️⃣ Run your experiment**

python src/dqn_model.py --dataset data/books.csv --epochs 20
python src/q_learning.py --dataset data/books.csv --epochs 20

**🧩 How It Works**

🔹 State Representation

Each product instance encodes:

[Age, Gender, ProductPrice, ProductCost, Profit, FootTraffic,
 InventoryLevel, CompetitorPrice, PurchaseMonth, PurchaseQuarter,
 DayOfWeek, ProductType, NewPrice]

🔹 Action Space

Relative price changes such as:

Books / Clothing → ±5 % or ±10 %

Electronics → ±15 % or ±20 %

Groceries / Health & Beauty → ±1 % or ±2 %

🔹 Reward Function
reward = (profit - penalty) / 2


Penalties are applied when:

Price ≈ competitor price → −5

Price ≈ original price → −3

🔹 DQN Model
Dense(64, relu)
Dense(64, relu)
Dense(len(actions), linear)


Optimizer: Adam (lr = 1e-3)
Loss: MSE

🔹 Q-Learning Baseline

Implements tabular Q-Learning with learning-rate, discount-factor, and exploration-probability tuning.

**📊 Visualization**

Loss Curves — Training vs Validation loss per epoch

Epoch Rewards — Reward trend across training iterations

Cumulative Rewards — Aggregated performance showing agent improvement

Example:

Epoch 1 → Reward ≈ 150
Epoch 20 → Reward ≈ 220 (↑ 46%)

**🧠 Insights**

The DQN outperforms tabular Q-Learning on most categories.

Small price-change environments (e.g. Groceries) converge faster.

Larger action spaces (Electronics, Home & Kitchen) yield slower but steady improvement.

**🧪 Customization**

Tune reward penalties or thresholds.

Modify action magnitudes per category.

Add demand modeling for realistic revenue feedback.

Extend DQN to Double / Dueling / Prioritized Replay variants.

Log experiments using Weights & Biases or MLflow.

**⚠️ Notes**

Each row acts as an independent environment (no temporal demand model yet).

TensorFlow may show deprecation warnings (tf.compat.v1.*) — safe to ignore.

Replace absolute Windows paths (C:\Users\...) with relative paths (e.g., data/books.csv).

**🗺️ Roadmap**

 Add revenue = price × predicted demand feedback loop

 Introduce inventory & promotion constraints

 Implement Double DQN / Dueling DQN

 Add CLI & config file support

 Dockerize for reproducibility


**👤 Author**

Vinay Mandora
