# backend_with_pca.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

np.random.seed(42)
n = 1000

# --- Synthetic dataset generation (logical rules) ---
ages = np.random.randint(20, 65, n)
city_type = np.random.choice([1,2,3], n, p=[0.5,0.3,0.2])
education_level = np.random.choice([1,2,3,4], n, p=[0.25,0.35,0.3,0.1])
employment_type = np.random.choice([1,2,3,4], n, p=[0.55,0.25,0.15,0.05])
marital_status = np.where(ages < 28, 0, np.where(ages < 50, 1, np.random.choice([1,2], n, p=[0.8,0.2])))

# Salary
salary = []
for i in range(n):
    base = 15000 + (ages[i] * np.random.uniform(1500, 2500))
    if city_type[i] == 1:
        base *= np.random.uniform(1.2, 1.4)
    elif city_type[i] == 2:
        base *= np.random.uniform(0.9, 1.1)
    else:
        base *= np.random.uniform(0.7, 0.9)
    base *= 1 + (education_level[i] / 10)
    if employment_type[i] == 3:
        base = np.random.randint(5000, 15000)
    elif employment_type[i] == 4:
        base = np.random.randint(8000, 30000)
    salary.append(base)
salary = np.clip(salary, 8000, 300000).astype(int)

# Dependents
dependents = []
for i in range(n):
    if marital_status[i] == 0:
        dependents.append(0)
    elif marital_status[i] == 1:
        dep = np.random.choice([1,2,3,4], p=[0.3,0.4,0.2,0.1])
        dependents.append(dep)
    else:
        dependents.append(np.random.choice([0,1,2], p=[0.6,0.3,0.1]))
dependents = np.array(dependents)

# Risk level
risk_level = []
for i in range(n):
    risk = 3 - (ages[i] / 25) + np.random.normal(0,0.5)
    if dependents[i] >= 3:
        risk -= 0.8
    if employment_type[i] in [3, 4]:
        risk -= 0.5
    if city_type[i] == 1:
        risk += 0.3
    if marital_status[i] == 1:
        risk -= 0.2
    risk_level.append(int(np.clip(round(risk), 1, 3)))
risk_level = np.array(risk_level)

# Financial knowledge
financial_knowledge = []
for i in range(n):
    base = 1.5
    if city_type[i] == 1:
        base += np.random.uniform(0.4, 1.0)
    elif city_type[i] == 2:
        base += np.random.uniform(0.1, 0.6)
    else:
        base += np.random.uniform(-0.2, 0.3)
    if risk_level[i] == 3:
        base += 0.3
    elif risk_level[i] == 1:
        base -= 0.3
    if employment_type[i] == 2:
        base += 0.2
    if employment_type[i] == 3:
        base -= 0.4
    financial_knowledge.append(int(np.clip(round(base), 1, 3)))
financial_knowledge = np.array(financial_knowledge)

# Monthly expenses and savings
monthly_expenses = []
current_savings = []
for i in range(n):
    exp = salary[i] * np.random.uniform(0.35, 0.55)
    exp *= 1 + (dependents[i] * 0.1)
    if city_type[i] == 1:
        exp *= 1.3
    elif city_type[i] == 3:
        exp *= 0.8
    if marital_status[i] == 1:
        exp *= 1.1
    monthly_expenses.append(int(np.clip(exp, 8000, salary[i]*0.9)))

    if employment_type[i] == 3:
        save = salary[i] * np.random.uniform(0.1, 0.3)
    elif employment_type[i] == 4:
        save = salary[i] * np.random.uniform(0.8, 1.2)
    else:
        save = (ages[i]/100) * salary[i] * np.random.uniform(0.5, 1.2)
    if dependents[i] > 2:
        save *= 0.7
    current_savings.append(int(np.clip(save, 1000, 2000000)))

monthly_expenses = np.array(monthly_expenses)
current_savings = np.array(current_savings)

# Loans and goals
loan_amount = []
for i in range(n):
    if current_savings[i] < 100000 and salary[i] > 50000:
        amt = np.random.uniform(100000, 600000)
    elif dependents[i] >= 3:
        amt = np.random.uniform(50000, 300000)
    else:
        amt = np.random.uniform(0, 200000)
    loan_amount.append(int(amt))
has_loans = (np.array(loan_amount) > 0).astype(int)

investment_goal = np.where(ages < 30, 3, np.where(ages < 45, 2, 1))
emergency_fund_ratio = np.clip(current_savings / monthly_expenses, 0, 50)

# Allocation logic
stocks = np.clip((0.05 + (risk_level*0.12) + (financial_knowledge*0.06) - (ages/200) + (city_type==1)*0.05), 0.05, 0.7)
mutual_funds = np.clip((0.3 - (risk_level/10) + (financial_knowledge/20) + (city_type==2)*0.04 + (city_type==3)*0.02), 0.1, 0.45)
gold_silver = np.clip(0.15 + (ages/300) - (risk_level/20) + (city_type==3)*-0.03, 0.03, 0.20)
bonds = np.clip(1 - (stocks + mutual_funds + gold_silver), 0.05, 0.6)
total = stocks + mutual_funds + gold_silver + bonds

df = pd.DataFrame({
    'age': ages,
    'salary': salary,
    'city_type': city_type,
    'education_level': education_level,
    'employment_type': employment_type,
    'marital_status': marital_status,
    'dependents': dependents,
    'risk_level': risk_level,
    'financial_knowledge': financial_knowledge,
    'monthly_expenses': monthly_expenses,
    'current_savings': current_savings,
    'loan_amount': loan_amount,
    'has_loans': has_loans,
    'investment_goal': investment_goal,
    'emergency_fund_ratio': emergency_fund_ratio,
    'stocks': stocks / total,
    'mutual_funds': mutual_funds / total,
    'gold_silver': gold_silver / total,
    'bonds': bonds / total
})

df.to_csv("realistic_investment_dataset_v4.csv", index=False)
print("Dataset saved: realistic_investment_dataset_v4.csv")

# Preprocess, PCA, train
X = df.drop(columns=['stocks','mutual_funds','gold_silver','bonds'])
y = df[['stocks','mutual_funds','gold_silver','bonds']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=0.95, svd_solver='full')  # keep 95% variance
X_pca = pca.fit_transform(X_scaled)

X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=400, max_depth=14, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"Model trained. R2: {r2:.3f}, MAE: {mae:.4f}")

joblib.dump(rf, "rf_investment_model.pkl")
joblib.dump(scaler, "rf_scaler.pkl")
joblib.dump(pca, "rf_pca.pkl")
print("Saved rf_investment_model.pkl, rf_scaler.pkl, rf_pca.pkl")
