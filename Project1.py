import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv("Customer Purchase dataset.csv")
print(df.head())
print(df.describe())


df.columns = df.columns.str.strip()


df['Returns'] = pd.to_numeric(df['Returns'], errors='coerce')

df['Returns'] = df['Returns'].fillna(df['Returns'].mean())




# Check if nulls remain
print(df['Returns'].isnull().sum())
print(df.isnull().sum()) 
print(df.duplicated().sum())  
final_dt = df.to_csv("customer purchase_cleaned.csv", index=False)


#top selling products
top_selling_products = df.groupby('Product Category')['Quantity'].sum().sort_values(ascending=False).head(10)
print("Top Selling Products:\n", top_selling_products)  
# Visualize top selling products
plt.figure(figsize=(12, 6))
sns.barplot(x=top_selling_products.index, y=top_selling_products.values, palette='viridis')
plt.title('Top Selling Products by Quantity')
plt.show ()



# Visualize the distribution of the target variable 

plt.figure(figsize=(6,4))
sns.countplot(x='Returns', data=df) 
plt.title('Returns Distribution')
plt.xlabel('Returns')
plt.ylabel('Count')
plt.show()


# Clean column names
df.columns = df.columns.str.strip().str.replace(' ', '_')
# Gender behavior plot: Average Total Purchase Amount by Gender
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x='Gender',y='Total_Purchase_Amount', estimator=np.mean, palette='Set2')
plt.title('Average Total Purchase Amount by Gender')
plt.xlabel('Gender')

plt.tight_layout()
plt.show()




# Set seaborn style
sns.set(style="whitegrid")

# Create visualizations

# Histogram: Total Purchase Amount
plt.figure(figsize=(8, 5))
sns.histplot(df['Total_Purchase_Amount'], bins=50, kde=True, color='orange')
plt.title('Distribution of Total Purchase Amount')
plt.xlabel('Total Purchase Amount')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Countplot: Product Category
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='Product_Category', order=df['Product_Category'].value_counts().index, palette='Set2')
plt.title('Product Category Distribution')
plt.xlabel('Product Category')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Countplot: Payment Method
plt.figure(figsize=(10, 5))
sns.countplot(data=df, x='Payment_Method', order=df['Payment_Method'].value_counts().index, palette='Set1')
plt.title('Payment Method Distribution')
plt.xlabel('Payment Method')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Heatmap: Numeric Correlation
plt.figure(figsize=(10, 6))
num_cols = ['Product_Price', 'Quantity', 'Total_Purchase_Amount', 'Customer_Age', 'Returns', 'Age', 'Churn']
sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()

# Pairplot: Purchase and Age Patterns
"""sns.pairplot(df[['Product_Price', 'Quantity', 'Total_Purchase_Amount', 'Age', 'Churn']].dropna(), hue='Churn', palette='husl')
plt.suptitle('Pairplot of Purchase Patterns by Churn Status', y=1.02)
plt.show()"""

#Linear Regression
X = df[['Total_Purchase_Amount','Product_Price','Quantity']]
y = df['Returns']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
plt.scatter(X_test['Total_Purchase_Amount'], y_test, color='red', label='Actual')
plt.scatter(X_test['Total_Purchase_Amount'], y_pred, color='blue', label='Predicted')
plt.xlabel("Total Purchase Amount")
plt.ylabel("Returns")
plt.title('Returns vs Total Purchase Amount (Linear Regression)')
plt.legend()
plt.show()
















