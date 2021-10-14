import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt

df = pd.read_csv("Bai1.csv", delimiter=";")

Refund = {'No': 0, 'Yes':1}
Marital_Status = {'Single':0, 'Married':1}
Taxable_income = {'40K':0,'50K':1,'75K':2,'80K':3,'150K':4}
Cheate ={'No':0, 'Yes':1}

# map() dùng để chỉnh sửa các giá trị
df['Refund'] = df['Refund'].map(Refund)
df['Marital Status'] = df['Marital Status'].map(Marital_Status)
df['Taxable income'] = df['Taxable income'].map(Taxable_income)
df['Cheate'] = df['Cheate'].map(Cheate)

features = ['Refund','Marital Status','Taxable income']
X = df[features]
y = df['Cheate']
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)

# Đưa vào mẫu test để kiểm tra
predicted = clf.predict([[0,1,3]])
print("predicted: ", predicted)
if(predicted==1):
    print("Kết quả dự đoán chính xác")
else:
    print("Kết quả dự đoán không chính xác")

# Vẽ cây
fig = plt.figure(figsize=(50,50))
_ = tree.plot_tree(clf)
fig.savefig('CayQuyetdinhBai1.png')

