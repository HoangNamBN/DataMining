import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt

df = pd.read_csv("Bai3.csv", delimiter=";")

Time = {'Afternoon': 0, 'Morning':1,'Night':2}
Match_type = {'Friendly':0, 'Grand slam':1, 'Master':2}
Court_surface = {'Mixed':0,'Grass':1,'Hard':2,'Glay':3}
Outcome ={'N':0, 'W':1}

# map() dùng để chỉnh sửa các giá trị
df['Time'] = df['Time'].map(Time)
df['Match type'] = df['Match type'].map(Match_type)
df['Court surface'] = df['Court surface'].map(Court_surface)
df['Outcome'] = df['Outcome'].map(Outcome)

features = ['Time','Match type','Court surface']
X = df[features]
y = df['Outcome']
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)

# Đưa vào mẫu test để kiểm tra
predicted = clf.predict([[1,1,1]])
print("predicted: ", predicted)
if(predicted==1):
    print("Kết quả dự đoán chính xác")
else:
    print("Kết quả dự đoán không chính xác")

# Vẽ cây
fig = plt.figure(figsize=(50,50))
_ = tree.plot_tree(clf)
fig.savefig('CayQuyetdinhBai3.png')
