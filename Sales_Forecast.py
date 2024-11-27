from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd

# 加载数据
df = pd.read_csv('Video_Games_Sales_as_at_22_Dec_2016.csv')

# 查看数据基本信息和前几行
print('数据基本信息：')
df.info()

# 查看数据集行数和列数
rows, columns = df.shape

if rows < 100 and columns < 20:
    # 短表数据（行数少于100且列数少于20）查看全量数据信息
    print('数据全部内容信息：')
    print(df.to_markdown(numalign='left', stralign='left'))
else:
    # 长表数据查看数据前几行信息
    print('数据前几行内容信息：')
    print(df.head().to_markdown(numalign='left', stralign='left'))

# 提取需要的列
selected_columns = ['Platform', 'Year_of_Release', 'Genre', 'Publisher', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Critic_Score', 'Critic_Count', 'User_Score', 'User_Count', 'Rating', 'Global_Sales']
df = df[selected_columns]

# 处理空值
df = df.dropna()  

# 将文本数据转换为数字
categorical_columns = ['Platform', 'Genre', 'Publisher', 'Rating']
for column in categorical_columns:
    df[column] = LabelEncoder().fit_transform(df[column])

# 划分特征和目标变量
X = df.drop('Global_Sales', axis=1)
y = df['Global_Sales']

# 标准化特征变量
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

# 创建线性回归模型
model = LinearRegression()

# 训练模型
model.fit(X_train, y_train)

# 预测测试集结果
y_pred = model.predict(X_test)

# 模型评估
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 打印评估指标
print({'均方误差': mse, 'R方值': r2})

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 设置中文字体为黑体
plt.rcParams['font.sans-serif'] = ['SimHei']

# 创建包含两个子图的画布
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 绘制预测值与真实值的散点图
ax1.scatter(y_test, y_pred)
ax1.set_xlabel('真实值')
ax1.set_ylabel('预测值')
ax1.set_title('预测值与真实值的散点图')

# 绘制残差图
residuals = y_test - y_pred
ax2.scatter(y_pred, residuals)
ax2.set_xlabel('预测值')
ax2.set_ylabel('残差')
ax2.set_title('残差图')
ax2.axhline(y=0, color='r', linestyle='--')

# 调整子图之间的间距
plt.subplots_adjust(wspace=0.4)

# 显示图形
plt.show()