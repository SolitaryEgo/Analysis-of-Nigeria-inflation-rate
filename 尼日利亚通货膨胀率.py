import pandas as pd
import numpy as np
import plotly.express as px
from pyecharts import options as opts
from pyecharts.charts import Line,Scatter
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

df = pd.read_csv('./NigeriaInflationRates.csv')
print(df.head())

print(df.isna().sum())
df['Crude Oil Price'] = df['Crude Oil Price'].fillna(df['Crude Oil Price'].mean())
df['Production'] = df['Production'].fillna(df['Production'].median())
df['Crude Oil Export'] = df['Crude Oil Export'].fillna(df['Crude Oil Export'].mean())
print(df.isna().sum())

corr_df = df.select_dtypes(include=[np.number]).corr().round(2)
fig = px.imshow(corr_df,text_auto=True,aspect='auto')
fig.update_layout(width=1600,height=900)
fig.update_xaxes(side='top')
fig.show()


df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str))
df = df.sort_values(by='Date')


line = (
    Line()
    .add_xaxis(df['Date'].dt.strftime('%Y-%m').tolist())  # 将日期格式化为 "Year-Month"
    .add_yaxis("Inflation Rate", df['Inflation_Rate'].tolist(), is_smooth=True)
    .set_global_opts(
        title_opts=opts.TitleOpts(title="尼日利亚历年通货膨胀率"),
        xaxis_opts=opts.AxisOpts(type_="category", name="日期"),
        yaxis_opts=opts.AxisOpts(name="通货膨胀率 (%)"),
        tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross")
    )
)

line.render('尼日利亚历年通货膨胀率.html')

x = df[['Crude Oil Price','Production','Crude Oil Export','CPI_Food','CPI_Energy','CPI_Health','CPI_Transport','CPI_Communication','CPI_Education']]
y = df['Inflation_Rate']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=42)

model = LinearRegression()
model.fit(x_train,y_train)

model2 = RandomForestRegressor()
model2.fit(x_train,y_train)

y_pred = model.predict(x_test)
y2_pred = model2.predict(x_test)

mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)
print(f'线性回归 mse:{mse} , r2:{r2 * 100:.2f}%')

mse2 = mean_squared_error(y_test,y2_pred)
r22 = r2_score(y_test,y2_pred)
print(f'随机森林 mse:{mse2} , r2:{r22 * 100:.2f}%')

true_values = y_test.tolist()
predicted_values = y_pred.tolist()

line_chart = (
    Line(init_opts=opts.InitOpts(width="1000px", height="600px"))
    .add_xaxis(list(range(len(true_values))))
    .add_yaxis("True Values", true_values, is_smooth=True, label_opts=opts.LabelOpts(is_show=False))
    .add_yaxis("Predicted Values", predicted_values, is_smooth=True, label_opts=opts.LabelOpts(is_show=False))
    .set_global_opts(
        title_opts=opts.TitleOpts(title="线性回归 真实通胀率与预测通胀率对比"),
        xaxis_opts=opts.AxisOpts(name="样本索引"),
        yaxis_opts=opts.AxisOpts(name="通货膨胀率"),
        tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
        toolbox_opts=opts.ToolboxOpts(),
        datazoom_opts=[opts.DataZoomOpts(), opts.DataZoomOpts(type_="inside")]
    )
)

line_chart.render('尼日利亚历通货膨胀率预测线性回归.html')

true_values = y_test.tolist()
predicted_values_2 = y2_pred.tolist()

line_chart2 = (
    Line(init_opts=opts.InitOpts(width="1000px", height="600px"))
    .add_xaxis(list(range(len(true_values))))
    .add_yaxis("True Values", true_values, is_smooth=True, label_opts=opts.LabelOpts(is_show=False))
    .add_yaxis("Predicted Values", predicted_values_2, is_smooth=True, label_opts=opts.LabelOpts(is_show=False))
    .set_global_opts(
        title_opts=opts.TitleOpts(title="随机森林 真实通胀率与预测通胀率对比"),
        xaxis_opts=opts.AxisOpts(name="样本索引"),
        yaxis_opts=opts.AxisOpts(name="通货膨胀率"),
        tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
        toolbox_opts=opts.ToolboxOpts(),
        datazoom_opts=[opts.DataZoomOpts(), opts.DataZoomOpts(type_="inside")]
    )
)

line_chart2.render('尼日利亚历通货膨胀率预测随机森林.html')


features = df[['Crude Oil Price','Production','Crude Oil Export','CPI_Food','CPI_Energy','CPI_Health','CPI_Transport','CPI_Communication','CPI_Education']]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=3,random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_features)

scatter_chart = (
    Scatter(init_opts=opts.InitOpts(width="1000px", height="600px"))
    .add_xaxis(df['CPI_Food'].tolist())
    .add_yaxis("Cluster 0", df[df['Cluster'] == 0]['Inflation_Rate'].tolist(), label_opts=opts.LabelOpts(is_show=False))
    .add_yaxis("Cluster 1", df[df['Cluster'] == 1]['Inflation_Rate'].tolist(), label_opts=opts.LabelOpts(is_show=False))
    .add_yaxis("Cluster 2", df[df['Cluster'] == 2]['Inflation_Rate'].tolist(), label_opts=opts.LabelOpts(is_show=False))
    .set_global_opts(
        title_opts=opts.TitleOpts(title="通货膨胀率的 KMeans 聚类方法"),
        xaxis_opts=opts.AxisOpts(name="食品消费价格指数"),
        yaxis_opts=opts.AxisOpts(name="通货膨胀率"),
        tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
        toolbox_opts=opts.ToolboxOpts(),
        datazoom_opts=[opts.DataZoomOpts(), opts.DataZoomOpts(type_="inside")]
    )
)

scatter_chart.render('尼日利亚历通货膨胀率预测Kmeans.html')

