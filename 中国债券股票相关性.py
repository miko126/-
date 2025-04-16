import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.vecm import select_coint_rank, VECM

# 1. 股票数据 - 获取2015年后的CSI 300指数数据
print("Fetching CSI 300 Index data from Yahoo Finance...")
csi300 = yf.download('000300.SS', start='2015-01-04', end='2024-11-30')
csi300 = csi300[['Close']].reset_index()
csi300.columns = ['Date', 'Stock_Close']
csi300['Date'] = pd.to_datetime(csi300['Date'])

# 2. 加载国债收益率数据
print("Loading bond yield data from Excel...")
bond_yield_data = pd.read_excel('二维数据_中国_国债收益率_10年.xlsx')
bond_yield_data['Date'] = pd.to_datetime(bond_yield_data['Date'], errors='coerce')
bond_yield_data = bond_yield_data.dropna(subset=['Date'])
bond_yield_data.columns = ['Date', 'Bond_Close']

# 3. 数据清理和对齐
print("Cleaning and merging data...")
data = pd.merge(csi300, bond_yield_data, on='Date', how='inner')

# 4. 计算每日收益率
print("Calculating daily returns...")
data['Return_stock'] = data['Stock_Close'].pct_change()
data['Return_bond'] = data['Bond_Close'].pct_change()
data = data.dropna()

# 5. 滚动相关性分析
print("Calculating rolling correlation...")
window = 126  # 滚动窗口（约半年）
data['Rolling_Correlation'] = data['Return_stock'].rolling(window).corr(data['Return_bond'])

# 绘制滚动相关性图
plt.figure(figsize=(12, 8))
plt.plot(data['Date'], data['Rolling_Correlation'], label='Rolling Correlation (6-month)', color='blue')
plt.axhline(0, color='red', linestyle='--')
plt.title('China Stock-Bond Rolling Correlation (2015-2024)', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Correlation', fontsize=12)
plt.legend(fontsize=12)
plt.grid(alpha=0.5)
plt.tight_layout()
plt.savefig('china_stock_bond_rolling_correlation_2015_2024_small_window.png', dpi=300)
plt.show()

# 6. Johansen 协整检验
print("Performing Johansen cointegration test...")
ts_data = data[['Return_stock', 'Return_bond']].values
rank_test = select_coint_rank(ts_data, det_order=0, k_ar_diff=1, method="trace")
print("Johansen Test Results:")
print(f"Cointegration rank: {rank_test.rank}")
print("Trace Statistics:")
for i, stat in enumerate(rank_test.test_stats):
    print(f"Statistic for rank {i}: {stat}")
print("Critical Values (Trace):")
print(rank_test.crit_vals)

# 7. VECM 模型拟合
print("Fitting VECM model...")
vecm_model = VECM(data[['Return_stock', 'Return_bond']], k_ar_diff=1, coint_rank=rank_test.rank)
vecm_fitted = vecm_model.fit()
print(vecm_fitted.summary())

# 8. Impulse Response Function (IRF)
print("Plotting Impulse Response Function (IRF)...")
irf = vecm_fitted.irf(10)
irf.plot(orth=True)
plt.title('Impulse Response Function (IRF)')
plt.savefig('china_irf_plot.png', dpi=300)
plt.show()

# 9. 计算并绘制 FEVD
print("Calculating and plotting FEVD...")

# 计算 IRF 结果的累积响应作为 FEVD 近似
irf_data = irf.orth_irfs  # 获取正交化冲击响应
fevd_data = irf_data.cumsum(axis=0)  # 累积 IRF

# 打印 irf_data 和 fevd_data 的形状以及部分数据
print(f"IRF Data Shape: {irf_data.shape}")
print(f"FEVD Data Shape: {fevd_data.shape}")
print(f"FEVD Data Sample: {fevd_data[:5]}")  # 打印前5个数据点，检查结果

# 确保绘图数据的维度一致
num_periods = irf_data.shape[0]  # 获取期数作为绘图的 x 轴

# 如果没有问题，则继续绘图
plt.figure(figsize=(10, 6))
for i in range(fevd_data.shape[2]):
    plt.plot(range(1, num_periods + 1), fevd_data[:, :, i].sum(axis=1), label=f'Variable {i+1}')
plt.title('Forecast Error Variance Decomposition (FEVD)')
plt.xlabel('Periods')
plt.ylabel('Variance Explained')
plt.legend()
plt.savefig('china_fevd_plot.png', dpi=300)
plt.show()

# 10. 综合滚动相关性和动态分析
plt.figure(figsize=(12, 8))
plt.plot(data['Date'], data['Rolling_Correlation'], label='Rolling Correlation (6-month)', color='blue')
plt.axhline(0, color='red', linestyle='--')
plt.title('Rolling Correlation with Cointegration Insights')
plt.xlabel('Date')
plt.ylabel('Correlation')
plt.legend()
plt.grid(alpha=0.5)
plt.tight_layout()
plt.savefig('china_combined_analysis_plot.png', dpi=300)
plt.show()

print("Analysis complete!")