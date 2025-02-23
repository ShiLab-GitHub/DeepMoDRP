import pandas as pd

# 读取包含药物和细胞系反应IC50值的表格，假设文件名为data.csv
data = pd.read_csv('80cell_line_ic50.csv')

# 获取表格中存在的药物和细胞系组合
existing_combinations = set(zip(data['Drug Name'], data['Cell Line Name']))

# 获取所有可能的药物和细胞系组合
all_combinations = [(drug, cell_line) for drug in data['Drug Name'].dropna().unique() for cell_line in data['Cell Line Name'].dropna().unique()]

# 找到缺失的药物和细胞系组合
missing_combinations = set(all_combinations) - existing_combinations

# 显示缺失的组合
#print("缺失的组合：")
#for combination in missing_combinations:
#    print(combination)
missing_df = pd.DataFrame(list(missing_combinations), columns=['Drug Name', 'Cell Line Name'])
print(missing_df)
# 保存到 CSV 文件
missing_df.to_csv('missing_data.csv', index=False)