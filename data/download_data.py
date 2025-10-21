import numpy as np
import pandas as pd
import os
import requests
import pandas as pd
from pathlib import Path
import time



def download_data(year,month,typedata):
    if typedata == 'nyc':
        str_ym = str(year) + '-' + str(month).zfill(2)
        yellow_taxi_url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_" + str_ym + ".parquet"
        print(f"📥 真实数据下载链接: {yellow_taxi_url}")
        data_folder = Path("data/parquet")
        targetf1 = "yellow_tripdata_"+ str_ym + ".parquet"
        target_file = data_folder / targetf1

        print("📥 准备下载NYC Yellow Taxi真实数据...")
        print("=" * 60)
        print(f"🔗 数据源URL: {yellow_taxi_url}")
        print(f"💾 保存路径: {target_file}")

        if target_file.exists():
            file_size = target_file.stat().st_size / (1024 * 1024)
            print(f"✅ 文件已存在: {target_file.name} ({file_size:.1f} MB)")
            print("跳过下载...")
        else:
            print("🚀 开始下载... (这可能需要几分钟时间)")
            
            try:
                # 使用流式下载以处理大文件
                response = requests.get(yellow_taxi_url, stream=True)
                response.raise_for_status()
                
                # 获取文件大小
                total_size = int(response.headers.get('content-length', 0))
                total_size_mb = total_size / (1024 * 1024)
                print(f"📊 文件大小: {total_size_mb:.1f} MB")
                
                # 确保data文件夹存在
                data_folder.mkdir(exist_ok=True)
                
                # 下载文件
                downloaded = 0
                start_time = time.time()
                
                with open(target_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            
                            # 每10MB显示一次进度
                            if downloaded % (10 * 1024 * 1024) == 0 or downloaded == total_size:
                                progress = (downloaded / total_size) * 100 if total_size > 0 else 0
                                elapsed = time.time() - start_time
                                speed = downloaded / (1024 * 1024) / elapsed if elapsed > 0 else 0
                                print(f"⬇️  进度: {progress:.1f}% ({downloaded/(1024*1024):.1f}/{total_size_mb:.1f} MB) "
                                    f"速度: {speed:.1f} MB/s")
                
                elapsed = time.time() - start_time
                print(f"✅ 下载完成! 用时: {elapsed:.1f}秒")
                
                # 验证文件
                if target_file.exists():
                    file_size = target_file.stat().st_size / (1024 * 1024)
                    print(f"✅ 文件验证成功: {file_size:.1f} MB")
                else:
                    print("❌ 文件下载失败")
                    
            except requests.exceptions.RequestException as e:
                print(f"❌ 下载失败: {e}")
                print("💡 你可能需要:")
                print("   1. 检查网络连接")
                print("   2. 或手动从以下URL下载文件:")
                print(f"      {yellow_taxi_url}")
                print(f"   3. 然后保存到: {target_file}")
            except Exception as e:
                print(f"❌ 意外错误: {e}")
    else:
        print("❌ 只支持下载NYC Yellow Taxi数据。请检查输入参数。")
        print("💡 可用选项: 'nyc'")
        print("📄 示例: download_data(2023, 10, 'nyc')")


def describedata(year, month, typedata):
    if typedata == 'nyc':
        str_ym = str(year) + '-' + str(month).zfill(2)
        data_file = Path("data/parquet/yellow_tripdata_" + str_ym + ".parquet")

        try:
            # 读取parquet文件
            print(f"📖 正在读取文件: {data_file.name}")
            df_real = pd.read_parquet(data_file)
            
            print(f"✅ 数据读取成功!")
            print(f"📊 数据形状: {df_real.shape[0]:,} 行 × {df_real.shape[1]} 列")
            
            # 显示基本信息
            print(f"\n🔍 数据列信息:")
            print("-" * 40)
            for i, col in enumerate(df_real.columns, 1):
                dtype = str(df_real[col].dtype)
                non_null = df_real[col].count()
                null_pct = (1 - non_null/len(df_real)) * 100
                print(f"{i:2d}. {col:<25} {dtype:<15} 非空: {non_null:>8,} ({100-null_pct:.1f}%)")
            
            # 显示数据样本
            print(f"\n📋 数据前5行:")
            print("-" * 40)
            display_cols = ['tpep_pickup_datetime', 'tpep_dropoff_datetime', 
                        'PULocationID', 'DOLocationID', 'passenger_count', 
                        'trip_distance', 'fare_amount']
            
            available_cols = [col for col in display_cols if col in df_real.columns]
            print(df_real[available_cols].head())
            
            # 基本统计信息
            print(f"\n📈 关键字段统计:")
            print("-" * 40)
            
            if 'trip_distance' in df_real.columns:
                print(f"行程距离:")
                print(f"  平均: {df_real['trip_distance'].mean():.2f} 英里")
                print(f"  中位数: {df_real['trip_distance'].median():.2f} 英里")
                print(f"  范围: {df_real['trip_distance'].min():.2f} - {df_real['trip_distance'].max():.2f} 英里")
            
            if 'fare_amount' in df_real.columns:
                print(f"车费:")
                print(f"  平均: ${df_real['fare_amount'].mean():.2f}")
                print(f"  中位数: ${df_real['fare_amount'].median():.2f}")
                print(f"  范围: ${df_real['fare_amount'].min():.2f} - ${df_real['fare_amount'].max():.2f}")
            
            if 'passenger_count' in df_real.columns:
                print(f"乘客数:")
                passenger_counts = df_real['passenger_count'].value_counts().sort_index()
                print(f"  分布: {dict(passenger_counts.head())}")
            
            # 位置ID统计
            if 'PULocationID' in df_real.columns and 'DOLocationID' in df_real.columns:
                unique_pickup = df_real['PULocationID'].nunique()
                unique_dropoff = df_real['DOLocationID'].nunique()
                print(f"位置区域:")
                print(f"  上车区域数: {unique_pickup}")
                print(f"  下车区域数: {unique_dropoff}")
                print(f"  最热门上车区域: {df_real['PULocationID'].mode().iloc[0]} (ID)")
                print(f"  最热门下车区域: {df_real['DOLocationID'].mode().iloc[0]} (ID)")
            
            # 时间分析
            if 'tpep_pickup_datetime' in df_real.columns:
                df_real['pickup_hour'] = pd.to_datetime(df_real['tpep_pickup_datetime']).dt.hour
                df_real['pickup_day'] = pd.to_datetime(df_real['tpep_pickup_datetime']).dt.day
                
                print(f"时间分析:")
                print(f"  数据时间范围: {df_real['tpep_pickup_datetime'].min()} 至 {df_real['tpep_pickup_datetime'].max()}")
                print(f"  最繁忙小时: {df_real['pickup_hour'].mode().iloc[0]}点")
                
                hourly_counts = df_real['pickup_hour'].value_counts().sort_index()
                print(f"  小时分布 (前5): {dict(hourly_counts.head())}")
            
            print(f"\n💾 变量名: df_real (包含 {len(df_real):,} 条真实出租车记录)")
        except FileNotFoundError:
            print(f"❌ 文件未找到: {data_file.name}。请先下载数据。")
    else:
        print("❌ 只支持描述NYC Yellow Taxi数据。请检查输入参数。")
        print("💡 可用选项: 'nyc'")
        print("📄 示例: describedata(2023, 10, 'nyc')")


def cleandata(year,month,typedata):
    if typedata == 'nyc':
        str_ym = str(year) + '-' + str(month).zfill(2)
        data_file = Path("data/parquet/yellow_tripdata_" + str_ym + ".parquet")
        try:
            
            df_real = pd.read_parquet(data_file)
            df = df_real.copy()
            original_count = len(df)
            df = df[df['trip_distance'] > 0]  # 行程距离大于0
            df = df[df['fare_amount'] > 0]    # 车费大于0
            df = df[df['PULocationID'] != df['DOLocationID']]  # 上下车不同地点
            df = df[df['PULocationID'].notna() & df['DOLocationID'].notna()]  # 位置ID不为空
            
            # 移除极端异常值
            df = df[df['trip_distance'] <= 50]  # 行程距离小于50英里
            df = df[df['fare_amount'] <= 500]   # 车费小于500美元
            
            cleaned_count = len(df)
            removed_count = original_count - cleaned_count
            removed_pct = (removed_count / original_count) * 100
            
            print(f"   ✅ 清洗完成: 移除 {removed_count:,} 条异常记录 ({removed_pct:.1f}%)")
            print(f"   📊 清洗后数据: {cleaned_count:,} 条记录")
            
            # 2. 提取时间特征
            print(f"\n📅 提取时间特征...")
            
            # 转换时间格式
            df['pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
            df['dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])
            
            # 提取时间特征
            df['pickup_date'] = df['pickup_datetime'].dt.date
            df['pickup_hour'] = df['pickup_datetime'].dt.hour
            df['pickup_day_of_week'] = df['pickup_datetime'].dt.day_of_week  # 0=Monday, 6=Sunday
            df['pickup_day_name'] = df['pickup_datetime'].dt.day_name()
            
            # 判断工作日/周末
            df['is_weekend'] = df['pickup_day_of_week'].isin([5, 6])  # Saturday=5, Sunday=6
            df['day_type'] = df['is_weekend'].map({True: 'Weekend', False: 'Weekday'})
            
            # 3. 数据统计
            print(f"   ✅ 时间特征提取完成")
            
            # 时间范围
            date_range = f"{df['pickup_date'].min()} 至 {df['pickup_date'].max()}"
            print(f"   📅 数据时间范围: {date_range}")
            
            # 工作日/周末分布
            day_type_counts = df['day_type'].value_counts()
            print(f"   📊 数据分布:")
            for day_type, count in day_type_counts.items():
                pct = (count / len(df)) * 100
                print(f"      {day_type}: {count:,} 条记录 ({pct:.1f}%)")
            
            # 每日记录数统计
            daily_counts = df.groupby('pickup_date').size()
            print(f"   📈 每日记录数: 平均 {daily_counts.mean():.0f} 条 (范围: {daily_counts.min():,} - {daily_counts.max():,})")
            
            # 小时分布
            hourly_counts = df['pickup_hour'].value_counts().sort_index()
            peak_hour = hourly_counts.idxmax()
            print(f"   🕐 高峰小时: {peak_hour}点 ({hourly_counts[peak_hour]:,} 条记录)")
            
            # 位置统计
            unique_locations = df[['PULocationID', 'DOLocationID']].stack().nunique()
            unique_od_pairs = len(df.groupby(['PULocationID', 'DOLocationID']))
            print(f"   📍 唯一位置数: {unique_locations}")
            print(f"   🔄 唯一OD对数: {unique_od_pairs:,}")
            
            print(f"\n💾 预处理完成的数据保存为变量: df")
            print(f"📊 最终数据: {len(df):,} 条清洗后的出租车记录")
        except FileNotFoundError:
            print(f"❌ 文件未找到: {data_file.name}。请先下载数据。")
    else:
        print("❌ 只支持清洗NYC Yellow Taxi数据。请检查输入参数。")
        print("💡 可用选项: 'nyc'")
        print("📄 示例: cleandata(2023, 10, 'nyc')")
    return df


def calculate_distance(lat_1, lon_1, lat_2, lon_2):
    """
    计算两个经纬度之间的距离（单位：米）
    使用Haversine公式
    """
    from math import radians, sin, cos, sqrt, atan2

    R = 6371000  # 地球半径，单位为米
    phi_1 = radians(lat_1)
    phi_2 = radians(lat_2)
    delta_phi = radians(lat_2 - lat_1)
    delta_lambda = radians(lon_2 - lon_1)

    a = sin(delta_phi / 2) ** 2 + cos(phi_1) * cos(phi_2) * sin(delta_lambda / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c/1000


def generatestaticdata(year,month,df):
    str_ym = str(year) + '-' + str(month).zfill(2)
    xlsx_weekday_file = Path("data/xlsx/static/yellow_sod_weekday_" + str_ym + ".xlsx")
    xlsx_weekend_file = Path("data/xlsx/static/yellow_sod_weekend_" + str_ym + ".xlsx")
    weekday_od_np_path = Path("data/npy/static/yellow_sod_weekday_" + str_ym + ".npy")
    weekend_od_np_path = Path("data/npy/static/yellow_sod_weekend_" + str_ym + ".npy") 


    # 1. 生成工作日静态OD数据集
    print("📈 生成工作日静态OD数据集...")
    print("=" * 60)

    # 只使用2024年1月的数据确保数据一致性
    df_main = df[df['pickup_datetime'].dt.year == year].copy()
    print(f"📊 使用{year}年数据: {len(df_main):,} 条记录")

    # 重新计算工作日和周末的天数
    weekday_dates_main = df_main[df_main['day_type'] == 'Weekday']['pickup_date'].unique()
    weekend_dates_main = df_main[df_main['day_type'] == 'Weekend']['pickup_date'].unique()

    n_weekdays = len(weekday_dates_main)
    n_weekends = len(weekend_dates_main)

    print(f"📅 {year}年数据覆盖:")
    print(f"   工作日: {n_weekdays} 天")
    print(f"   周末: {n_weekends} 天")

    # 1. 重新生成工作日静态OD数据集（平均每天需求）
    print(f"\n📈 生成工作日静态OD数据集（平均每天需求）...")
    weekday_data_main = df_main[df_main['day_type'] == 'Weekday'].copy()

    # 按OD对聚合并计算平均每天需求
    weekday_od_correct = weekday_data_main.groupby(['PULocationID', 'DOLocationID']).agg({
        'fare_amount': ['count', 'sum', 'mean'],
        'trip_distance': ['sum', 'mean'],
        'passenger_count': 'sum'
    }).round(3)

    # 展平列名
    weekday_od_correct.columns = ['total_trips', 'total_fare', 'avg_fare', 'total_distance', 'avg_distance', 'total_passengers']
    weekday_od_correct = weekday_od_correct.reset_index()

    # 计算平均每天需求
    weekday_od_correct['daily_demand'] = (weekday_od_correct['total_trips'] / n_weekdays).round(2)
    weekday_od_correct['daily_fare'] = (weekday_od_correct['total_fare'] / n_weekdays).round(2)
    weekday_od_correct['daily_distance'] = (weekday_od_correct['total_distance'] / n_weekdays).round(2)
    weekday_od_correct['daily_passengers'] = (weekday_od_correct['total_passengers'] / n_weekdays).round(2)

    # 添加标识
    weekday_od_correct['day_type'] = 'Weekday'
    weekday_od_correct['n_days'] = n_weekdays

    print(f"   ✅ 工作日OD对: {len(weekday_od_correct):,} 对")
    print(f"   📊 需求统计 (基于 {n_weekdays} 个工作日):")
    print(f"      平均每天总需求: {weekday_od_correct['daily_demand'].sum():.0f} 行程/天")
    print(f"      平均OD对需求: {weekday_od_correct['daily_demand'].mean():.2f} 行程/天")
    print(f"      最高单OD日需求: {weekday_od_correct['daily_demand'].max():.0f} 行程/天")

    # 2. 重新生成周末静态OD数据集（平均每天需求）
    print(f"\n📈 生成周末静态OD数据集（平均每天需求）...")
    weekend_data_main = df_main[df_main['day_type'] == 'Weekend'].copy()

    # 按OD对聚合并计算平均每天需求
    weekend_od_correct = weekend_data_main.groupby(['PULocationID', 'DOLocationID']).agg({
        'fare_amount': ['count', 'sum', 'mean'],
        'trip_distance': ['sum', 'mean'],
        'passenger_count': 'sum'
    }).round(3)

    # 展平列名
    weekend_od_correct.columns = ['total_trips', 'total_fare', 'avg_fare', 'total_distance', 'avg_distance', 'total_passengers']
    weekend_od_correct = weekend_od_correct.reset_index()

    # 计算平均每天需求
    weekend_od_correct['daily_demand'] = (weekend_od_correct['total_trips'] / n_weekends).round(2)
    weekend_od_correct['daily_fare'] = (weekend_od_correct['total_fare'] / n_weekends).round(2)
    weekend_od_correct['daily_distance'] = (weekend_od_correct['total_distance'] / n_weekends).round(2)
    weekend_od_correct['daily_passengers'] = (weekend_od_correct['total_passengers'] / n_weekends).round(2)

    # 添加标识
    weekend_od_correct['day_type'] = 'Weekend'
    weekend_od_correct['n_days'] = n_weekends

    print(f"   ✅ 周末OD对: {len(weekend_od_correct):,} 对")
    print(f"   📊 需求统计 (基于 {n_weekends} 个周末日):")
    print(f"      平均每天总需求: {weekend_od_correct['daily_demand'].sum():.0f} 行程/天")
    print(f"      平均OD对需求: {weekend_od_correct['daily_demand'].mean():.2f} 行程/天")
    print(f"      最高单OD日需求: {weekend_od_correct['daily_demand'].max():.0f} 行程/天")


    print(f"   📊 weekday_od_correct: 工作日静态OD数据 (平均每天需求, {len(weekday_od_correct):,} 对)")
    print(f"   📊 weekend_od_correct: 周末静态OD数据 (平均每天需求, {len(weekend_od_correct):,} 对)")


    weekday_len_od_data_np = max(len(weekday_od_correct['PULocationID'].unique()), len(weekday_od_correct['DOLocationID'].unique()))
    weekend_len_od_data_np = max(len(weekend_od_correct['PULocationID'].unique()), len(weekend_od_correct['DOLocationID'].unique()))  
    weekday_od_np = np.zeros((weekday_len_od_data_np, weekday_len_od_data_np))
    weekend_od_np = np.zeros((weekend_len_od_data_np, weekend_len_od_data_np))
    print(weekday_od_correct.head())
    for i in range(len(weekday_od_correct)):
        pu_id = weekday_od_correct['PULocationID'].iloc[i]
        do_id = weekday_od_correct['DOLocationID'].iloc[i]
        daily_demand = weekday_od_correct['daily_demand'].iloc[i]
        if pu_id < weekday_len_od_data_np and do_id < weekday_len_od_data_np:
            weekday_od_np[pu_id, do_id] = daily_demand
    for i in range(len(weekend_od_correct)):
        pu_id = weekend_od_correct['PULocationID'].iloc[i]
        do_id = weekend_od_correct['DOLocationID'].iloc[i]
        daily_demand = weekend_od_correct['daily_demand'].iloc[i]
        if pu_id < weekend_len_od_data_np and do_id < weekend_len_od_data_np:
            weekend_od_np[pu_id, do_id] = daily_demand
    print(f"   📊 工作日OD矩阵形状: {weekday_od_np.shape}")
    print(f"   📊 周末OD矩阵形状: {weekend_od_np.shape}")
    print("maxdailydemand_weekday:", weekday_od_correct['daily_demand'].max())
    print("maxdailydemand_weekend:", weekend_od_correct['daily_demand'].max())
    weekend_od_correct.to_excel(xlsx_weekend_file, index=False)
    weekday_od_correct.to_excel(xlsx_weekday_file, index=False)
    np.save(weekday_od_np_path, weekday_od_np)
    np.save(weekend_od_np_path, weekend_od_np)
    weekday_distance_np = np.zeros((weekday_len_od_data_np, weekday_len_od_data_np))
    weekend_distance_np = np.zeros((weekend_len_od_data_np, weekend_len_od_data_np))
    weekday_adjacency_np = np.zeros((weekday_len_od_data_np, weekday_len_od_data_np))
    weekend_adjacency_np = np.zeros((weekend_len_od_data_np, weekend_len_od_data_np))
    
    print(f"   📍 正在计算距离矩阵和邻接矩阵...")
    distancedata = pd.read_csv("data/nyc_taxi_coordinates.csv")
    npdistance = np.zeros((len(distancedata), len(distancedata)))
    for i in range(len(distancedata)):
        for j in range(len(distancedata)):
            if i != j:
                distance = calculate_distance(distancedata['lat'].iloc[i], distancedata['lon'].iloc[i],
                                            distancedata['lat'].iloc[j], distancedata['lon'].iloc[j])
                npdistance[i, j] = distance
    print("median distance:", np.median(npdistance))
    maxtaxirange = np.max(npdistance) * 0.8
    # 计算工作日距离矩阵和邻接矩阵
    for i in range(weekday_distance_np.shape[0]):
        for j in range(weekday_distance_np.shape[1]):
            if i != j and i < len(distancedata) and j < len(distancedata):
                # 计算距离并赋值给矩阵的特定位置
                distance = calculate_distance(distancedata['lat'].iloc[i], distancedata['lon'].iloc[i],
                                            distancedata['lat'].iloc[j], distancedata['lon'].iloc[j])
                weekday_distance_np[i, j] = distance
                
                # 根据距离设置邻接矩阵
                if distance > maxtaxirange:
                    weekday_adjacency_np[i, j] = 0
                else:
                    weekday_adjacency_np[i, j] = 1
            elif i == j:
                weekday_distance_np[i, j] = 0
                weekday_adjacency_np[i, j] = 1  # 自己到自己设为1
    
    # 计算周末距离矩阵和邻接矩阵
    for i in range(weekend_distance_np.shape[0]):
        for j in range(weekend_distance_np.shape[1]):
            if i != j and i < len(distancedata) and j < len(distancedata):
                # 计算距离并赋值给矩阵的特定位置
                distance = calculate_distance(distancedata['lat'].iloc[i], distancedata['lon'].iloc[i],
                                            distancedata['lat'].iloc[j], distancedata['lon'].iloc[j])
                weekend_distance_np[i, j] = distance
                
                # 根据距离设置邻接矩阵
                if distance > maxtaxirange:
                    weekend_adjacency_np[i, j] = 0
                else:
                    weekend_adjacency_np[i, j] = 1
            elif i == j:
                weekend_distance_np[i, j] = 0
                weekend_adjacency_np[i, j] = 1  # 自己到自己设为1
    np.save("data/npy/static/yellow_sod_weekday_distance_" + str_ym + ".npy", weekday_distance_np)
    np.save("data/npy/static/yellow_sod_weekend_distance_" + str_ym + ".npy", weekend_distance_np)
    np.save("data/npy/static/yellow_sod_weekday_adjacency_" + str_ym + ".npy", weekday_adjacency_np)
    np.save("data/npy/static/yellow_sod_weekend_adjacency_" + str_ym + ".npy", weekend_adjacency_np)
    print(f"📂 工作日静态OD数据已保存到: {xlsx_weekday_file}")
    print(f"📂 周末静态OD数据已保存到: {xlsx_weekend_file}")
    print(f"📂 工作日OD矩阵已保存到: {weekday_od_np_path}")
    print(f"📂 周末OD矩阵已保存到: {weekend_od_np_path}")





if __name__ == "__main__":
    year = 2025
    month = 5
    typedata = 'nyc'
    download_data(year, month, typedata)
    describedata(year, month, typedata)
    df = cleandata(year, month, typedata)
    generatestaticdata(year, month, df)
