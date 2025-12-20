#!/usr/bin/env python3
"""
测试Q-value分析工具
"""

import sys
import os
sys.path.append('src')

def test_q_value_analyzer():
    """测试Q-value分析器"""
    print("🚀 Testing Q-Value Analyzer...")
    
    # 导入分析器
    try:
        from analyze_q_values import QValueAnalyzer
        analyzer = QValueAnalyzer()
        print("✅ Q-Value Analyzer imported successfully")
    except Exception as e:
        print(f"❌ Failed to import analyzer: {e}")
        return False
    
    # 检查是否有训练数据集文件
    dataset_dir = "results/training_datasets"
    if os.path.exists(dataset_dir):
        dataset_files = [f for f in os.listdir(dataset_dir) if f.endswith(('.pkl', '.json'))]
        print(f"📁 Found {len(dataset_files)} dataset files")
        
        if dataset_files:
            # 分析最新的文件
            latest_file = max(dataset_files, key=lambda x: os.path.getctime(os.path.join(dataset_dir, x)))
            file_path = os.path.join(dataset_dir, latest_file)
            print(f"📊 Analyzing: {latest_file}")
            
            try:
                df = analyzer.run_analysis_from_file(file_path)
                print("✅ Analysis completed successfully")
                return True
            except Exception as e:
                print(f"❌ Analysis failed: {e}")
                return False
        else:
            print("⚠️  No dataset files found - run training first to generate data")
    else:
        print("⚠️  Dataset directory not found - run training first")
    
    return False

def create_sample_experience_data():
    """创建示例experience数据用于测试"""
    import json
    import os
    from datetime import datetime
    
    print("🔧 Creating sample experience data for testing...")
    
    # 创建目录
    os.makedirs("results/training_datasets", exist_ok=True)
    
    # 生成示例数据
    sample_experiences = []
    
    # 模拟Q-value问题：assign动作奖励普遍较低
    for i in range(200):
        # Idle动作 - 较高奖励
        if i % 3 == 0:
            exp = {
                'vehicle_id': i % 5,
                'action_type': 'idle',
                'vehicle_location': i % 100,
                'target_location': i % 100,
                'battery_level': 0.8,
                'current_time': i * 5.0,
                'reward': -0.1 + (0.3 if i % 5 == 0 else 0),  # 大多数idle奖励较好
                'next_vehicle_location': i % 100,
                'next_battery_level': 0.8,
                'num_requests': 10,
                'request_value': 0.0,
                'is_idle': True
            }
        # Assign动作 - 较低奖励
        elif i % 3 == 1:
            distance = abs((i % 100) - ((i + 15) % 100))
            exp = {
                'vehicle_id': i % 5,
                'action_type': f'assign_{i % 10}',
                'vehicle_location': i % 100,
                'target_location': (i + 15) % 100,
                'battery_level': 0.7,
                'current_time': i * 5.0,
                'reward': -0.5 - (distance * 0.1),  # 距离越远惩罚越大
                'next_vehicle_location': (i + 15) % 100,
                'next_battery_level': 0.6,
                'num_requests': 8,
                'request_value': 5.0 + (i % 10),
                'is_idle': False
            }
        # Charge动作
        else:
            exp = {
                'vehicle_id': i % 5,
                'action_type': f'charge_{i % 3}',
                'vehicle_location': i % 100,
                'target_location': (i + 5) % 100,
                'battery_level': 0.3,
                'current_time': i * 5.0,
                'reward': -0.2,  # 充电固定小幅负奖励
                'next_vehicle_location': (i + 5) % 100,
                'next_battery_level': 1.0,
                'num_requests': 12,
                'request_value': 0.0,
                'is_idle': False
            }
        
        sample_experiences.append(exp)
    
    # 保存示例数据
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sample_data = {
        'timestamp': timestamp,
        'current_time': 1000.0,
        'dataset_size': len(sample_experiences),
        'experiences': sample_experiences,
        'environment_info': {
            'grid_size': 10,
            'num_vehicles': 5,
            'num_charging_stations': 3
        }
    }
    
    # 保存为JSON
    json_file = f"results/training_datasets/sample_dataset_{timestamp}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Sample data created: {json_file}")
    print(f"📊 Contains {len(sample_experiences)} experiences")
    
    return json_file

if __name__ == "__main__":
    print("🧪 Q-Value Analysis Test Suite")
    print("=" * 40)
    
    # 首先尝试分析现有数据
    success = test_q_value_analyzer()
    
    if not success:
        # 如果没有数据，创建示例数据进行测试
        print("\n🔧 No existing data found, creating sample data...")
        sample_file = create_sample_experience_data()
        
        print("\n🚀 Running analysis with sample data...")
        try:
            from analyze_q_values import QValueAnalyzer
            analyzer = QValueAnalyzer()
            df = analyzer.run_analysis_from_file(sample_file)
            print("✅ Sample data analysis completed!")
        except Exception as e:
            print(f"❌ Sample analysis failed: {e}")
    
    print("\n" + "=" * 40)
    print("🏁 Test completed!")