import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

# ================= 学术期刊图表全局设置 =================
# 设置字体为 Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
# 设置字体大小
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 16  # 轴标签稍大
plt.rcParams['legend.fontsize'] = 12 # 图例稍小
# 设置坐标轴线宽
plt.rcParams['axes.linewidth'] = 1.5
# 设置刻度线向内，且加粗
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
# ========================================================

# 为 Train 和 Validation 设定固定配色 (深蓝 vs 砖红)
PHASE_COLORS = {
    'Train': '#1f77b4',       # 经典深蓝
    'Validation': '#d62728',  # 经典砖红
    'Other': '#2ca02c'        # 备用绿色
}

def smooth_curve(scalars, weight=0.85):
    """
    使用指数移动平均 (EMA) 对曲线进行平滑。
    weight 越大，曲线越平滑 (取值范围 0~0.99)
    """
    if not scalars:
        return scalars
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def group_tags(tags):
    """
    自动识别并分组 tags。
    将例如 'train/loss' 和 'val/loss' 分配到 'Loss' 组中。
    """
    groups = {}
    for tag in tags:
        lower_tag = tag.lower()
        
        # 1. 确定基础指标 (Base Metric)
        if 'loss' in lower_tag:
            base = 'Loss'
        elif 'acc' in lower_tag:
            base = 'Accuracy'
        elif 'iou' in lower_tag:
            base = 'IoU'
        else:
            # 如果不包含常规关键字，取最后一部分作为基础指标名
            base = tag.split('/')[-1].capitalize()
            
        # 2. 确定阶段 (Train / Validation)
        if 'train' in lower_tag:
            phase = 'Train'
        elif 'val' in lower_tag:
            phase = 'Validation'
        else:
            # 如果都没有，取第一部分
            phase = tag.split('/')[0].capitalize()
            if phase == base: 
                phase = 'Other'

        if base not in groups:
            groups[base] = {}
        groups[base][phase] = tag
        
    return groups

def plot_train_val_together(log_dir, output_dir="./journal_plots"):
    print(f"正在读取目录: {log_dir}")
    
    # 初始化 EventAccumulator
    ea = event_accumulator.EventAccumulator(log_dir, size_guidance={
        event_accumulator.SCALARS: 0
    })
    ea.Reload()
    
    tags = ea.Tags().get('scalars', [])
    if not tags:
        print("警告: 在这些文件中没有找到任何标量数据 (Scalars)。")
        return

    # 对读取到的 tags 进行 Train/Val 分组配对
    grouped_tags = group_tags(tags)
    print(f"成功将指标分组: {grouped_tags}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 开始按组绘图
    for base_metric, phase_dict in grouped_tags.items():
        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
        
        # 遍历该指标下的 Train 和 Validation
        for phase, original_tag in phase_dict.items():
            events = ea.Scalars(original_tag)
            steps = [event.step for event in events]
            values = [event.value for event in events]
            
            # 计算平滑后的值
            smoothed_values = smooth_curve(values, weight=0.85)
            
            # 获取颜色
            color = PHASE_COLORS.get(phase, PHASE_COLORS['Other'])
            
            # 1. 绘制原始数据的浅色背景线 (透明度 0.2)
            ax.plot(steps, values, color=color, alpha=0.2, linewidth=1.0, label='_nolegend_')
            
            # 2. 绘制平滑后的主线
            ax.plot(steps, smoothed_values, color=color, linewidth=2.0, label=phase)
        
        # 坐标轴设置
        ax.set_xlabel('Epochs (or Steps)') 
        ax.set_ylabel(base_metric)
        
        # 图例设置 (去掉图例外框)
        ax.legend(frameon=False, loc='best')
        
        # 网格线
        ax.grid(True, linestyle='--', color='gray', alpha=0.3)
        
        # 保存文件
        safe_metric_name = base_metric.replace("/", "_").replace("\\", "_")
        png_path = os.path.join(output_dir, f"TrainVal_{safe_metric_name}.png")
        pdf_path = os.path.join(output_dir, f"TrainVal_{safe_metric_name}.pdf")
        
        plt.savefig(png_path, bbox_inches='tight', dpi=600)
        plt.savefig(pdf_path, bbox_inches='tight', format='pdf')
        
        print(f"✅ 已生成学术标准图像: TrainVal_{safe_metric_name}.pdf/png")
        
        plt.close()

if __name__ == '__main__':
    # ================= 使用说明 =================
    # 把这里的 log_directory 换成你存放单次实验 tensorboard 文件的文件夹
    log_directory = './rizhi2'  
    
    plot_train_val_together(log_directory, output_dir="./journal_plots")