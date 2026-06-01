import pandas as pd
import numpy as np
from pathlib import Path

# Note: adjusting the PYTHONPATH or making sure fci_engine is importable is important.
# I'll rely on the terminal running from the project root.
from fci_engine import fci
from fci_engine.reports import render_interactive_report

def create_custom_data(n_samples=3000):
    # Setting seed for reproducibility
    np.random.seed(42)
    
    # 真实因果结构模拟:
    # 隐藏变量 Z 同时影响 X 和 Y (引入混杂)
    # 其他变量:
    # V -> X (独立工具变量)
    # X -> Y (X 和 Y 之间由隐藏变量 Z 造成的混杂)
    # Y -> W (观察结果 Y 导致 W)
    # P -> W (独立因素导致 W)
    
    Z_latent = np.random.normal(0, 1, n_samples)
    
    V = np.random.normal(0, 1, n_samples)
    P = np.random.normal(0, 1, n_samples)
    
    X = 0.8 * V + 0.8 * Z_latent + np.random.normal(0, 0.3, n_samples)
    Y = 0.7 * X + 0.6 * Z_latent + np.random.normal(0, 0.3, n_samples)
    W = 0.9 * Y + 0.8 * P + np.random.normal(0, 0.4, n_samples)
    
    # 构建 DataFrame (只给算法暴露观测变量)
    df = pd.DataFrame({"V": V, "X": X, "Y": Y, "W": W, "P": P})
    return df

def main():
    print("1. 生成自定义测试数据 (无预定义模板)...")
    df = create_custom_data()
    print(f"   数据维度: {df.shape}")
    print(f"   观测变量: {list(df.columns)}")
    
    print("\n2. 使用 FCI 引擎进行因果推断寻找 PAG 结构...")
    # alpha="auto" 自动选择显著性水平
    result = fci(df, alpha="auto", max_cond_set_size=3)
    
    print("\n3. 分析已完成。总结信息:")
    print(result.summary())
    
    print("\n输出边的边端形式字典：")
    for x, y in result.graph.edges():
        print(f"  {result.graph.edge_repr(x, y)}")

    print("\n4. 生成可交互视图报告...")
    report_html = render_interactive_report(result, title="Custom Data CI/PAG Analysis")
    
    output_path = Path("custom_data_report.html")
    output_path.write_text(report_html, encoding="utf-8")
    
    print(f"\n✅ 报告已生成: {output_path.absolute()}")

if __name__ == "__main__":
    main()
