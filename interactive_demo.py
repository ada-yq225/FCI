import numpy as np
import pandas as pd
from fci_engine import fci


EXPECTED_ADJACENCIES = {
    frozenset(("X1", "A")),
    frozenset(("X2", "B")),
    frozenset(("A", "B")),
    frozenset(("A", "D")),
}


def generate_medical_data(n: int, seed: int = 42) -> pd.DataFrame:
    # 场景：医疗数据分析
    # U1 (Latent): 隐藏的先天遗传体质 (未观测)
    
    # 观测数据:
    # X1 (外用药涂抹): 仅作用于 A
    # X2 (内部理疗): 仅作用于 B
    # A  (表皮症状): <- X1, U1
    # B  (内部炎症): <- X2, U1
    # D  (发烧表现): <- A

    np.random.seed(seed)

    # 隐藏变量 (Latents)
    U1 = np.random.normal(0, 1, n)

    # 独立干预因子 (Instruments / Exogenous drivers)
    X1 = np.random.normal(0, 1, n)
    X2 = np.random.normal(0, 1, n)

    A = 0.8 * X1 + 0.8 * U1 + np.random.normal(0, 0.4, n)
    B = 0.8 * X2 + 0.8 * U1 + np.random.normal(0, 0.4, n)

    D = 0.8 * A + np.random.normal(0, 0.4, n)

    return pd.DataFrame({"X1": X1, "X2": X2, "A": A, "B": B, "D": D})


def validate_result(result):
    observed_adjacencies = {frozenset(edge) for edge in result.graph.edges()}
    missing = EXPECTED_ADJACENCIES - observed_adjacencies
    extra = observed_adjacencies - EXPECTED_ADJACENCIES

    print("\n🧪 验证结果:")
    if not missing and not extra:
        print("  PASS 骨架完全命中: X1-A, X2-B, A-B, A-D")
    else:
        print(f"  FAIL 骨架不一致: missing={missing}, extra={extra}")

    checks = [
        (
            "X1 对 A 有 arrowhead",
            result.graph.is_adjacent("X1", "A")
            and result.graph.has_arrowhead("X1", "A"),
        ),
        (
            "X2 对 B 有 arrowhead",
            result.graph.is_adjacent("X2", "B")
            and result.graph.has_arrowhead("X2", "B"),
        ),
        (
            "A-B 被识别为潜在混杂 A <-> B",
            result.graph.is_adjacent("A", "B")
            and result.graph.is_bidirected_edge("A", "B"),
        ),
        (
            "A-D 的 A 端是 tail",
            result.graph.is_adjacent("A", "D")
            and result.graph.has_tail("D", "A"),
        ),
    ]

    for description, ok in checks:
        status = "PASS" if ok else "FAIL"
        print(f"  {status} {description}")

    if result.graph.is_adjacent("A", "D"):
        edge = result.graph.edge_repr("A", "D")
        if edge == "A --> D":
            print("  PASS A-D 完全定向为 A --> D")
        else:
            print(
                "  WARN A-D 只被部分定向为 "
                f"{edge}; causal-learn reference 在该数据上通常给 A --> D"
            )


def run_experiment(n: int, description: str):
    print(f"\n{'='*55}")
    print(f"🚀 实验: {description} (样本量 N={n})")
    print(f"{'='*55}")

    df = generate_medical_data(n)

    print("🔍 FCI 引擎自动计算中...")
    result = fci(df, alpha="auto", verbose=False)

    print(f"⚙️ 引擎智能定组 Alpha = {result.config.alpha} (因为 N={n})")
    print("\n✅ 图发现完成! 边解析如下:\n")

    edges_found = []
    for x, y in result.graph.edges():
        edges_found.append(result.graph.edge_repr(x, y))

    for edge_str in sorted(edges_found):
        if "<->" in edge_str:
            print(f"  {edge_str:12}  <-- 🚨 精准命中: 发现未观测到的混杂因子 (Latent)")
        elif "->" in edge_str:
            print(f"  {edge_str:12}  <-- 🎯 有向影响 (Directed)")
        else:
            print(f"  {edge_str:12}  <-- ❔ 其它状态")

    validate_result(result)


def main():
    print("📊 医疗数据拓扑 [上帝视角]:")
    print("   1. 未知混杂 U1 -> (A, B)")
    print("   2. 工具变量 X1 -> A")
    print("   3. 工具变量 X2 -> B")
    print("   4. 直达因果 A -> D")
    print("\n📌 观测层预期骨架: X1-A, X2-B, A-B, A-D")
    print("📌 关键预期: A-B 应表现为潜在混杂 A <-> B；A-D reference 通常为 A --> D\n")

    # 分别测试完全不同的量级来触发 "auto" 分段
    run_experiment(500, "小样本急诊数据 (易欠拟合，系统容忍更高波动)")
    run_experiment(2500, "中等规模体检库 (标准分析)")
    run_experiment(8000, "国家级庞大人口数据 (极易遭遇假阳性，系统必须收紧判定)")


if __name__ == "__main__":
    main()
