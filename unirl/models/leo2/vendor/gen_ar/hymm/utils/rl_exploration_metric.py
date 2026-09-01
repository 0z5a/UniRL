from typing import List, Optional, Tuple

import numpy as np


def exploration_score(
    rewards: List[float],
    alpha: float = 0.5
) -> float:
    """
    Calculate the exploration score for a single prompt.
    
    Exploration score = alpha × range(max-min) + (1-alpha) × std
    
    Args:
        rewards: List of rewards for a single prompt
        alpha: Weight of range in exploration score calculation (0.5 means range and std each take 50%)
    
    Returns:
        Exploration score (float)
    
    Example:
        >>> rewards = [0.5, 0.6, 0.7, 0.4, 0.55]
        >>> score = exploration_score(rewards, alpha=0.5)
    """
    arr = np.asarray(rewards, dtype=float)
    
    # Boundary case handling
    if arr.size <= 1:
        return 0.0
    
    # Calculate range and standard deviation
    reward_range = float(np.max(arr) - np.min(arr))
    reward_std = float(np.std(arr, ddof=1))  # sample standard deviation
    
    return alpha * reward_range + (1 - alpha) * reward_std


def average_exploration_score(
    list_of_rewards: List[List[float]],
    alpha: float = 0.5,
    weighted: bool = False
) -> Tuple[float, float, float]:
    """
    Calculate the average exploration score for multiple prompts.
    
    Args:
        list_of_rewards: List of rewards for multiple prompts, each element is a list of rewards
        alpha: Weight of range in exploration score calculation (0.5 means range and std each take 50%)
        weighted: Whether to weight the average exploration score by sample size (default False)
    
    Returns:
        (average exploration score, average range, average standard deviation) tuple
    
    Example:
        >>> r1 = [0.5, 0.6, 0.7, 0.4, 0.55]
        >>> r2 = [0.45, 0.48, 0.50, 0.52, 0.47]
        >>> avg_exp, avg_range, avg_std = average_exploration_score([r1, r2])
    """
    if len(list_of_rewards) == 0:
        return 0.0, 0.0, 0.0
    
    exploration_scores = []
    ranges = []
    stds = []
    weights = []
    
    for rewards in list_of_rewards:
        arr = np.asarray(rewards, dtype=float)
        
        # If sample size is less than or equal to 1, contribution is 0
        if arr.size <= 1:
            exploration_scores.append(0.0)
            ranges.append(0.0)
            stds.append(0.0)
            weights.append(len(rewards) if weighted else 1.0)
            continue
        
        # Calculate metrics
        reward_range = float(np.max(arr) - np.min(arr))
        reward_std = float(np.std(arr, ddof=1))
        exp_score = alpha * reward_range + (1 - alpha) * reward_std
        
        exploration_scores.append(exp_score)
        ranges.append(reward_range)
        stds.append(reward_std)
        weights.append(len(rewards) if weighted else 1.0)
    
    # Weighted average
    total_weight = sum(weights)
    if total_weight == 0:
        return 0.0, 0.0, 0.0
    
    avg_exploration = sum(e * w for e, w in zip(exploration_scores, weights)) / total_weight
    avg_range = sum(r * w for r, w in zip(ranges, weights)) / total_weight
    avg_std = sum(s * w for s, w in zip(stds, weights)) / total_weight
    
    return avg_exploration, avg_range, avg_std


def adaptive_noise_scale(
    current_exploration: float,
    baseline_exploration: float,
    initial_noise_scale: float = 0.7,
    min_noise_scale: float = 0.7,
    max_noise_scale: float = 2.0,
    decay_factor: float = 1.0
) -> float:
    """
    Dynamically adjust the noise scale based on the change of exploration score relative to the baseline.
    
    Strategy:
    - Exploration score decreases (convergence) → Increase noise (increase exploration, prevent premature convergence)
    - Exploration score increases (sufficient exploration) → Decrease noise (encourage utilization)
    
    Args:
        current_exploration: Current exploration score
        baseline_exploration: Initial/baseline exploration score
        initial_noise_scale: Initial noise scale
        min_noise_scale: Minimum noise scale
        max_noise_scale: Maximum noise scale (relative to initial value)
        decay_factor: Sensitivity of adjustment (default 1.0, >1 is more aggressive, <1 is more conservative)
    
    Returns:
        Adjusted noise scale
    
    Example:
        >>> # At the beginning of training
        >>> baseline = average_exploration_score(initial_rewards)[0]
        >>> noise = 1.0  # Initial noise
        >>> 
        >>> # During training
        >>> current = average_exploration_score(current_rewards)[0]
        >>> noise = adaptive_noise_scale(current, baseline, noise)
    """
    if baseline_exploration <= 0:
        return initial_noise_scale
    
    ratio = baseline_exploration / current_exploration
    adjusted_ratio = ratio ** decay_factor
    new_noise = initial_noise_scale * adjusted_ratio
    new_noise = max(min_noise_scale, min(max_noise_scale, new_noise))
    
    return new_noise


class AdaptiveNoiseController:
    """
    Adaptive noise controller, supports EMA to maintain the baseline exploration score.
    
    Example:
        >>> # Initialize
        >>> controller = AdaptiveNoiseController(
        ...     initial_noise_scale=0.7,
        ...     alpha=0.5,
        ...     ema_decay=0.9
        ... )
        >>> 
        >>> # In training loop
        >>> for epoch in range(num_epochs):
        ...     # Collect rewards of current epoch
        ...     current_rewards = [[...], [...], ...]
        ...     
        ...     # Update and get noise scale
        ...     noise_scale = controller.update(current_rewards)
        ...     
        ...     # Use noise_scale for sampling
        ...     ...
    """
    
    def __init__(
        self,
        initial_noise_scale: float = 0.7,
        min_noise_scale: float = 0.7,
        max_noise_scale: float = 2.0,
        alpha: float = 0.5,
        decay_factor: float = 1.0,
        ema_decay: float = 0.9,
        weighted: bool = False
    ):
        """
        Initialize the adaptive noise controller.
        
        Args:
            initial_noise_scale: Initial noise scale
            min_noise_scale: Minimum noise scale
            max_noise_scale: Maximum noise scale
            alpha: Weight of range in exploration score calculation (0.5 means range and std each take 50%)
            decay_factor: Sensitivity of noise adjustment (1.0 is neutral, >1 is more aggressive, <1 is more conservative)
            ema_decay: EMA decay factor (0.9 means history takes 90%, current takes 10%)
            weighted: Whether to weight the average exploration score by sample size
        """
        self.initial_noise_scale = initial_noise_scale
        self.min_noise_scale = min_noise_scale
        self.max_noise_scale = max_noise_scale
        self.alpha = alpha
        self.decay_factor = decay_factor
        self.ema_decay = ema_decay
        self.weighted = weighted
        
        self.baseline_exploration: Optional[float] = None
        self.current_noise_scale = initial_noise_scale
        self.step_count = 0
    
    def update(
        self,
        rewards: List[List[float]],
        return_stats: bool = False
    ) -> float:
        """
        Update the exploration score and return the adjusted noise scale.
        
        Args:
            rewards: Current rewards list, each element is a list of rewards for a prompt
            return_stats: Whether to return detailed statistics
        
        Returns:
            If return_stats=False, return the noise scale (float)
            If return_stats=True, return (noise_scale, stats_dict)
        """
        # Calculate the current exploration score
        current_exp, current_range, current_std = average_exploration_score(
            rewards, alpha=self.alpha, weighted=self.weighted
        )
        
        # Initialize baseline (first call)
        if self.baseline_exploration is None:
            self.baseline_exploration = current_exp
            stats = {
                'current_exploration': current_exp,
                'baseline_exploration': self.baseline_exploration,
                'noise_scale': self.current_noise_scale,
                'step': self.step_count,
                'range': current_range,
                'std': current_std
            }
            self.step_count += 1
            return (self.current_noise_scale, stats) if return_stats else self.current_noise_scale
        
        # Calculate new noise scale
        new_noise = adaptive_noise_scale(
            current_exploration=current_exp,
            baseline_exploration=self.baseline_exploration,
            initial_noise_scale=self.initial_noise_scale,
            min_noise_scale=self.min_noise_scale,
            max_noise_scale=self.max_noise_scale,
            decay_factor=self.decay_factor
        )
        
        # Update baseline (EMA)
        old_baseline = self.baseline_exploration
        self.baseline_exploration = (
            self.ema_decay * self.baseline_exploration + 
            (1 - self.ema_decay) * current_exp
        )
        
        # Update current noise scale
        self.current_noise_scale = new_noise
        self.step_count += 1
        
        # Prepare statistics
        stats = {
            'current_exploration': current_exp,
            'baseline_exploration': self.baseline_exploration,
            'old_baseline': old_baseline,
            'noise_scale': new_noise,
            'step': self.step_count,
            'range': current_range,
            'std': current_std,
            'exploration_ratio': current_exp / old_baseline if old_baseline > 0 else 1.0
        }
        
        return (new_noise, stats) if return_stats else new_noise
    
    def get_noise_scale(self) -> float:
        return self.current_noise_scale
    
    def get_baseline(self) -> Optional[float]:
        return self.baseline_exploration
    
    def reset(self):
        self.baseline_exploration = None
        self.current_noise_scale = self.initial_noise_scale
        self.step_count = 0
    
    def state_dict(self) -> dict:
        return {
            'baseline_exploration': self.baseline_exploration,
            'current_noise_scale': self.current_noise_scale,
            'step_count': self.step_count,
            'initial_noise_scale': self.initial_noise_scale,
            'min_noise_scale': self.min_noise_scale,
            'max_noise_scale': self.max_noise_scale,
            'alpha': self.alpha,
            'decay_factor': self.decay_factor,
            'ema_decay': self.ema_decay,
            'weighted': self.weighted
        }
    
    def load_state_dict(self, state_dict: dict):
        self.baseline_exploration = state_dict['baseline_exploration']
        self.current_noise_scale = state_dict['current_noise_scale']
        self.step_count = state_dict['step_count']
        self.initial_noise_scale = state_dict.get('initial_noise_scale', self.initial_noise_scale)
        self.min_noise_scale = state_dict.get('min_noise_scale', self.min_noise_scale)
        self.max_noise_scale = state_dict.get('max_noise_scale', self.max_noise_scale)
        self.alpha = state_dict.get('alpha', self.alpha)
        self.decay_factor = state_dict.get('decay_factor', self.decay_factor)
        self.ema_decay = state_dict.get('ema_decay', self.ema_decay)
        self.weighted = state_dict.get('weighted', self.weighted)


if __name__ == "__main__":
    print("=" * 80)
    print("探索性指标与自适应噪声调整测试")
    print("=" * 80)
    
    # 测试数据：模拟训练不同阶段的 rewards
    # 初始阶段（探索性高）
    initial_rewards = [
        [0.55, 0.59, 0.63, 0.41, 0.5],
        [0.5, 0.52, 0.49, 0.51, 0.54],
        [0.4, 0.45, 0.55, 0.6, 0.52]
    ]
    
    # 中期阶段（探索性下降）
    mid_rewards = [
        [0.52, 0.54, 0.56, 0.50, 0.53],
        [0.51, 0.52, 0.50, 0.51, 0.52],
        [0.48, 0.50, 0.52, 0.53, 0.51]
    ]
    
    # 后期阶段（探索性更低，收敛）
    late_rewards = [
        [0.53, 0.54, 0.55, 0.52, 0.53],
        [0.52, 0.52, 0.51, 0.52, 0.52],
        [0.51, 0.52, 0.52, 0.53, 0.52]
    ]
    
    print("\n" + "=" * 70)
    print("【模拟训练过程中的探索性变化】")
    print("=" * 70)
    
    # 计算各阶段的探索性分数
    initial_exp, _, _ = average_exploration_score(initial_rewards, alpha=0.5)
    mid_exp, _, _ = average_exploration_score(mid_rewards, alpha=0.5)
    late_exp, _, _ = average_exploration_score(late_rewards, alpha=0.5)
    
    print(f"\n初始阶段探索性分数: {initial_exp:.4f} (基准)")
    print(f"中期阶段探索性分数: {mid_exp:.4f} (相对变化: {mid_exp/initial_exp:.2%})")
    print(f"后期阶段探索性分数: {late_exp:.4f} (相对变化: {late_exp/initial_exp:.2%})")
    
    # 动态调整噪声尺度
    print("\n" + "=" * 70)
    print("【自适应噪声尺度调整】")
    print("=" * 70)
    
    initial_noise = 1.0  # 初始噪声尺度
    print(f"\n初始设置: noise_scale = {initial_noise:.4f}")
    
    # 中期调整
    mid_noise = adaptive_noise_scale(
        current_exploration=mid_exp,
        baseline_exploration=initial_exp,
        initial_noise_scale=initial_noise,
        min_noise_scale=0.1,
        max_noise_scale=2.0,
        decay_factor=1.0
    )
    print(f"\n中期阶段:")
    print(f"  探索性: {initial_exp:.4f} → {mid_exp:.4f} (下降{(1-mid_exp/initial_exp)*100:.1f}%)")
    print(f"  噪声调整: {initial_noise:.4f} → {mid_noise:.4f} (增大噪声，增加探索)")
    
    # 后期调整
    late_noise = adaptive_noise_scale(
        current_exploration=late_exp,
        baseline_exploration=initial_exp,
        initial_noise_scale=initial_noise,
        min_noise_scale=0.1,
        max_noise_scale=2.0,
        decay_factor=1.0
    )
    print(f"\n后期阶段:")
    print(f"  探索性: {initial_exp:.4f} → {late_exp:.4f} (下降{(1-late_exp/initial_exp)*100:.1f}%)")
    print(f"  噪声调整: {initial_noise:.4f} → {late_noise:.4f} (继续增大噪声，防止过早收敛)")
    
    # 不同 decay_factor 的影响
    print("\n" + "=" * 70)
    print("【不同 decay_factor 的影响】")
    print("=" * 70)
    print("decay_factor 控制噪声调整的敏感度：")
    print()
    
    for decay in [0.5, 1.0, 1.5, 2.0]:
        noise = adaptive_noise_scale(
            current_exploration=mid_exp,
            baseline_exploration=initial_exp,
            initial_noise_scale=initial_noise,
            min_noise_scale=0.1,
            max_noise_scale=2.0,
            decay_factor=decay
        )
        print(f"  decay={decay:.1f}: noise = {noise:.4f} ({'激进' if decay > 1.0 else '保守' if decay < 1.0 else '中性'}调整)")
    
    # 探索性上升的情况
    print("\n" + "=" * 70)
    print("【探索性上升场景】")
    print("=" * 70)
    
    # 模拟探索性反弹（例如策略变化）
    increased_rewards = [
        [0.6, 0.65, 0.70, 0.35, 0.45],
        [0.55, 0.58, 0.42, 0.62, 0.50],
        [0.35, 0.40, 0.60, 0.65, 0.48]
    ]
    
    increased_exp, _, _ = average_exploration_score(increased_rewards, alpha=0.5)
    increased_noise = adaptive_noise_scale(
        current_exploration=increased_exp,
        baseline_exploration=initial_exp,
        initial_noise_scale=initial_noise,
        min_noise_scale=0.1,
        max_noise_scale=2.0,
        decay_factor=1.0
    )
    
    print(f"\n探索性分数: {initial_exp:.4f} → {increased_exp:.4f} (上升{(increased_exp/initial_exp-1)*100:.1f}%)")
    print(f"噪声调整: {initial_noise:.4f} → {increased_noise:.4f} (减小噪声，已充分探索)")
    
    print("\n" + "=" * 70)
    print("测试完成！")
    print("=" * 70)
    
    print("\n【使用建议】")
    print("1. 训练初始时记录 baseline_exploration 作为基准")
    print("2. 每个 epoch/step 计算当前 exploration，动态调整 noise_scale")
    print("3. decay_factor=1.0 适合大多数场景，<1.0 更保守，>1.0 更激进")
    print("4. 设置合理的 min/max_noise_scale 避免噪声过大或过小")
    print("5. 探索性下降 → 增大噪声（探索）；探索性上升 → 减小噪声（利用）")
    
    # ========== 测试 AdaptiveNoiseController 类 ==========
    print("\n" + "=" * 80)
    print("AdaptiveNoiseController 类测试（支持EMA）")
    print("=" * 80)
    
    # 创建控制器
    controller = AdaptiveNoiseController(
        initial_noise_scale=0.7,
        min_noise_scale=0.7,
        max_noise_scale=2.0,
        alpha=0.5,
        decay_factor=1.0,
        ema_decay=0.9,  # 历史占90%，当前占10%
        weighted=False
    )
    
    # 模拟训练过程
    training_stages = [
        ("初始阶段", initial_rewards),
        ("中期阶段1", mid_rewards),
        ("中期阶段2", mid_rewards),
        ("后期阶段1", late_rewards),
        ("后期阶段2", late_rewards),
        ("探索反弹", increased_rewards),
    ]
    
    print("\n模拟训练过程（使用EMA更新baseline）：\n")
    print(f"{'阶段':<12} {'当前探索性':<12} {'EMA基准':<12} {'噪声尺度':<12} {'变化':<8}")
    print("-" * 80)
    
    for stage_name, rewards in training_stages:
        noise, stats = controller.update(rewards, return_stats=True)
        
        if 'old_baseline' in stats:
            baseline_change = stats['baseline_exploration'] - stats['old_baseline']
            change_str = f"{baseline_change:+.4f}"
        else:
            change_str = "初始化"
        
        print(f"{stage_name:<12} {stats['current_exploration']:<12.4f} "
              f"{stats['baseline_exploration']:<12.4f} {noise:<12.4f} {change_str:<8}")
    
    # 测试状态保存和恢复
    print("\n" + "=" * 80)
    print("测试状态保存和恢复")
    print("=" * 80)
    
    # 保存状态
    state = controller.state_dict()
    print(f"\n保存的状态: step={state['step_count']}, baseline={state['baseline_exploration']:.4f}, "
          f"noise={state['current_noise_scale']:.4f}")
    
    # 创建新控制器并恢复状态
    new_controller = AdaptiveNoiseController()
    new_controller.load_state_dict(state)
    print(f"恢复的状态: step={new_controller.step_count}, baseline={new_controller.get_baseline():.4f}, "
          f"noise={new_controller.get_noise_scale():.4f}")
    
    # 验证恢复的控制器行为一致
    noise1, _ = controller.update(mid_rewards, return_stats=True)
    noise2, _ = new_controller.update(mid_rewards, return_stats=True)
    print(f"\n继续训练: 原控制器noise={noise1:.4f}, 恢复控制器noise={noise2:.4f} ({'一致' if abs(noise1-noise2)<1e-6 else '不一致'})")
    
    # EMA效果对比
    print("\n" + "=" * 80)
    print("不同EMA decay的影响")
    print("=" * 80)
    
    print("\nEMA decay控制baseline更新速度（越大baseline越稳定）：\n")
    
    for ema in [0.5, 0.7, 0.9, 0.95]:
        ctrl = AdaptiveNoiseController(
            initial_noise_scale=0.7,
            ema_decay=ema,
            alpha=0.5
        )
        
        # 初始化
        ctrl.update(initial_rewards)
        init_baseline = ctrl.get_baseline()
        
        # 更新一次
        ctrl.update(late_rewards)
        final_baseline = ctrl.get_baseline()
        
        change = final_baseline - init_baseline
        print(f"  EMA={ema:.2f}: baseline {init_baseline:.4f} → {final_baseline:.4f} (变化: {change:+.4f})")
    
    print("\n" + "=" * 80)
    print("所有测试完成！")
    print("=" * 80)
    
    print("\n【AdaptiveNoiseController 使用指南】")
    print("1. 创建控制器并设置参数（initial_noise_scale, ema_decay等）")
    print("2. 训练循环中调用 controller.update(rewards) 获取噪声尺度")
    print("3. ema_decay控制baseline稳定性：0.9-0.95适合大多数场景")
    print("4. 使用 state_dict() 和 load_state_dict() 保存/恢复checkpoint")
    print("5. EMA方式使baseline平滑跟踪探索性变化，避免剧烈波动")
