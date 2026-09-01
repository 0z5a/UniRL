import re
import torch
import loguru
from hy_parallelism.parallel_states import get_parallel_state

def _copy_torch_to_torchtitan_weights(torch_moe, torchtitan_moe):
    """
    Copy weights from Torch MoE to Torchtitan MoE, handling the SwiGLU weight mapping.
    
    This method converts Torch MoE's weight format to Torchtitan MoE's format. The main
    challenge is mapping SwiGLU weights correctly:
    
    - Torch MoE SwiGLU (with expert_plan='default'): Each expert is a HunYuanMLP with
        gate_and_up_proj.weight [intermediate_size*2, hidden_size] and down_proj.weight [hidden_size, intermediate_size]
        The gate_and_up_proj is split: first half is gate, second half is up.
        Computes: down_proj(gate * silu(up))
    - Torch MoE SwiGLU (with expert_plan='ep'): Uses HunYuanExpert with separate
        gate_proj [num_experts, intermediate_size, hidden_size],  # gate
        up_proj [num_experts, intermediate_size, hidden_size],    # up
        down_proj [num_experts, hidden_size, intermediate_size]
    - Torchtitan SwiGLU: Computes w2(silu(w1(x)) * w3(x))
    
    To make them equivalent:
    - Torchtitan w1 ← Torch gate (first half of split in 'default', gate_proj in 'ep')    [so silu(w1(x)) = silu(gate(x))]
    - Torchtitan w3 ← Torch up   (second half of split in 'default', up_proj in 'ep')     [so w3(x) = up(x)]
    - Torchtitan w2 ← Torch down_proj (down projection, same)
    
    Args:
        torch_moe: Torch MoE model instance
        torchtitan_moe: Torchtitan MoE model instance to load weights into
        
    Raises:
        NotImplementedError: If EP (Expert Parallel) size > 1 or expert_plan != 'default'
        AssertionError: If expert counts don't match
        ValueError: If state_dict keys don't match expected keys
    """
    loguru.logger.info("Converting Torch MoE state_dict to Torchtitan MoE format...")

    # Check EP size - Expert Parallel > 1 is not yet supported in this test
    ep_size = get_parallel_state().ep
    if ep_size != 1:
        raise NotImplementedError("EP > 1 not yet supported in this test")

    # Get Torch MoE state_dict
    torch_state_dict = torch_moe.state_dict()
    
    # Get Torch router/gate weight for expert selection
    # Shape: [num_experts, hidden_size] - one weight vector per expert
    torch_gate_weight = torch_state_dict['gate.wg.weight']  # [num_experts, hidden_size]
    
    # Determine expert structure
    expert_plan = torch_moe.expert_plan
    num_experts = torch_moe.num_experts
    
    # Verify expert count matches between Torch and Torchtitan
    assert num_experts == torchtitan_moe.experts.num_experts, \
        f"Expert count mismatch: Torch has {num_experts} experts, Torchtitan expects {torchtitan_moe.experts.num_experts}"

    if expert_plan == 'default':
        # Torch MoE uses ModuleList of HunYuanMLP
        # Each expert has gate_and_up_proj.weight [intermediate_size*2, hidden_size] and down_proj.weight [hidden_size, intermediate_size]
        experts = torch_moe.experts
        assert len(experts) == num_experts, f"Expected {num_experts} experts, got {len(experts)}"
        
        # Get shapes from first expert
        first_expert = experts[0]
        gate_and_up_weight = first_expert.gate_and_up_proj.weight.data  # [intermediate_size*2, hidden_size]
        down_weight = first_expert.down_proj.weight.data  # [hidden_size, intermediate_size]
        
        intermediate_size = gate_and_up_weight.shape[0] // 2  # SwiGLU doubles the intermediate size
        hidden_size = gate_and_up_weight.shape[1]
        
        loguru.logger.info(f"Torch experts (default plan): num_experts={num_experts}, intermediate_size={intermediate_size}, hidden_size={hidden_size}")
        
        # Stack weights from all experts
        # gate_and_up_proj: [intermediate_size*2, hidden_size] -> split into gate and up
        # down_proj: [hidden_size, intermediate_size]
        gate_data_list = []
        up_data_list = []
        down_data_list = []
        
        for expert_idx in range(num_experts):
            expert = experts[expert_idx]
            gate_and_up_weight = expert.gate_and_up_proj.weight.data  # [intermediate_size*2, hidden_size]
            down_weight = expert.down_proj.weight.data  # [hidden_size, intermediate_size]
            
            # Split gate_and_up_proj: first half is gate, second half is up
            gate_weight = gate_and_up_weight[:intermediate_size, :]  # [intermediate_size, hidden_size]
            up_weight = gate_and_up_weight[intermediate_size:, :]  # [intermediate_size, hidden_size]
            
            gate_data_list.append(gate_weight)
            up_data_list.append(up_weight)
            down_data_list.append(down_weight)
        
        # Stack to [num_experts, ...] format
        torch_gate_data = torch.stack(gate_data_list, dim=0)  # [num_experts, intermediate_size, hidden_size], gate
        torch_up_data = torch.stack(up_data_list, dim=0)      # [num_experts, intermediate_size, hidden_size], up
        torch_down_data = torch.stack(down_data_list, dim=0)  # [num_experts, hidden_size, intermediate_size]
        
    elif expert_plan == 'ep':
        # Torch MoE uses HunYuanExpert with separate gate_proj, up_proj, down_proj
        expert_module = torch_moe.experts
        torch_gate_data = expert_module.gate_proj.data  # [num_experts, intermediate_size, hidden_size], gate
        torch_up_data = expert_module.up_proj.data      # [num_experts, intermediate_size, hidden_size], up
        torch_down_data = expert_module.down_proj.data  # [num_experts, hidden_size, intermediate_size]
        
        intermediate_size = torch_gate_data.shape[1]
        hidden_size = torch_gate_data.shape[2]
        
        loguru.logger.info(f"Torch experts (ep plan): num_experts={num_experts}, intermediate_size={intermediate_size}, hidden_size={hidden_size}")
    else:
        raise NotImplementedError(f"expert_plan '{expert_plan}' not yet supported in this test")

    # Torchtitan SwiGLU structure:
    #   - Computes: w2(silu(w1(x)) * w3(x)) where w1(x) = input @ w1.T, w3(x) = input @ w3.T
    #   - To match Torch's gate * silu(up):
    #     * w1 should map to gate (so silu(w1(x)) = silu(gate(x)))
    #     * w3 should map to up (so w3(x) = up(x))
    #   - Therefore: Titan w1 ← Torch gate, Titan w3 ← Torch up
    
    # Build Torchtitan state_dict
    torchtitan_state_dict = {}
    
    # Copy router/gate weight (expert selection weights)
    # PyTorch Linear: nn.Linear(in_features, out_features) -> weight shape [out_features, in_features]
    # Both Torch and Torchtitan use the same shape: [num_experts, hidden_size]
    torchtitan_state_dict['router.gate.weight'] = torch_gate_weight
    
    # Copy expert weights with correct SwiGLU mapping
    # Torchtitan w1 ← Torch gate (so silu(w1(x)) matches Torch's silu(gate(x)))
    torchtitan_state_dict['experts.w1'] = torch_gate_data
    # Torchtitan w2 ← Torch down_proj (down projection, same for both)
    torchtitan_state_dict['experts.w2'] = torch_down_data
    # Torchtitan w3 ← Torch up (so w3(x) matches Torch's up(x))
    torchtitan_state_dict['experts.w3'] = torch_up_data
    
    # Log shapes for verification
    loguru.logger.info(f"Converting weights:")
    loguru.logger.info(f"  router.gate.weight: {torch_gate_weight.shape}")
    loguru.logger.info(f"  experts.w1 (gate): {torch_gate_data.shape}")
    loguru.logger.info(f"  experts.w2 (down): {torch_down_data.shape}")
    loguru.logger.info(f"  experts.w3 (up): {torch_up_data.shape}")

    # Get expected keys from Torchtitan MoE to ensure we provide all required parameters
    torchtitan_full_state_dict = torchtitan_moe.state_dict()
    torchtitan_expected_keys = set(torchtitan_full_state_dict.keys())
    torchtitan_provided_keys = set(torchtitan_state_dict.keys())
    
    # Add buffers that are initialized to zeros (they don't need to be copied from Torch)
    # tokens_per_expert: non-persistent buffer for tracking token distribution per expert
    if 'tokens_per_expert' in torchtitan_expected_keys:
        torchtitan_state_dict['tokens_per_expert'] = torchtitan_full_state_dict['tokens_per_expert']
    
    # expert_bias: persistent buffer for load balancing, initialized to zeros when load_balance_coeff is None
    # (When load balancing is disabled, expert_bias is zero-initialized)
    if 'expert_bias' in torchtitan_expected_keys:
        torchtitan_state_dict['expert_bias'] = torchtitan_full_state_dict['expert_bias']
    
    # Update provided keys after adding buffers
    torchtitan_provided_keys = set(torchtitan_state_dict.keys())
    
    # Verify all expected keys are provided
    missing_keys = torchtitan_expected_keys - torchtitan_provided_keys
    if missing_keys:
        raise ValueError(f"Missing keys in state_dict: {missing_keys}")
    
    # Verify no unexpected keys
    unexpected_keys = torchtitan_provided_keys - torchtitan_expected_keys
    if unexpected_keys:
        raise ValueError(f"Unexpected keys in state_dict: {unexpected_keys}")
    
    # Load state_dict with strict=True to ensure no missing or unexpected keys
    missing_params, unexpected_params = torchtitan_moe.load_state_dict(torchtitan_state_dict, strict=True)
    
    # With strict=True, these should be empty (all keys should match)
    if missing_params:
        raise ValueError(f"Missing parameters after load_state_dict: {missing_params}")
    if unexpected_params:
        raise ValueError(f"Unexpected parameters after load_state_dict: {unexpected_params}")
    
    loguru.logger.info("Weight copying completed successfully with load_state_dict (strict=True, no missing, no unexpected)!")


def _standard_convert_torch_moe_to_titan_moe(
    sd,
    # dcp_ckpt='/apdcephfs_zwfy/share_303937731/kevinkhwu/pretrain/ep_dcp/weights'
):
    """
    Convert Torch MoE checkpoint (with expert_plan='ep') to Torchtitan MoE format.
    
    This function handles checkpoint conversion by:
    1. Loading state_dict from DCP checkpoint
    2. Finding all MoE modules (keys may have prefixes like 'layer.0.moe')
    3. Converting weights for each MoE module:
       - {prefix}.gate.wg.weight -> {prefix}.router.gate.weight
       - {prefix}.experts.gate_proj -> {prefix}.experts.w1
       - {prefix}.experts.up_proj -> {prefix}.experts.w3
       - {prefix}.experts.down_proj -> {prefix}.experts.w2
    4. Preserving all non-MoE keys unchanged
    
    Args:
        dcp_ckpt: Path to DCP checkpoint directory
        
    Returns:
        Converted state_dict in Torchtitan format
    """
    # from hy_parallelism.checkpoint.checkpoint_manager import dcp_to_torch_state_dict, torch_state_dict_to_dcp
    
    # loguru.logger.info(f"Loading checkpoint from {dcp_ckpt}...")
    # sd = dcp_to_torch_state_dict(dcp_ckpt)
    # sd = sd['model']
    
    # Create new state_dict for converted checkpoint
    converted_sd = {}
    
    # Find all MoE prefixes by looking for gate.wg.weight keys
    # Pattern: {prefix}.gate.wg.weight or {prefix}.moe.gate.wg.weight
    moe_prefixes = set()
    
    for key in sd.keys():
        # Match patterns like:
        # - layer.0.moe.gate.wg.weight
        # - moe.gate.wg.weight
        # - model.layer.0.moe.gate.wg.weight
        if key.endswith('.gate.wg.weight'):
            # Extract prefix (everything before '.gate.wg.weight')
            prefix = key[:-len('.gate.wg.weight')]
            # Find 'moe' in the prefix to extract the MoE module prefix
            # Examples:
            # - 'moe.gate.wg.weight' -> prefix='moe' -> moe_prefix='moe'
            # - 'layer.0.moe.gate.wg.weight' -> prefix='layer.0.moe' -> moe_prefix='layer.0.moe'
            # - 'model.layer.0.moe.gate.wg.weight' -> prefix='model.layer.0.moe' -> moe_prefix='model.layer.0.moe'
            parts = prefix.split('.')
            moe_idx = None
            for i, part in enumerate(parts):
                if part == 'mlp':
                    moe_idx = i
                    break
            
            if moe_idx is not None:
                # Extract prefix up to and including 'moe'
                moe_prefix = '.'.join(parts[:moe_idx+1])
                moe_prefixes.add(moe_prefix)
            else:
                loguru.logger.warning(f"Could not find 'moe' in prefix for key {key}, skipping")
    
    loguru.logger.info(f"Found {len(moe_prefixes)} MoE module(s) with prefixes: {sorted(moe_prefixes)}")
    
    # Convert each MoE module
    for moe_prefix in sorted(moe_prefixes):
        loguru.logger.info(f"Converting MoE module with prefix: {moe_prefix}")
        
        # Find keys for this MoE module
        gate_wg_key = None
        gate_proj_key = None
        up_proj_key = None
        down_proj_key = None
        shared_mlp_gate_and_up_key = None
        shared_mlp_down_key = None
        
        for key in sd.keys():
            if not key.startswith(moe_prefix + '.'):
                continue
            
            suffix = key[len(moe_prefix) + 1:]  # Remove prefix and '.'
            
            if suffix == 'gate.wg.weight':
                gate_wg_key = key
            elif suffix == 'experts.gate_proj':
                gate_proj_key = key
            elif suffix == 'experts.up_proj':
                up_proj_key = key
            elif suffix == 'experts.down_proj':
                down_proj_key = key
            elif suffix == 'shared_mlp.gate_and_up_proj.weight':
                shared_mlp_gate_and_up_key = key
            elif suffix == 'shared_mlp.down_proj.weight':
                shared_mlp_down_key = key
        
        # Verify we found all required keys for expert_plan='ep'
        if gate_wg_key is None:
            raise ValueError(f"Missing gate.wg.weight for prefix {moe_prefix}, skipping")
        if gate_proj_key is None or up_proj_key is None or down_proj_key is None:
            raise ValueError(f"Missing expert weights for prefix {moe_prefix} (ep plan requires gate_proj, up_proj, down_proj), skipping")
        
        # Extract weights
        gate_wg_weight = sd[gate_wg_key]  # [num_experts, hidden_size]
        gate_proj_weight = sd[gate_proj_key]  # [num_experts, intermediate_size, hidden_size]
        up_proj_weight = sd[up_proj_key]  # [num_experts, intermediate_size, hidden_size]
        down_proj_weight = sd[down_proj_key]  # [num_experts, hidden_size, intermediate_size]
        
        # Log shapes for verification
        loguru.logger.info(f"  Converting weights for {moe_prefix}:")
        loguru.logger.info(f"    gate.wg.weight: {gate_wg_weight.shape}")
        loguru.logger.info(f"    experts.gate_proj: {gate_proj_weight.shape}")
        loguru.logger.info(f"    experts.up_proj: {up_proj_weight.shape}")
        loguru.logger.info(f"    experts.down_proj: {down_proj_weight.shape}")
        
        # Convert to Torchtitan format
        # Torchtitan w1 ← Torch gate_proj
        converted_sd[f"{moe_prefix}.router.gate.weight"] = gate_wg_weight
        converted_sd[f"{moe_prefix}.experts.w1"] = gate_proj_weight
        converted_sd[f"{moe_prefix}.experts.w2"] = down_proj_weight
        converted_sd[f"{moe_prefix}.experts.w3"] = up_proj_weight
        
        loguru.logger.info(f"  Converted to:")
        loguru.logger.info(f"    router.gate.weight: {gate_wg_weight.shape}")
        loguru.logger.info(f"    experts.w1: {gate_proj_weight.shape}")
        loguru.logger.info(f"    experts.w2: {down_proj_weight.shape}")
        loguru.logger.info(f"    experts.w3: {up_proj_weight.shape}")
        
        # Handle shared_mlp -> shared_experts conversion
        if shared_mlp_gate_and_up_key is not None and shared_mlp_down_key is not None:
            loguru.logger.info(f"  Converting shared_mlp to shared_experts for {moe_prefix}:")
            
            # Extract shared_mlp weights
            shared_mlp_gate_and_up_weight = sd[shared_mlp_gate_and_up_key]  # [intermediate_size*2, hidden_size]
            shared_mlp_down_weight = sd[shared_mlp_down_key]  # [hidden_size, intermediate_size]
            
            loguru.logger.info(f"    shared_mlp.gate_and_up_proj.weight: {shared_mlp_gate_and_up_weight.shape}")
            loguru.logger.info(f"    shared_mlp.down_proj.weight: {shared_mlp_down_weight.shape}")
            
            # Split gate_and_up_proj: first half is gate (w1), second half is up (w3)
            # gate_and_up_proj: [intermediate_size*2, hidden_size]
            intermediate_size = shared_mlp_gate_and_up_weight.shape[0] // 2
            shared_w1_weight = shared_mlp_gate_and_up_weight[intermediate_size:, :]  # [intermediate_size, hidden_size]
            shared_w3_weight = shared_mlp_gate_and_up_weight[:intermediate_size, :]  # [intermediate_size, hidden_size]
            
            # Convert to Torchtitan format
            # Torchtitan shared_experts uses FeedForward with w1, w2, w3
            converted_sd[f"{moe_prefix}.shared_experts.w1.weight"] = shared_w1_weight
            converted_sd[f"{moe_prefix}.shared_experts.w2.weight"] = shared_mlp_down_weight
            converted_sd[f"{moe_prefix}.shared_experts.w3.weight"] = shared_w3_weight
            
            loguru.logger.info(f"  Converted shared_experts to:")
            loguru.logger.info(f"    shared_experts.w1.weight: {shared_w1_weight.shape}")
            loguru.logger.info(f"    shared_experts.w2.weight: {shared_mlp_down_weight.shape}")
            loguru.logger.info(f"    shared_experts.w3.weight: {shared_w3_weight.shape}")
        elif shared_mlp_gate_and_up_key is not None or shared_mlp_down_key is not None:
            raise ValueError(f"Found partial shared_mlp keys for prefix {moe_prefix}, skipping shared_experts conversion")
    
    # Copy all non-MoE keys unchanged
    moe_keys_to_skip = set()
    for moe_prefix in moe_prefixes:
        for key in sd.keys():
            if key.startswith(moe_prefix + '.'):
                moe_keys_to_skip.add(key)
    
    for key in sd.keys():
        if key not in moe_keys_to_skip:
            converted_sd[key] = sd[key]
    
    loguru.logger.info(f"Checkpoint conversion completed. Converted {len(moe_prefixes)} MoE module(s).")
    loguru.logger.info(f"Original checkpoint had {len(sd)} keys, converted checkpoint has {len(converted_sd)} keys.")

    # from pathlib import Path
    # torch_state_dict_to_dcp(Path(dcp_ckpt).parent.parent/'ep_dcp_cvt2titan'/'weights', sd_input=converted_sd)
    return converted_sd


def convert_torch_to_titan_moe(torch_sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Convert Torch MoE state_dict to Torchtitan MoE format.
    
    This function converts Torch MoE checkpoint keys to Torchtitan format:
    - {prefix}.gate.wg.weight -> {prefix}.router.gate.weight
    - {prefix}.experts.gate_proj -> {prefix}.experts.w1
    - {prefix}.experts.up_proj -> {prefix}.experts.w3
    - {prefix}.experts.down_proj -> {prefix}.experts.w2
    - {prefix}.shared_mlp.gate_and_up_proj.weight -> split into w1 and w3
    - {prefix}.shared_mlp.down_proj.weight -> {prefix}.shared_experts.w2.weight
    
    Non-MoE keys are preserved unchanged.
    
    Args:
        torch_sd: Torch MoE state_dict
        
    Returns:
        Converted state_dict in Torchtitan format
    """
    moe_patterns = [
        (r'^(.+)\.gate\.wg\.weight$', r'\1.router.gate.weight'),
        (r'^(.+)\.experts\.gate_proj$', r'\1.experts.w1'),
        (r'^(.+)\.experts\.up_proj$', r'\1.experts.w3'),
        (r'^(.+)\.experts\.down_proj$', r'\1.experts.w2'),
        (r'^(.+)\.shared_mlp\.down_proj\.weight$', r'\1.shared_experts.w2.weight'),
    ]
    
    gate_up_pattern = re.compile(r'^(.+)\.shared_mlp\.gate_and_up_proj\.weight$')
    
    converted_sd = {}
    
    for key, weight in torch_sd.items():
        # Handle gate_and_up_proj (needs splitting)
        if match := gate_up_pattern.match(key):
            prefix = match.group(1)
            m = weight.shape[0] // 2
            converted_sd[f"{prefix}.shared_experts.w1.weight"] = weight[m:]
            converted_sd[f"{prefix}.shared_experts.w3.weight"] = weight[:m]
            continue
        
        # Handle other MoE patterns
        for pattern, replacement in moe_patterns:
            if match := re.match(pattern, key):
                converted_sd[match.expand(replacement)] = weight
                break
        else:
            # Non-MoE keys: copy directly
            converted_sd[key] = weight
    
    return converted_sd


def convert_ptm_to_titan_moe(ptm_sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Convert PTM MoE state_dict to Torchtitan MoE format.
    
    This function handles checkpoint conversion by:
    1. Matching PTM MoE keys using regex patterns
    2. Converting weights for each MoE module:
       - {prefix}.deepspeed_moe.gate.wg.weight -> {prefix}.router.gate.weight
       - {prefix}.deepspeed_moe.experts.deepspeed_experts.group_h_to_4h -> 
         split into {prefix}.experts.w1 (gate, second half) and {prefix}.experts.w3 (up, first half)
       - {prefix}.deepspeed_moe.experts.deepspeed_experts.group_4h_to_h -> {prefix}.experts.w2
    3. Preserving all non-MoE keys unchanged
    
    PTM SwiGLU mapping:
    - PTM computes: w2(up * silu(gate))
    - group_h_to_4h: [num_experts, intermediate_size*2, hidden_size]
      * First half [:, :intermediate_size, :] is up projection
      * Second half [:, intermediate_size:, :] is gate projection
    - Torchtitan computes: w2(silu(w1(x)) * w3(x))
    - Mapping: w1 ← gate (second half), w3 ← up (first half), w2 ← group_4h_to_h
    
    Args:
        ptm_sd: PTM MoE state_dict
        
    Returns:
        Converted state_dict in Torchtitan format
    """
    moe_patterns = [
        (r'^(.+)\.deepspeed_moe\.gate\.wg\.weight$', r'\1.router.gate.weight'),
        (r'^(.+)\.deepspeed_moe\.experts\.deepspeed_experts\.group_4h_to_h$', r'\1.experts.w2'),
    ]
    
    # Pattern for group_h_to_4h which needs splitting
    group_h_to_4h_pattern = re.compile(r'^(.+)\.deepspeed_moe\.experts\.deepspeed_experts\.group_h_to_4h$')
    
    converted_sd = {}
    
    for key, weight in ptm_sd.items():
        # Handle group_h_to_4h (needs splitting)
        # Shape: [num_experts, intermediate_size*2, hidden_size]
        # First half is up (w3), second half is gate (w1)
        if match := group_h_to_4h_pattern.match(key):
            prefix = match.group(1)
            m = weight.shape[1] // 2  # Split along intermediate_size dimension
            # First half is up (w3), second half is gate (w1)
            converted_sd[f"{prefix}.experts.w3"] = weight[:, :m, :]  # up projection
            converted_sd[f"{prefix}.experts.w1"] = weight[:, m:, :]  # gate projection
            continue
        
        # Handle other MoE patterns
        for pattern, replacement in moe_patterns:
            if match := re.match(pattern, key):
                converted_sd[match.expand(replacement)] = weight
                break
        else:
            # Non-MoE keys: copy directly
            converted_sd[key] = weight
    
    return converted_sd
