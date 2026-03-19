"""
GRPO (Group Relative Policy Optimization) Implementation

GRPO is a reinforcement learning algorithm for training language models that optimizes
policies by comparing groups of outputs rather than using individual value estimates.
This approach is particularly useful for preference-based optimization and reward
modeling.

Key Features:
- Group-based advantage estimation
- Importance sampling for policy updates
- KL divergence constraint for stability
- Support for multiple generations per prompt
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import List, Dict, Tuple, Optional
import numpy as np


class GRPOConfig:
    """Configuration for GRPO training."""

    def __init__(
        self,
        learning_rate: float = 1e-5,
        gamma: float = 0.99,
        group_size: int = 4,
        kl_coef: float = 0.1,
        clip_range: float = 0.2,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 1.0,
        max_epochs: int = 3,
        batch_size: int = 16,
        max_length: int = 512,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.group_size = group_size
        self.kl_coef = kl_coef
        self.clip_range = clip_range
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.max_length = max_length
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p


class GRPOPolicy(nn.Module):
    """Policy network for GRPO."""

    def __init__(self, model: nn.Module, vocab_size: int):
        super().__init__()
        self.model = model
        self.vocab_size = vocab_size

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the policy network."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        return logits

    def get_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get log probabilities for the given inputs."""
        logits = self.forward(input_ids, attention_mask)
        log_probs = F.log_softmax(logits, dim=-1)

        # Gather log probs for actual tokens
        # Shift logits and input_ids for next token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()

        # Get log probs for each token
        log_probs = F.log_softmax(shift_logits, dim=-1)
        gathered_log_probs = torch.gather(
            log_probs,
            2,
            shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        return gathered_log_probs


class GRPOOptimizer:
    """GRPO Optimizer for training language models."""

    def __init__(
        self,
        policy: GRPOPolicy,
        ref_policy: GRPOPolicy,
        config: GRPOConfig,
        device: str = "cuda",
    ):
        self.policy = policy.to(device)
        self.ref_policy = ref_policy.to(device)
        self.ref_policy.eval()  # Reference policy is frozen
        self.config = config
        self.device = device

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=config.learning_rate,
        )

    def generate_responses(
        self,
        prompts: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        num_generations: int = None,
    ) -> List[torch.Tensor]:
        """
        Generate multiple responses for each prompt.

        Args:
            prompts: Input prompt tensors [batch_size, seq_len]
            attention_mask: Attention mask for prompts
            num_generations: Number of generations per prompt

        Returns:
            List of generated response tensors
        """
        if num_generations is None:
            num_generations = self.config.group_size

        batch_size = prompts.size(0)
        all_responses = []

        with torch.no_grad():
            for _ in range(num_generations):
                # Generate responses
                outputs = self.policy.model.generate(
                    input_ids=prompts,
                    attention_mask=attention_mask,
                    max_length=self.config.max_length,
                    temperature=self.config.temperature,
                    top_k=self.config.top_k,
                    top_p=self.config.top_p,
                    do_sample=True,
                    pad_token_id=self.policy.model.config.pad_token_id,
                )
                all_responses.append(outputs)

        return all_responses

    def compute_rewards(
        self,
        prompts: torch.Tensor,
        responses: List[torch.Tensor],
        reward_fn: callable,
    ) -> List[List[float]]:
        """
        Compute rewards for generated responses.

        Args:
            prompts: Input prompts
            responses: List of generated responses
            reward_fn: Function to compute rewards (takes prompt, response -> float)

        Returns:
            List of rewards for each response
        """
        all_rewards = []

        for prompt, response_batch in zip(prompts, zip(*responses)):
            batch_rewards = []
            for response in response_batch:
                # Extract only the generated part (excluding prompt)
                generated = response[len(prompt):]
                reward = reward_fn(prompt, generated)
                batch_rewards.append(reward)
            all_rewards.append(batch_rewards)

        return all_rewards

    def compute_advantages(
        self,
        rewards: List[List[float]],
    ) -> List[List[float]]:
        """
        Compute advantages using group relative estimation.

        Args:
            rewards: List of rewards for each group of responses

        Returns:
            List of advantages for each response
        """
        advantages = []

        for group_rewards in rewards:
            group_rewards = np.array(group_rewards)
            # Group relative advantage: subtract group mean
            group_mean = np.mean(group_rewards)
            group_advantages = group_rewards - group_mean
            advantages.append(group_advantages.tolist())

        return advantages

    def compute_kl_divergence(
        self,
        prompts: torch.Tensor,
        responses: List[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute KL divergence between policy and reference policy.

        Args:
            prompts: Input prompts
            responses: Generated responses
            attention_mask: Attention mask

        Returns:
            KL divergence tensor
        """
        kl_divs = []

        for prompt, response in zip(prompts, responses):
            # Concatenate prompt and response
            full_sequence = torch.cat([prompt, response[1:]], dim=0)

            # Get log probs from policy
            policy_log_probs = self.policy.get_log_probs(
                full_sequence.unsqueeze(0),
                attention_mask,
            )

            # Get log probs from reference policy
            with torch.no_grad():
                ref_log_probs = self.ref_policy.get_log_probs(
                    full_sequence.unsqueeze(0),
                    attention_mask,
                )

            # Compute KL divergence
            kl_div = (policy_log_probs - ref_log_probs).sum(dim=-1).mean()
            kl_divs.append(kl_div)

        return torch.stack(kl_divs)

    def compute_policy_loss(
        self,
        prompts: torch.Tensor,
        responses: List[torch.Tensor],
        advantages: List[List[float]],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute GRPO policy loss.

        Args:
            prompts: Input prompts
            responses: Generated responses
            advantages: Computed advantages
            attention_mask: Attention mask

        Returns:
            Loss tensor and metrics dictionary
        """
        policy_losses = []
        kl_losses = []
        entropy_losses = []

        for prompt, response_group, adv_group in zip(prompts, responses, advantages):
            for i, (response, advantage) in enumerate(zip(response_group, adv_group)):
                # Concatenate prompt and response
                full_sequence = torch.cat([prompt, response[1:]], dim=0)

                # Get log probabilities from current policy
                policy_log_probs = self.policy.get_log_probs(
                    full_sequence.unsqueeze(0),
                    attention_mask,
                )

                # Get log probabilities from reference policy
                with torch.no_grad():
                    ref_log_probs = self.ref_policy.get_log_probs(
                        full_sequence.unsqueeze(0),
                        attention_mask,
                    )

                # Importance sampling ratio
                ratio = torch.exp(policy_log_probs - ref_log_probs)

                # Compute surrogate loss (similar to PPO)
                advantage_tensor = torch.tensor(advantage, device=self.device)
                surrogate1 = ratio * advantage_tensor
                surrogate2 = torch.clamp(
                    ratio,
                    1 - self.config.clip_range,
                    1 + self.config.clip_range
                ) * advantage_tensor

                # Policy loss (negative because we want to maximize)
                policy_loss = -torch.min(surrogate1, surrogate2).mean()
                policy_losses.append(policy_loss)

                # KL divergence loss
                kl_div = (policy_log_probs - ref_log_probs).sum(dim=-1).mean()
                kl_losses.append(kl_div)

                # Entropy bonus for exploration
                probs = torch.exp(policy_log_probs)
                entropy = -(probs * policy_log_probs).sum(dim=-1).mean()
                entropy_losses.append(-entropy)  # Negative because we maximize entropy

        # Combine losses
        total_policy_loss = torch.stack(policy_losses).mean()
        total_kl_loss = torch.stack(kl_losses).mean()
        total_entropy_loss = torch.stack(entropy_losses).mean()

        loss = (
            total_policy_loss +
            self.config.kl_coef * total_kl_loss +
            self.config.entropy_coef * total_entropy_loss
        )

        metrics = {
            "policy_loss": total_policy_loss.item(),
            "kl_loss": total_kl_loss.item(),
            "entropy_loss": total_entropy_loss.item(),
            "total_loss": loss.item(),
        }

        return loss, metrics

    def train_step(
        self,
        prompts: torch.Tensor,
        responses: List[torch.Tensor],
        advantages: List[List[float]],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Perform a single training step.

        Args:
            prompts: Input prompts
            responses: Generated responses
            advantages: Computed advantages
            attention_mask: Attention mask

        Returns:
            Training metrics
        """
        # Compute loss
        loss, metrics = self.compute_policy_loss(
            prompts, responses, advantages, attention_mask
        )

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(),
            self.config.max_grad_norm
        )

        # Update policy
        self.optimizer.step()

        return metrics

    def train_epoch(
        self,
        dataset: List[Dict[str, torch.Tensor]],
        reward_fn: callable,
    ) -> Dict[str, float]:
        """
        Train for one epoch over the dataset.

        Args:
            dataset: List of examples with 'prompt' and 'attention_mask'
            reward_fn: Function to compute rewards

        Returns:
            Average metrics over the epoch
        """
        self.policy.train()

        epoch_metrics = []
        num_batches = len(dataset) // self.config.batch_size

        for batch_idx in range(num_batches):
            # Get batch
            start_idx = batch_idx * self.config.batch_size
            end_idx = start_idx + self.config.batch_size
            batch = dataset[start_idx:end_idx]

            prompts = torch.stack([item['prompt'] for item in batch]).to(self.device)
            attention_masks = torch.stack([
                item.get('attention_mask')
                for item in batch
            ]).to(self.device) if batch[0].get('attention_mask') is not None else None

            # Generate responses
            responses = self.generate_responses(
                prompts,
                attention_masks,
                self.config.group_size
            )

            # Compute rewards
            rewards = self.compute_rewards(prompts, responses, reward_fn)

            # Compute advantages
            advantages = self.compute_advantages(rewards)

            # Multiple optimization epochs (like PPO)
            for _ in range(self.config.max_epochs):
                metrics = self.train_step(
                    prompts, responses, advantages, attention_masks
                )
                epoch_metrics.append(metrics)

        # Average metrics
        avg_metrics = {}
        for key in epoch_metrics[0].keys():
            avg_metrics[key] = np.mean([
                m[key] for m in epoch_metrics
            ])

        return avg_metrics


def create_grpo_trainer(
    model: nn.Module,
    ref_model: nn.Module,
    config: GRPOConfig,
    device: str = "cuda",
) -> GRPOOptimizer:
    """
    Helper function to create a GRPO trainer.

    Args:
        model: The model to train
        ref_model: Reference model (frozen copy)
        config: GRPO configuration
        device: Device to use

    Returns:
        GRPOOptimizer instance
    """
    vocab_size = model.config.vocab_size
    policy = GRPOPolicy(model, vocab_size)
    ref_policy = GRPOPolicy(ref_model, vocab_size)

    return GRPOOptimizer(policy, ref_policy, config, device)


# Example usage and reward functions
def example_reward_fn(prompt: torch.Tensor, response: torch.Tensor) -> float:
    """Example reward function."""
    # This is a placeholder - in practice, you would use:
    # - A learned reward model
    # - Human preferences
    # - Heuristic metrics (e.g., BLEU, ROUGE)
    # - Task-specific rewards

    # Simple example: reward longer responses
    return float(len(response)) * 0.1


def preference_based_reward_fn(
    reward_model: nn.Module,
    prompt: torch.Tensor,
    response: torch.Tensor,
) -> float:
    """
    Reward function using a learned reward model.

    Args:
        reward_model: Trained reward model
        prompt: Input prompt
        response: Generated response

    Returns:
        Reward score
    """
    with torch.no_grad():
        # Concatenate prompt and response
        full_sequence = torch.cat([prompt, response])
        # Get reward from model
        reward = reward_model(full_sequence)
    return float(reward)


if __name__ == "__main__":
    # Example usage
    from transformers import AutoModelForCausalLM

    # Load model
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    ref_model = AutoModelForCausalLM.from_pretrained("gpt2")

    # Create config
    config = GRPOConfig(
        learning_rate=1e-5,
        group_size=4,
        kl_coef=0.1,
    )

    # Create trainer
    trainer = create_grpo_trainer(model, ref_model, config)

    # Dummy dataset
    dataset = [
        {"prompt": torch.randint(0, 50257, (64,)), "attention_mask": torch.ones(64)}
        for _ in range(16)
    ]

    # Train
    metrics = trainer.train_epoch(dataset, example_reward_fn)
    print("Training metrics:", metrics)
