"""Training-step orchestration. Feedback is measured around the actual optimizer.step."""
from contextlib import nullcontext
import torch
from diffusion_policy.common.pytorch_util import dict_apply
from .losses import clean_actions, encode_condition, epsilon_mse, feedback_kl
from .sampler import sample_timesteps, reinforce_loss, DeltaQueue


def _single_batch(batch, index):
    return dict_apply(batch, lambda x: x[index:index + 1])


class AdaptiveStepEngine:
    def __init__(self, policy, optimizer, actor=None, actor_optimizer=None,
                 total_timesteps=100, accumulation=1, update_every=40,
                 queue_size=5, n_select=3, ent_coef=0.01, probe_chunk=8):
        if accumulation < 1 or update_every < 1:
            raise ValueError("accumulation and update_every must be positive")
        if (actor is None) != (actor_optimizer is None):
            raise ValueError("actor and its optimizer must be supplied together")
        self.policy = policy
        self.optimizer = optimizer
        self.actor = actor
        self.actor_optimizer = actor_optimizer
        self.total_timesteps = total_timesteps
        self.accumulation = accumulation
        self.update_every = update_every
        self.ent_coef = ent_coef
        self.probe_chunk = probe_chunk
        self.queue = DeltaQueue(total_timesteps, queue_size, n_select) if actor else None
        self.optimizer_steps = 0
        self.micro_in_window = 0
        self.pending = []
        self.optimizer.zero_grad(set_to_none=True)
        if actor_optimizer:
            actor_optimizer.zero_grad(set_to_none=True)

    @property
    def adaptive(self):
        return self.actor is not None

    @property
    def update_due(self):
        # Official i=0 update, then every f_S model updates.
        return self.adaptive and self.optimizer_steps % self.update_every == 0

    def _selected_values(self, batch, noise):
        actions = clean_actions(self.policy, batch)
        condition = encode_condition(self.policy, batch)
        values = []
        for selected_t in self.queue.selected:
            t = torch.full((actions.shape[0],), selected_t,
                           device=actions.device, dtype=torch.long)
            values.append(feedback_kl(self.policy, actions, condition, t, noise))
        return torch.stack(values, dim=1)

    def _full_values(self, batch, probe_noise):
        actions = clean_actions(self.policy, batch)
        condition = encode_condition(self.policy, batch)
        values = []
        for start in range(0, self.total_timesteps, self.probe_chunk):
            end = min(start + self.probe_chunk, self.total_timesteps)
            n = end - start
            t = torch.arange(start, end, device=actions.device)
            values.append(feedback_kl(
                self.policy, actions.expand(n, -1, -1),
                condition.expand(n, -1), t, probe_noise[start:end]))
        return torch.cat(values)

    def process(self, batch, last_microbatch=False):
        self.policy.train()
        if not any(p.requires_grad for p in self.policy.obs_encoder.parameters()):
            self.policy.obs_encoder.eval()
        self.micro_in_window += 1
        if self.adaptive:
            context = nullcontext() if self.update_due else torch.no_grad()
            actions = clean_actions(self.policy, batch)
            with context:
                draw = sample_timesteps(self.actor, actions.detach(), self.total_timesteps)
            condition = encode_condition(self.policy, batch)
            noise = torch.randn_like(actions)
            raw_loss = epsilon_mse(self.policy, actions, condition, draw.timesteps, noise).mean()
            if self.update_due:
                self.pending.append((batch, draw))
        else:
            # Exact original Vanilla loss path; no sampler or feedback.
            raw_loss = self.policy.compute_loss(batch)
        (raw_loss / self.accumulation).backward()
        should_step = self.micro_in_window == self.accumulation or last_microbatch
        result = dict(loss=float(raw_loss.detach()), optimizer_stepped=False,
                      sampler_updated=False, optimizer_steps=self.optimizer_steps)
        if not should_step:
            return result
        if self.micro_in_window < self.accumulation:
            # Final short accumulation window still represents the mean of its actual microbatches.
            scale = self.accumulation / self.micro_in_window
            for parameter in self.policy.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(scale)

        if self.update_due:
            # Official implementation redraws noise for before and after evaluations.
            # All microbatches in the accumulation window belong to this update.
            evaluation = []
            with torch.no_grad():
                self.policy.eval()  # deterministic center crop and frozen normalization
                for pending_batch, draw in self.pending:
                    action = clean_actions(self.policy, pending_batch)
                    noise_eval = torch.randn_like(action)
                    before = self._selected_values(pending_batch, noise_eval)
                    evaluation.append((pending_batch, draw, before))
                probe_batch_source = self.pending[-1][0]
                probe_index = int(torch.randint(
                    probe_batch_source["action"].shape[0], (1,),
                    device=probe_batch_source["action"].device).item())
                probe_batch = _single_batch(probe_batch_source, probe_index)
                probe_action = clean_actions(self.policy, probe_batch)
                probe_noise = torch.randn((self.total_timesteps,) + tuple(probe_action.shape[1:]),
                                           device=probe_action.device, dtype=probe_action.dtype)
                full_before = self._full_values(probe_batch, probe_noise)

        # The parameter state changes exactly here; feedback never spans a skipped optimizer step.
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.optimizer_steps += 1
        result.update(optimizer_stepped=True, optimizer_steps=self.optimizer_steps)

        if self.update_due_after_step():
            with torch.no_grad():
                per_sample_delta = []
                for pending_batch, draw, before in evaluation:
                    action = clean_actions(self.policy, pending_batch)
                    after = self._selected_values(pending_batch, torch.randn_like(action))
                    per_sample_delta.append(before - after)
                full_after = self._full_values(probe_batch, torch.randn_like(probe_noise))
                full_delta = full_before - full_after
            self.queue.add(full_delta)
            log_probs = torch.cat([item[1].log_prob.reshape(-1) for item in evaluation])
            entropies = torch.cat([item[1].entropy.reshape(-1) for item in evaluation])
            deltas = torch.cat(per_sample_delta, dim=0)
            loss_pi = reinforce_loss(log_probs, entropies, deltas, self.ent_coef)
            loss_pi.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()
            self.actor_optimizer.zero_grad(set_to_none=True)
            self.queue.select()
            result.update(sampler_updated=True, actor_loss=float(loss_pi.detach()),
                          kl_delta_mean=float(deltas.sum(dim=1).mean()),
                          selected=list(self.queue.selected))
            self.policy.train()

        self.pending.clear()
        self.micro_in_window = 0
        return result

    def update_due_after_step(self):
        # Called after optimizer_steps increments.
        return self.adaptive and (self.optimizer_steps - 1) % self.update_every == 0

    def state_dict(self):
        return dict(optimizer_steps=self.optimizer_steps,
                    queue=self.queue.state_dict() if self.queue else None)

    def load_state_dict(self, state):
        self.optimizer_steps = int(state["optimizer_steps"])
        if self.queue and state["queue"] is not None:
            self.queue.load_state_dict(state["queue"])
