import torch


class SAM(torch.optim.Optimizer):
    supports_sharpness_aware_steps = True

    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        if rho <= 0.0:
            raise ValueError("rho must be > 0")
        if not callable(base_optimizer):
            raise TypeError("base_optimizer must be callable")
        defaults = dict(rho=float(rho), adaptive=bool(adaptive), **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        scale_denom = grad_norm.clamp(min=1e-12)
        for group in self.param_groups:
            scale = group["rho"] / scale_denom
            for parameter in group["params"]:
                grad = parameter.grad
                if grad is None:
                    continue
                state = self.state[parameter]
                state["old_p"] = parameter.detach().clone()
                perturbation = grad * scale.to(parameter)
                if group["adaptive"]:
                    perturbation = parameter.pow(2) * perturbation
                parameter.add_(perturbation)
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for parameter in group["params"]:
                state = self.state[parameter]
                old_parameter = state.pop("old_p", None)
                if old_parameter is not None:
                    parameter.copy_(old_parameter)
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def step(self, closure=None):
        if closure is None:
            raise RuntimeError("SAM.step requires a closure")
        with torch.enable_grad():
            loss = closure()
        self.first_step(zero_grad=True)
        with torch.enable_grad():
            closure()
        self.second_step()
        return loss

    def state_dict(self):
        return {
            "sam_state": super().state_dict(),
            "base_optimizer_state": self.base_optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict["sam_state"])
        self.base_optimizer.param_groups = self.param_groups
        self.base_optimizer.load_state_dict(state_dict["base_optimizer_state"])
        self.param_groups = self.base_optimizer.param_groups

    def _grad_norm(self):
        shared_device = None
        norms = []
        for group in self.param_groups:
            for parameter in group["params"]:
                grad = parameter.grad
                if grad is None:
                    continue
                if shared_device is None:
                    shared_device = parameter.device
                scaled_grad = grad
                if group["adaptive"]:
                    scaled_grad = parameter.abs() * scaled_grad
                norms.append(scaled_grad.norm(p=2).to(shared_device))
        if not norms:
            if shared_device is None:
                shared_device = torch.device("cpu")
            return torch.zeros((), device=shared_device)
        return torch.norm(torch.stack(norms), p=2)


class ASAM(SAM):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        if "adaptive" in kwargs:
            raise TypeError("ASAM does not accept adaptive; it is always enabled")
        super().__init__(
            params,
            base_optimizer,
            rho=rho,
            adaptive=True,
            **kwargs,
        )


class GSAM(SAM):
    def __init__(self, params, base_optimizer, rho=0.05, alpha=0.2, adaptive=False, eps=1e-12, **kwargs):
        if alpha < 0.0 or alpha > 1.0:
            raise ValueError("alpha must be between 0 and 1")
        if eps <= 0.0:
            raise ValueError("eps must be > 0")
        self.alpha = float(alpha)
        self.eps = float(eps)
        super().__init__(
            params,
            base_optimizer,
            rho=rho,
            adaptive=adaptive,
            **kwargs,
        )
        for group in self.param_groups:
            group["alpha"] = self.alpha
            group["eps"] = self.eps

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                self.state[parameter]["reference_grad"] = parameter.grad.detach().clone()
        super().first_step(zero_grad=zero_grad)

    @torch.no_grad()
    def on_after_second_backward(self):
        grad_norm = self._grad_norm()
        reference_norm = self._reference_grad_norm()
        if float(grad_norm) == 0.0 or float(reference_norm) == 0.0:
            return
        inner_product = self._gradient_inner_product()
        cosine = inner_product / (grad_norm * reference_norm).clamp(min=self.eps)
        for group in self.param_groups:
            alpha = group["alpha"]
            if alpha == 0.0:
                continue
            for parameter in group["params"]:
                grad = parameter.grad
                reference_grad = self.state[parameter].get("reference_grad")
                if grad is None or reference_grad is None:
                    continue
                vertical = reference_grad - cosine * reference_norm * grad / grad_norm.clamp(min=group["eps"])
                grad.add_(vertical, alpha=-alpha)

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        super().second_step(zero_grad=zero_grad)
        for group in self.param_groups:
            for parameter in group["params"]:
                self.state[parameter].pop("reference_grad", None)

    def step(self, closure=None):
        if closure is None:
            raise RuntimeError("GSAM.step requires a closure")
        with torch.enable_grad():
            loss = closure()
        self.first_step(zero_grad=True)
        with torch.enable_grad():
            closure()
        self.on_after_second_backward()
        self.second_step()
        return loss

    def _reference_grad_norm(self):
        shared_device = None
        norms = []
        for group in self.param_groups:
            for parameter in group["params"]:
                reference_grad = self.state[parameter].get("reference_grad")
                if reference_grad is None:
                    continue
                if shared_device is None:
                    shared_device = parameter.device
                norms.append(reference_grad.norm(p=2).to(shared_device))
        if not norms:
            if shared_device is None:
                shared_device = torch.device("cpu")
            return torch.zeros((), device=shared_device)
        return torch.norm(torch.stack(norms), p=2)

    def _gradient_inner_product(self):
        shared_device = None
        inner_product = None
        for group in self.param_groups:
            for parameter in group["params"]:
                grad = parameter.grad
                reference_grad = self.state[parameter].get("reference_grad")
                if grad is None or reference_grad is None:
                    continue
                if shared_device is None:
                    shared_device = parameter.device
                contribution = torch.sum(grad * reference_grad).to(shared_device)
                inner_product = contribution if inner_product is None else inner_product + contribution
        if inner_product is None:
            if shared_device is None:
                shared_device = torch.device("cpu")
            return torch.zeros((), device=shared_device)
        return inner_product


__all__ = ["ASAM", "GSAM", "SAM"]
