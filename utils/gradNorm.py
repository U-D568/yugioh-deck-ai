import torch

class GradNorm:
    def __init__(self, num_task, lr0=1e-4, alpha=0.1):
        self.weights = torch.ones(num_task)
        self.lr0 = lr0
        params = torch.nn.Parameter(self.weights)
        self.optimizer = torch.optim.Adam([params], lr=lr0)
        self.init_loss = None
        self.T = self.weights.sum().detach()
        self.alpha = alpha

    def __call__(self, loss, layer):
        if self.init_loss is None:
            self.init_loss = loss.detach()

        # calculate weighted loss for entire model
        weighted_loss = self.weights @ loss

        gw = []
        for i in range(len(loss)):
            grad_weight, grad_bias = torch.autograd.grad(
                self.weights[i] * loss[i],
                layer.parameters(),
                retain_graph=True,
                create_graph=True,
            )
            gw.append(torch.norm(grad_weight))
        gw = torch.stack(gw)
        gw_avg = gw.mean().detach()

        loss_ratio = loss.detach() / self.init_loss
        rt = loss_ratio / loss_ratio.mean()

        lgrad = torch.abs(gw - gw_avg * rt ** self.alpha).sum()
        self.optimizer.zero_grad()
        lgrad.backward()
        self.optimizer.step()

        # renormalize weights
        self.weights = (self.weights / self.weights.sum() * T).detach()
        self.weights = torch.nn.Parameter(self.weights)
        self.optimizer = torch.optim.Adam([self.weights], lr=self.lr0)

        return weighted_loss