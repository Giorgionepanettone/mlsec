import torch
from tqdm import tqdm


class Attack:
    """ base class implementing different attacks """
    def __init__(self, model, targeted=False):
        self.target_model = model
        self.targeted = targeted

    def perturb(self, x, y, log=True):
        pass

    def perturb_all(self, dataloader, device):
        """ compute perturbation over all the samples in dataloader. Returns computed perturbations """
        deltas = []
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            deltas.append(self.perturb(x_batch, y_batch))
        return torch.cat(deltas, dim=0).detach().cpu()


class FGSM(Attack):
    def __init__(self, model, epsilon, targeted=False):
        super().__init__(model, targeted)
        self.type = 'fgsm'
        self.targeted = False
        self.epsilon = epsilon

    def perturb(self, x, y, log=True):
        """ computes adversarial example using FGSM. Returns computed perturbation"""
        # initialize delta (the perturbation)
        perturbation = torch.rand(size = x.shape[0])
        # predict, compute loss and gradients
        self.target_model.train()
        pred = self.target_model.forward(x)
        loss = self.target_model.loss_fn_avg(pred, y)
        self.target_model.backward()
                # TODO
        # compute FGSM adv example
        # TODO
        # fix out of bounds and compute final perturbation
        # TODO
        return perturbation

class IFGSM(FGSM):
    def __init__(self, model, epsilon, alpha, num_iter, targeted=False):
        super().__init__(model, epsilon, targeted=targeted)
        self.type = 'ifgsm'
        self.alpha = alpha
        self.num_iter = num_iter

    def perturb(self, x, y, log=True):
        """ computes adversarial example using IFGSM. Returns computed perturbation"""
        # initialize adv_data
        # TODO
        iterator = range(self.num_iter)
        if log:
            # add logging through TQDM
            iterator = tqdm(iterator, desc='Generating IFGSM perturbation')
        for _ in iterator:
            # request gradients for adv examples
            adv_data.requires_grad = True
            # predict, compute loss and gradients
            # TODO
            # compute IFGSM step, project perturbation
            # TODO
        delta = adv_data.detach() - x
        return delta


class PGD(IFGSM):
    def __init__(self, model, epsilon, alpha, num_iter, restarts, targeted=False):
        super().__init__(model, epsilon, alpha, num_iter, targeted=targeted)
        self.type = 'pgd'
        self.restarts = restarts

    def perturb(self, x, y, log=True):
        """ computes adversarial example using PGD. Returns computed perturbation"""
        # data structures to keep track of max loss found and max perturbation to achieve the max loss
        max_loss = torch.zeros(y.shape[0]).to(y.device)
        max_delta = torch.zeros_like(x)

        iterator = range(self.restarts)
        if log:
            iterator = tqdm(iterator, desc='Generating PGD perturbation')
        for _ in iterator:
            # randomly initialize perturbation. variable name: delta
            # TODO
            # initialize starting adversarial data with perturbation. Remember the bounds [0,1]
            # TODO
            # compute adversarial examples - same as IFGSM
            # TODO
            # compute loss for current adversarial examples
            # TODO
            # update max_loss if a better adv. example has been found for a given sample. Update the corresponding max_delta
            # TODO

        return max_delta

