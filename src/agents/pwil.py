from beartype import beartype
import torch


class PWILRewarder:

    @beartype
    def __init__(self,
                 vectorized_expert_atoms: torch.Tensor,
                 device: torch.device,
                 input_mode: str,
                 ob_shape: tuple[int, ...],
                 ac_shape: tuple[int, ...],
                 horizon: int,
                 alpha: float = 5.0,
                 beta: float = 5.0):

        vectorized_expert_atoms = vectorized_expert_atoms.clone()
        self.avg_ = vectorized_expert_atoms.mean(
            dim=0, keepdim=True)
        self.std_ = vectorized_expert_atoms.std(
            dim=0, keepdim=True, unbiased=False) + 1e-6
        self.vectorized_expert_atoms = (vectorized_expert_atoms - self.avg_) / self.std_

        self.device = device
        self.input_mode = input_mode
        self.ob_dim = ob_shape[-1]
        self.ac_dim = ac_shape[-1]
        self.horizon = horizon
        self.alpha = alpha
        self.beta = beta

        tot_dim = self.ob_dim
        match self.input_mode:
            case "sa":
                tot_dim += self.ac_dim
            case "ss":
                tot_dim += self.ob_dim
            case _:
                raise ValueError("invalid input mode")

        # equation 6 in PWIL paper
        self.reward_sigma = self.beta * self.horizon / torch.sqrt(
            torch.tensor(tot_dim, dtype=torch.float, device=self.device),
        )

    @beartype
    def reset(self):
        """Reset expert weights and atoms for Algorithm 1"""
        # reset expert atoms (original is normalized)
        self.e_atoms = self.vectorized_expert_atoms.clone()
        # reset expert weights
        num_e_atoms = self.e_atoms.size(0)
        self.e_weights = torch.ones(num_e_atoms, device=self.device) / num_e_atoms

    @beartype
    def compute_reward(self,
                       state: torch.Tensor,
                       action: torch.Tensor,
                       next_state: torch.Tensor) -> torch.Tensor:
        """Apply Algorithm 1 from the PWIL paper.
        Careful: this function takes inputs from non-vectorized envs,
        and does not treat batches. Also, since the loop is dynamic,
        do not encapsulate this function in a CudaGraphModule.
        """
        with torch.no_grad():
            # prepare agent atom
            p_atom = state
            match self.input_mode:
                case "sa":
                    p_atom = torch.cat([state, action], dim=-1)  # dim=0
                case "ss":
                    p_atom = torch.cat([state, next_state], dim=-1)  # dim=0
                case _:
                    raise ValueError("invalid input mode")

            # normalize atoms
            p_atom = (p_atom - self.avg_) / self.std_

            # initialize cost and weight
            cost = 0.0
            weight = 1.0 / self.horizon - 1e-6

            # compute distances to expert atoms
            norms = torch.norm(self.e_atoms - p_atom, dim=1)

            while weight > 0 and len(norms) > 0:
                # find the closest expert atom
                argmin = torch.argmin(norms)
                e_weight = self.e_weights[argmin]

                if weight >= e_weight:
                    weight -= e_weight.item()
                    cost += e_weight.item() * norms[argmin].item()

                    # remove the argmin dim from every collection
                    indices = torch.arange(norms.size(0), device=self.device)
                    norms = norms[indices != argmin]
                    self.e_weights = self.e_weights[indices != argmin]
                    self.e_atoms = self.e_atoms[indices != argmin]
                else:
                    cost += weight * norms[argmin].item()
                    self.e_weights[argmin] -= weight
                    weight = 0

            # compute and return the reward
            return self.alpha * torch.exp(-self.reward_sigma * cost)  # equation 6 in PWIL paper
