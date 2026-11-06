import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# ==========================================================
# SETTINGS
# ==========================================================

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ==========================================================
# PHYSICAL PARAMETERS
# ==========================================================

L = 1.0; alpha = 0.01; T_left = 100.0; T_right = 0.0; T_max = 10.0

# ==========================================================
# ANALYTICAL SOLUTION
# ==========================================================

def analytical_solution(x, t, n_terms=300):

    x = np.asarray(x)

    steady = 100.0 * (1.0 - x)

    transient = np.zeros_like(x, dtype=np.float64)

    for n in range(1, n_terms + 1):

        transient += (
            -200.0/(n*np.pi)
            * np.sin(n*np.pi*x)
            * np.exp(
                -alpha*(n*np.pi)**2*t
            )
        )

    return steady + transient

# ==========================================================
# CRANK NICOLSON SOLVER
# ==========================================================

def crank_nicolson():

    Nx = 101; Nt = 1000

    dx = L/(Nx-1)
    dt = T_max/(Nt-1)

    r = alpha*dt/dx**2

    x = np.linspace(0,L,Nx)
    t = np.linspace(0,T_max,Nt)

    u = np.zeros((Nt,Nx))

    u[:,0] = T_left
    u[:,-1] = T_right

    N = Nx-2

    A = np.zeros((N,N))
    B = np.zeros((N,N))

    for i in range(N):

        A[i,i] = 1+r
        B[i,i] = 1-r

        if i>0:
            A[i,i-1] = -r/2
            B[i,i-1] = r/2

        if i<N-1:
            A[i,i+1] = -r/2
            B[i,i+1] = r/2

    Ainv = np.linalg.inv(A)

    for n in range(Nt-1):

        rhs = B @ u[n,1:-1]

        rhs[0] += r*T_left

        u[n+1,1:-1] = Ainv @ rhs

    return x,t,u

print("Running finite difference solution ...")
x_fd,t_fd,u_fd = crank_nicolson()

# ==========================================================
# TRAINING DATA
# ==========================================================

N_data = 40
noise_std = 5.0

x_data = np.random.uniform(0,L,N_data)
t_data = np.random.uniform(0,T_max,N_data)

u_exact = analytical_solution(x_data,t_data)

u_noisy = u_exact + np.random.normal(
    0,
    noise_std,
    N_data
)

# ==========================================================
# TORCH DATA
# ==========================================================

x_train = torch.tensor(
    x_data,
    dtype=torch.float32,
    requires_grad=True
).reshape(-1,1).to(device)

t_train = torch.tensor(
    t_data,
    dtype=torch.float32,
    requires_grad=True
).reshape(-1,1).to(device)

u_train = torch.tensor(
    u_noisy,
    dtype=torch.float32
).reshape(-1,1).to(device)

# ==========================================================
# COLLOCATION POINTS
# ==========================================================

N_colloc = 10000

x_c = torch.tensor(
    np.random.uniform(0,L,N_colloc),
    dtype=torch.float32,
    requires_grad=True
).reshape(-1,1).to(device)

t_c = torch.tensor(
    np.random.uniform(0,T_max,N_colloc),
    dtype=torch.float32,
    requires_grad=True
).reshape(-1,1).to(device)

# ==========================================================
# NETWORK
# ==========================================================

class BaseNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(2,64),
            nn.Tanh(),

            nn.Linear(64,64),
            nn.Tanh(),

            nn.Linear(64,64),
            nn.Tanh(),

            nn.Linear(64,1)
        )

    def forward(self,x,t):

        x = x/L
        t = t/T_max

        X = torch.cat([x,t],dim=1)

        return self.net(X)

# ==========================================================
# STANDARD NN
# ==========================================================

class StandardNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = BaseNet()

    def forward(self,x,t):
        return self.net(x,t)

    def loss(self,x,t,u):
        return ((self(x,t)-u)**2).mean()

# ==========================================================
# PINN
# ==========================================================

class PINN(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = BaseNet()

    def forward(self,x,t):
        return self.net(x,t)

    def residual(self,x,t):

        u = self(x,t)

        u_t = torch.autograd.grad(
            u,t,
            grad_outputs=torch.ones_like(u),
            create_graph=True
        )[0]

        u_x = torch.autograd.grad(
            u,x,
            grad_outputs=torch.ones_like(u),
            create_graph=True
        )[0]

        u_xx = torch.autograd.grad(
            u_x,x,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True
        )[0]

        return u_t - alpha*u_xx

    def loss(self):

        pred = self(x_train,t_train)

        loss_data = ((pred-u_train)**2).mean()

        loss_pde = (self.residual(x_c,t_c)**2).mean()

        # BC

        t_bc = torch.linspace(
            0,T_max,200,
            device=device
        ).reshape(-1,1)

        left = self(
            torch.zeros_like(t_bc),
            t_bc
        )

        right = self(
            torch.ones_like(t_bc),
            t_bc
        )

        loss_bc = (
            ((left-T_left)**2).mean()
            +
            ((right-T_right)**2).mean()
        )

        # IC

        x_ic = torch.linspace(
            0,L,200,
            device=device
        ).reshape(-1,1)

        ic = self(
            x_ic,
            torch.zeros_like(x_ic)
        )

        loss_ic = (
            (ic**2).mean()
        )

        total = (
            loss_data
            +100*loss_pde
            +50*loss_bc
            +50*loss_ic
        )

        return total,loss_data,loss_pde

# ==========================================================
# TRAIN
# ==========================================================

nn_model = StandardNN().to(device)
pinn_model = PINN().to(device)

opt_nn = optim.Adam(
    nn_model.parameters(),
    lr=1e-3
)

opt_pinn = optim.Adam(
    pinn_model.parameters(),
    lr=1e-3
)

epochs = 5000

loss_nn_hist=[]
loss_pinn_hist=[]

print("\nTraining Standard NN")

for ep in range(epochs):

    opt_nn.zero_grad()

    loss = nn_model.loss(
        x_train,
        t_train,
        u_train
    )

    loss.backward()

    opt_nn.step()

    loss_nn_hist.append(loss.item())

    if ep%500==0:
        print(ep,loss.item())

print("\nTraining PINN")

for ep in range(epochs):

    opt_pinn.zero_grad()

    loss,dloss,ploss = pinn_model.loss()

    loss.backward()

    opt_pinn.step()

    loss_pinn_hist.append(loss.item())

    if ep%500==0:
        print(
            ep,
            "Total=",loss.item(),
            "Data=",dloss.item(),
            "PDE=",ploss.item()
        )

# ==========================================================
# COMPARISON AT t = 10s
# ==========================================================

x_plot = np.linspace(0,L,200)

u_exact = analytical_solution(
    x_plot,
    T_max
)

fd_index = -1
u_fd_final = np.interp(
    x_plot,
    x_fd,
    u_fd[fd_index]
)

x_tensor = torch.tensor(
    x_plot,
    dtype=torch.float32
).reshape(-1,1).to(device)

t_tensor = torch.full_like(
    x_tensor,
    T_max
)

with torch.no_grad():

    u_nn = nn_model(
        x_tensor,
        t_tensor
    ).cpu().numpy().flatten()

    u_pinn = pinn_model(
        x_tensor,
        t_tensor
    ).cpu().numpy().flatten()

# ==========================================================
# PLOT
# ==========================================================

plt.figure(figsize=(10,6))

plt.plot(
    x_plot,
    u_exact,
    'k',
    lw=3,
    label='Analytical'
)

plt.plot(
    x_plot,
    u_fd_final,
    'g--',
    lw=2,
    label='Finite Difference'
)

plt.plot(
    x_plot,
    u_nn,
    'b:',
    lw=2,
    label='NN'
)

plt.plot(
    x_plot,
    u_pinn,
    'r-.',
    lw=2,
    label='PINN'
)

plt.xlabel("x")
plt.ylabel("Temperature")
plt.title("Temperature Distribution at t=10s")
plt.grid(True)
plt.legend()
plt.show()

# ==========================================================
# LOSS
# ==========================================================

plt.figure(figsize=(10,6))

plt.semilogy(loss_nn_hist,label="NN")

plt.semilogy(loss_pinn_hist,label="PINN")

plt.legend()
plt.grid(True)
plt.title("Training Loss")
plt.show()

# ==========================================================
# ERRORS
# ==========================================================

mse_nn = np.mean(
    (u_nn-u_exact)**2
)

mse_pinn = np.mean(
    (u_pinn-u_exact)**2
)

mse_fd = np.mean(
    (u_fd_final-u_exact)**2
)

print("\n==========================")

print("Finite Difference MSE :", mse_fd)

print("NN MSE :", mse_nn)

print("PINN MSE :", mse_pinn)

print("==========================")