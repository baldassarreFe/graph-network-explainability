import torch
import torch.utils.data

torch.manual_seed(0)

x = torch.rand(10000, 5) * 2 - 1
y = (3 * x[:, 0] + 8 * torch.cos(3.14 * x[:, 2]) - 5 * torch.pow(x[:, 4], 2)).view(-1, 1)

ds = torch.utils.data.TensorDataset(x, y)
dl = torch.utils.data.DataLoader(ds, batch_size=1000, shuffle=True, pin_memory=True, num_workers=0)

model = torch.nn.Sequential(
    torch.nn.Linear(5, 16), torch.nn.ReLU(),
    torch.nn.Linear(16, 32), torch.nn.ReLU(),
    torch.nn.Linear(32, 16), torch.nn.ReLU(),
    torch.nn.Linear(16, 4), torch.nn.ReLU(),
    torch.nn.Linear(4, 1)
).to('cuda')

model.train()
opt = torch.optim.SGD(model.parameters(), lr=.01)
scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.7)

for e in range(600):
    for xs, ys in dl:
        xs = xs.to('cuda')
        ys = ys.to('cuda')

        out = model(xs)
        model.zero_grad()

        loss = torch.nn.functional.mse_loss(out, ys)
        loss.backward()

        opt.step()

    scheduler.step()
    if e % 50 == 0:
        print(e, loss.item(), opt.param_groups[0]['lr'])

torch.set_grad_enabled(False)
model.cpu()
model.eval()

print(y[:10].squeeze())
print(model(x[:10]).squeeze())

from relevance import forward_relevance, backward_relevance

ctx = {}

y = forward_relevance(model, x[:10], ctx=ctx)
print(y.squeeze())
print()
print(*ctx.items(), sep='\n\n')
print()

rel_out = y
print(rel_out)
print()
rel = backward_relevance(model, rel_out, ctx=ctx)
print(rel)
print(rel / rel_out)
print()
print(rel.abs().mean(dim=0))

print()
rel_out = torch.ones_like(y)
print(rel_out)
print()
rel = backward_relevance(model, rel_out, ctx=ctx)
print(rel)
print()
print(rel.mean(dim=0))

print(torch.allclose(rel_out.sum(dim=1), rel.sum(dim=1)))
