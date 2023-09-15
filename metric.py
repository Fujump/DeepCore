import torch
import torch.nn as nn
import numpy as np

DEVICE = "cuda:7" if torch.cuda.is_available() else "cpu"

# class Uncertainty(object):
#     def __init__(self, model, selection_method="Entropy"):
#         self.model = model
#         self.selection_method = selection_method

#     def rank_uncertainty(self, data_loader):
#         with torch.no_grad():
#             scores = np.array([])
#             for input,_ in data_loader:
#                 if self.selection_method == "LeastConfidence":
#                         scores = np.append(scores, self.model(input.to(DEVICE)).max(axis=1).values.cpu().numpy())
#                 elif self.selection_method == "Entropy":
#                         preds = torch.nn.functional.softmax(self.model(input.to(DEVICE)), dim=1).cpu().numpy()
#                         scores = np.append(scores, (np.log(preds + 1e-6) * preds).sum(axis=1))
#                 elif self.selection_method == 'Margin':
#                         preds = torch.nn.functional.softmax(self.model(input.to(DEVICE)), dim=1)
#                         preds_argmax = torch.argmax(preds, dim=1)
#                         max_preds = preds[torch.ones(preds.shape[0], dtype=bool), preds_argmax].clone()
#                         preds[torch.ones(preds.shape[0], dtype=bool), preds_argmax] = -1.0
#                         preds_sub_argmax = torch.argmax(preds, dim=1)
#                         scores = np.append(scores, (max_preds - preds[
#                             torch.ones(preds.shape[0], dtype=bool), preds_sub_argmax]).cpu().numpy())
#         return scores 

# def calculate_uncertainty_score(model, data_loader, selection_method="Entropy"):
#     uncertainty = Uncertainty(model, selection_method)
#     scores = uncertainty.rank_uncertainty(data_loader)
#     return scores

# class Forgetting(object):
#     def __init__(self, dst_train, args, fraction=0.5, random_seed=None, epochs=200, specific_model=None, balance=True,
#                  dst_test=None, **kwargs):
#         # super().__init__(dst_train, args, fraction, random_seed, epochs, specific_model=specific_model,
#         #                  dst_test=dst_test)
#         self.args=args
#         self.n_train = len(dst_train)
#         self.balance = balance
#         self.criterion = nn.CrossEntropyLoss().to(self.args.device)

#     def calculate_importance_scores(self, model, data_loader):
#         model.eval()
#         importance_scores = torch.zeros(self.n_train, requires_grad=False).to(self.args.device)

#         with torch.no_grad():
#             for batch_idx, (inputs, targets) in enumerate(data_loader):
#                 inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)

#                 outputs = model(inputs)
#                 loss = self.criterion(outputs, targets)
#                 importance_scores[batch_idx * data_loader.batch_size:(batch_idx + 1) * data_loader.batch_size] = loss

#         return importance_scores.cpu().numpy()

# def get_last_layer(model):
#         last_layer = None
#         for child in model.children():
#             # Check if the child is a fully connected (linear) layer
#             if isinstance(child, torch.nn.Linear):
#                 last_layer = child
#         return last_layer

# class GraNd(object):
#     def __init__(self, dst_train, args, fraction=0.5, random_seed=None, epochs=200, repeat=10,
#                  specific_model=None, balance=False, **kwargs):
#         # super().__init__(dst_train, args, fraction, random_seed, epochs, specific_model)
#         self.dst_train=dst_train
#         self.args=args
#         self.epochs = epochs
#         self.n_train = len(dst_train)
#         self.coreset_size = round(self.n_train * fraction)
#         self.specific_model = specific_model
#         self.repeat = repeat

#         self.balance = balance
#         self.criterion=nn.CrossEntropyLoss().to(self.args.device)

#     def calculate_importance_scores_gradients(self, model, data_loader):
#         # model.eval()
#         # # importance_scores = torch.zeros(self.n_train, requires_grad=False).to(self.args.device)

#         # self.norm_matrix = torch.zeros([self.n_train, self.repeat], requires_grad=False).to(self.args.device)

#         # embedding_dim = get_last_layer(model).in_features
#         # # batch_loader = torch.utils.data.DataLoader(
#         #     # self.dst_train, batch_size=self.args.selection_batch, num_workers=self.args.workers)
#         # sample_num = self.n_train

#         # for i, (input, targets) in enumerate(data_loader):
#         #     # self.model_optimizer.zero_grad()
#         #     outputs = model(input.to(self.args.device))
#         #     loss = self.criterion(outputs.requires_grad_(True),
#         #                           targets.to(self.args.device)).sum()
#         #     batch_num = targets.shape[0]
#         #     with torch.no_grad():
#         #         bias_parameters_grads = torch.autograd.grad(loss, outputs)[0]
#         #         self.norm_matrix[i * self.args.selection_batch:min((i + 1) * self.args.selection_batch, sample_num),
#         #         self.cur_repeat] = torch.norm(torch.cat([bias_parameters_grads, (
#         #                 model.embedding_recorder.embedding.view(batch_num, 1, embedding_dim).repeat(1,
#         #                                      self.args.num_classes, 1) * bias_parameters_grads.view(
#         #                                      batch_num, self.args.num_classes, 1).repeat(1, 1, embedding_dim)).
#         #                                      view(batch_num, -1)], dim=1), dim=1, p=2)
        
#         # self.norm_mean = torch.mean(self.norm_matrix, dim=1).cpu().detach().numpy()
#         # return self.norm_mean

#         model.eval()
#         importance_scores = torch.zeros(self.n_train, requires_grad=False).to(self.args.device)

#         for param in model.parameters():
#             param.requires_grad = True

#         with torch.no_grad():
#             for batch_idx, (inputs, targets) in enumerate(data_loader):
#                 inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
#                 inputs.requires_grad_()  # Set requires_grad to True
#                 # Forward pass
#                 # outputs = model(inputs).to(self.args.device)
#                 # print(outputs.shape)
#                 # print(targets.shape)

#                 # # Calculate the loss
#                 # losses = nn.CrossEntropyLoss(reduction='none')(outputs, targets)
#                 # losses.requires_grad_()
#                 # print(losses.requires_grad)
#                 # print(inputs.requires_grad)
#                 # print(losses.shape)

#                 # Calculate the gradients of the loss with respect to the inputs
#                 # grads = torch.autograd.grad(losses, inputs)[0]
#                 # norms = torch.norm(grads, p=2, dim=(1, 2, 3))
#                 for idx in range(inputs.shape[0]):
#                     output=model(inputs[idx:idx+1])
#                     loss=nn.CrossEntropyLoss(reduction='none')(output,targets[idx:idx+1])
#                     loss.requires_grad_()
#                     inputs[idx:idx+1].requires_grad_()
#                     # loss = losses[idx]
#                     # cur=inputs[idx:idx+1].clone().detach().requires_grad_(True)
#                     # print(model)
#                     # print(outputs.grad_fn)
#                     print(loss)
#                     print(inputs[idx:idx+1].shape)
#                     grad = torch.autograd.grad(loss,inputs[idx:idx+1] ,allow_unused=True)[0]
#                     print(grad)
#                     importance_scores[batch_idx * data_loader.batch_size + idx] = torch.norm(grad, p=2)

#                 # Calculate the norm of the gradients for each data point
#                 # importance_scores[batch_idx * data_loader.batch_size:(batch_idx + 1) * data_loader.batch_size] = torch.norm(grads, p=2, dim=(1, 2, 3))
#                 # importance_scores[batch_idx * data_loader.batch_size:(batch_idx + 1) * data_loader.batch_size] = norms
                
#         return importance_scores
#         # with torch.no_grad():
#         #     for batch_idx, (inputs, targets) in enumerate(data_loader):
#         #         inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)

#         #         # Forward pass
#         #         outputs = model(inputs)

#         #         # Calculate the loss
#         #         loss = self.criterion(outputs, targets)

#         #         # Calculate the gradients of the loss with respect to the model parameters
#         #         grads = torch.autograd.grad(loss.sum(), model.parameters())

#         #         # Calculate the norm of the gradients for each data point
#         #         for i, p in enumerate(model.parameters()):
#         #             if p.grad is not None:
#         #                 importance_scores[batch_idx * data_loader.batch_size:(batch_idx + 1) * data_loader.batch_size] += torch.norm(p.grad, p=2, dim=(1, 2, 3))

#         # return importance_scores.cpu().numpy()

def calculate_importance_scores(model, method):
    # Assuming 'model' is already trained
    # You can train the model before passing it to this function

    # Create an instance of the Cal class
    cal_instance = method # Pass appropriate 'args' if needed

    # Set the model for the Cal instance
    cal_instance.model = model

    # Calculate the importance scores using the 'select' method
    result = cal_instance.select()
    print(len(result['indices']))

    return result['scores']